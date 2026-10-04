"""Local travel-vlog candidate selection, not an unattended final edit.

Run on small, orientation-corrected proxies. Outputs source-relative decisions
for human review; no source media is changed or uploaded.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from clipshow.detection.semantic import SemanticDetector

POSITIVE = [
    "a scenic turquoise alpine lake surrounded by majestic mountains",
    "a beautiful wide mountain landscape and forest in Banff",
    "a tranquil forest reflected in a turquoise mountain lake",
    "people paddling a canoe on a beautiful mountain lake",
    "a well composed travel photograph of a historic hotel or lively town street",
]
NEGATIVE = [
    "a hand or finger covering the camera lens",
    "a close up of legs, shoes, pavement or the ground",
    "a blurry shaky photograph with no clear subject",
    "the camera pointing at an empty ceiling or blank wall",
]


@dataclass
class Candidate:
    start: float
    end: float
    mean_semantic: float
    low_semantic: float
    quality_pass_fraction: float
    review_required: bool = True


def frame_quality(frame: np.ndarray) -> tuple[bool, dict]:
    """Conservative technical filter. Semantics, hands and roll need review.

    Sharpness is measured at a consistent 384 px width. Smooth sky/water is
    allowed; this is not an aesthetic score and never rewards camera motion.
    """
    if frame is None or frame.size == 0:
        raise ValueError("empty frame")
    height = max(1, round(frame.shape[0] * 384 / frame.shape[1]))
    gray = cv2.cvtColor(cv2.resize(frame, (384, height)), cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    black = float(np.mean(gray < 8))
    white = float(np.mean(gray > 248))
    contrast = float(np.std(gray))
    passed = black < 0.8 and white < 0.8 and contrast > 4 and sharpness > 3
    return passed, dict(
        sharpness=sharpness, black_fraction=black, white_fraction=white, contrast=contrast
    )


def select_windows(
    semantic, quality, time_step=0.1, duration=None, shot_seconds=6.0, threshold=0.62, limit=12
):
    """Rank FULL windows by mean and low quantile, without per-file rescaling.

    No padding or gap merging: neither can reintroduce a blocked frame.
    Quality failure vetoes the whole window. No forced 5-minute filler.
    """
    semantic = np.asarray(semantic, dtype=float)
    quality = np.asarray(quality, dtype=bool)
    if time_step <= 0 or shot_seconds <= 0 or limit < 0:
        raise ValueError("invalid selection parameters")
    if semantic.ndim != 1 or quality.shape != semantic.shape:
        raise ValueError("semantic and quality timelines must align")
    if not np.all(np.isfinite(semantic)):
        raise ValueError("non-finite semantic scores")
    duration = len(semantic) * time_step if duration is None else duration
    width = max(1, int(np.ceil(shot_seconds / time_step)))
    stride = max(1, round(1.0 / time_step))
    ranked = []
    for start in range(0, len(semantic) - width + 1, stride):
        end = start + width
        if end * time_step > duration + 1e-6:
            continue
        values = semantic[start:end]
        quality_fraction = float(np.mean(quality[start:end]))
        mean = float(np.mean(values))
        low = float(np.quantile(values, 0.1))
        if quality_fraction < 1.0 or mean < threshold or low < threshold - 0.12:
            continue
        ranked.append(Candidate(start * time_step, end * time_step, mean, low, quality_fraction))
    ranked.sort(key=lambda c: (c.mean_semantic, c.low_semantic), reverse=True)
    kept = []
    for candidate in ranked:
        if len(kept) >= limit:
            break
        if any(candidate.start < old.end and old.start < candidate.end for old in kept):
            continue
        kept.append(candidate)
    return sorted(kept, key=lambda c: c.start)


def analyze(path, detector, shot_seconds=6.0, threshold=0.62, limit=12):
    result = detector.detect(str(path))
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    interval = max(1, int(fps / 2))
    measurements = []
    quality = np.zeros(len(result.scores), dtype=bool)
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % interval == 0:
                passed, metrics = frame_quality(frame)
                start = int(index / fps / result.time_step)
                end = int((index + interval) / fps / result.time_step)
                quality[start : min(end, len(quality))] = passed
                measurements.append(dict(time=index / fps, passed=passed, **metrics))
            index += 1
    finally:
        cap.release()
    candidates = select_windows(
        result.scores, quality, result.time_step, duration, shot_seconds, threshold, limit
    )
    return dict(
        source=str(path),
        duration=duration,
        candidates=[asdict(c) for c in candidates],
        semantic_scores=result.scores.tolist(),
        time_step=result.time_step,
        quality_measurements=measurements,
        review_required=True,
        limitations=[
            "CLIP similarity is not probability or location identification",
            "2 FPS sampling cannot prove absence of brief defects",
            "Review whole shots for shake, obstruction, roll and crop",
            "Dialogue selection and daily narrative require separate review",
        ],
    )


def photo_capture_datetime(path):
    """Prefer EXIF capture time; DJI filename is an explicit fallback, not mtime."""
    with Image.open(path) as image:
        exif = image.getexif()
        original = exif.get(36867) or exif.get_ifd(34665).get(36867)
    if original:
        try:
            return datetime.strptime(str(original), "%Y:%m:%d %H:%M:%S"), "EXIF"
        except ValueError:
            pass
    match = re.match(r"DJI_(\d{14})_", path.name, re.IGNORECASE)
    if match:
        try:
            return datetime.strptime(match[1], "%Y%m%d%H%M%S"), "DJI filename"
        except ValueError:
            pass
    return None, "unknown"


def photo_capture_day(path):
    captured, basis = photo_capture_datetime(path)
    return (captured.date() if captured else None), basis


def analyze_photos(directory, day, detector, threshold=0.62, prompts=None, negative_prompts=None):
    """Return reviewable stills for the existing photo-to-video renderer.

    These are NOT exported movies. Ken Burns motion, crop and transitions are
    performed by the downstream editor after approval. Missing dates are skipped.
    """
    capture_day = datetime.strptime(day, "%Y-%m-%d").date()
    model = detector._model or detector._load_model()
    positive = model.get_text_embeddings(prompts or POSITIVE)
    negative = model.get_text_embeddings(negative_prompts or NEGATIVE)
    accepted = []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        measured_day, basis = photo_capture_day(path)
        if measured_day != capture_day:
            continue
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
            image.thumbnail((1024, 1024))
            passed, metrics = frame_quality(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
            if not passed:
                continue
            embedding = model.get_image_embeddings([image])
            margin = float(np.max(embedding @ positive.T) - np.max(embedding @ negative.T))
            # Same absolute mapping as the semantic video detector.
            from clipshow.detection.semantic import _SIGMOID_CENTER, _SIGMOID_SCALE

            score = float(1 / (1 + np.exp(-_SIGMOID_SCALE * (margin - _SIGMOID_CENTER))))
        if score >= threshold:
            accepted.append(
                dict(
                    kind="photo",
                    source=str(path),
                    day=day,
                    date_basis=basis,
                    semantic_score=score,
                    suggested_seconds=4.0,
                    quality_metrics=metrics,
                    review_required=True,
                )
            )
    return accepted


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="*", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--shot-seconds", type=float, default=6)
    ap.add_argument("--threshold", type=float, default=0.62)
    ap.add_argument("--limit-per-file", type=int, default=12)
    ap.add_argument("--photo-dir", type=Path)
    ap.add_argument("--day", help="Photo capture date, YYYY-MM-DD")
    args = ap.parse_args()
    if not args.inputs and not args.photo_dir:
        ap.error("provide videos and/or --photo-dir")
    if args.photo_dir and not args.day:
        ap.error("--photo-dir requires --day; never mix undated or other-day photos")
    detector = SemanticDetector(prompts=POSITIVE, negative_prompts=NEGATIVE)
    reports = []
    for path in args.inputs:
        reports.append(
            analyze(path, detector, args.shot_seconds, args.threshold, args.limit_per_file)
        )
        print(f"{path.name}: {len(reports[-1]['candidates'])} review candidates", flush=True)
    if args.photo_dir:
        photos = analyze_photos(args.photo_dir, args.day, detector, args.threshold)
        reports.extend(photos)
        print(f"{args.day}: {len(photos)} photo review candidates", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
