"""Review one catalog day locally, using coarse coverage before dense shot review."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from clipshow.detection.local_clip import LocalClip
from clipshow.travel import frame_quality

PROMPTS = [
    "a beautiful mountain panorama overlooking a valley in Banff",
    "a scenic turquoise river surrounded by pine trees and mountains",
    "a picturesque historic town street with mountains in the background",
    "friends smiling in a gondola cabin looking at mountains",
    "friends laughing over coffee at an outdoor cafe",
    "a wooden boardwalk and mountain overlook with a scenic view",
    "a delicious dinner and friends talking at a warm restaurant",
]
BAD = [
    "a hand or finger covering the camera lens",
    "a close up of legs, shoes, pavement or ground",
    "a blurry shaky photograph with no clear subject",
    "an empty ceiling or blank wall",
    "a photograph of only blue sky with no landscape",
]


def catalog_day(catalog, day):
    with catalog.open(encoding="utf-8-sig", newline="") as source:
        rows = [row for row in csv.DictReader(source) if row["Day"] == day]
    if not rows:
        raise ValueError(f"No videos cataloged for {day}")
    if any(not row["ClipID"].startswith("BANFF-D01-") for row in rows) and day == "2026-08-19":
        raise ValueError("Day 1 catalog membership mismatch")
    return rows


def analyze_row(row, ffmpeg, model, positive, negative, out, step):
    target = out / (row["ClipID"] + ".json")
    # Cache only this revision/step; source paths and original membership stay explicit.
    if target.exists():
        old = json.loads(target.read_text(encoding="utf-8"))
        if old.get("step_seconds") == step and old.get("source") == row["OriginalPath"]:
            return old
    path = row["ProxyPath"] if Path(row["ProxyPath"]).exists() else row["OriginalPath"]
    # fps start_time=0 samples the first frame, then every step seconds. All
    # camera display orientation is applied by FFmpeg before the model sees it.
    args = [ffmpeg, "-v", "error", "-i", path, "-map", "0:v:0", "-vf",
            f"fps=1/{step}:start_time=0,scale=216:384:force_original_aspect_ratio=decrease,"
            "pad=216:384:(ow-iw)/2:(oh-ih)/2", "-an", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-"]
    data = subprocess.check_output(args)
    frames = np.frombuffer(data, np.uint8).reshape(-1, 384, 216, 3)
    images = [Image.fromarray(frame) for frame in frames]
    embeddings = model.get_image_embeddings(images)
    positives = embeddings @ positive.T
    negatives = embeddings @ negative.T
    margin = positives.max(axis=1) - negatives.max(axis=1)
    scores = 1 / (1 + np.exp(-20 * (margin - 0.03)))
    measurements = []
    for index, frame in enumerate(frames):
        passed, metrics = frame_quality(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        measurements.append(dict(time=index * step, semantic=float(scores[index]),
                                 category=PROMPTS[int(np.argmax(positives[index]))],
                                 positive=float(positives[index].max()),
                                 negative=float(negatives[index].max()),
                                 quality_pass=passed, **metrics))
    # Prefer best contiguous coarse windows; each will still need a dense/full review.
    windows = []
    for index in range(len(frames) - 2):
        start = index * step
        if start + 2 * step + 1 > float(row["DurationSec"]):
            continue
        span = measurements[index:index + 3]
        mean = float(np.mean([m["semantic"] for m in span]))
        low = min(m["semantic"] for m in span)
        if all(m["quality_pass"] for m in span):
            windows.append(dict(start=start, end=start + 2 * step,
                                mean=mean, low=low, category=span[1]["category"]))
    windows.sort(key=lambda item: (item["mean"], item["low"]), reverse=True)
    kept = []
    for window in windows:
        if len(kept) >= 6:
            break
        if any(window["start"] < old["end"] + step and
               old["start"] < window["end"] + step for old in kept):
            continue
        kept.append(window)
    result = dict(clip_id=row["ClipID"], day=row["Day"], source=row["OriginalPath"],
                  logical_capture=row["LogicalCapture"], date_basis=row["DateBasis"],
                  duration=float(row["DurationSec"]), step_seconds=step,
                  measurements=measurements, candidate_windows=kept,
                  review_required=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    # Six separate coarse windows, not six photographs from one instant.
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default(size=14)
    canvas = Image.new("RGB", (6 * 224, 428), "#171d22")
    draw = ImageDraw.Draw(canvas)
    for column, window in enumerate(kept):
        index = min(len(images) - 1, round((window["start"] + step) / step))
        canvas.paste(images[index], (column * 224 + 4, 26))
        draw.text((column * 224 + 4, 4),
                  f'{window["start"]:.0f}-{window["end"]:.0f}s {window["mean"]:.2f}',
                  font=font, fill="white")
    canvas.save(out / (row["ClipID"] + ".jpg"), quality=90)
    print(f'{row["ClipID"]}: {len(frames)} frames; best={kept[0]["mean"]:.2f}'
          if kept else f'{row["ClipID"]}: no technical-quality window', flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", required=True, type=Path)
    ap.add_argument("--day", required=True)
    ap.add_argument("--ffmpeg", required=True)
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--step-seconds", type=float, default=3)
    args = ap.parse_args()
    if args.step_seconds <= 0:
        ap.error("--step-seconds must be positive")
    rows = catalog_day(args.catalog, args.day)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = LocalClip(args.model_dir, batch_size=8)
    positive = model.get_text_embeddings(PROMPTS)
    negative = model.get_text_embeddings(BAD)
    reports = [analyze_row(row, args.ffmpeg, model, positive, negative,
                           args.output_dir, args.step_seconds) for row in rows]
    (args.output_dir / "day_review.json").write_text(json.dumps(reports, indent=2),
                                                  encoding="utf-8")
    # Assemble six rows per contact sheet for human review.
    for page in range(0, len(reports), 6):
        canvas = Image.new("RGB", (1344, 6 * 450), "#171d22")
        draw = ImageDraw.Draw(canvas)
        for index, report in enumerate(reports[page:page + 6]):
            draw.text((8, index * 450), report["clip_id"], fill="white")
            with Image.open(args.output_dir / (report["clip_id"] + ".jpg")) as image:
                canvas.paste(image, (0, index * 450 + 20))
        canvas.save(args.output_dir / f"coarse_review_{page // 6 + 1:02d}.jpg", quality=90)


if __name__ == "__main__":
    main()
