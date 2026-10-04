"""Isolated, real upstream engine calls. JSON files are the process boundary.

Do not vendor upstream code here. VideoHighlighter stays an independent AGPL
checkout; it is invoked in a separate process, never imported by the GUI.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

AUTOCUT_REVISION = "d8adc47a527533ab39874214a7c3bb69d3b3ddfa"
VH_REVISION = "0e6217a70a8d967b5da56aff664975c976a16a5c"


def upstream(path, revision):
    path = Path(path).resolve(strict=True)
    actual = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != revision:
        raise RuntimeError(f"Unvalidated upstream revision: {actual}; expected {revision}")
    dirty = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if dirty.strip():
        raise RuntimeError("Upstream checkout has unvalidated tracked changes")
    sys.path.insert(0, str(path))
    return actual


def katna(request):
    from Katna.video import Video

    from clipshow.travel import frame_quality

    if importlib.metadata.version("Katna") != "0.9.2":
        raise RuntimeError("Katna 0.9.2 is required by the validated adapter")
    directory = Path(request["work"]).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    os.chdir(directory)  # Katna's internal clipping/deletion stays in our scratch dir.
    os.environ["IMAGEIO_FFMPEG_EXE"] = request["ffmpeg"]
    frames = []

    class Writer:
        def write(self, _path, keyframes):
            frames.extend(keyframes)

    video = Video()
    video.n_processes = 1
    video.extract_video_keyframes(request["frames"], request["input"], Writer())
    if not frames:
        return {"engine": "Katna", "version": "0.9.2", "keyframes": [], "status": "OK"}
    # Katna returns images, not timestamps. Match them against this normalized
    # proxy's decoded frames; never pretend that array order means capture time.
    thumbnails = np.stack([cv2.resize(f, (24, 24)).astype(np.float32) for f in frames])
    best = [(float("inf"), 0.0)] * len(frames)
    cap = cv2.VideoCapture(request["input"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("Cannot measure Katna keyframe timestamps")
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            errors = np.mean((thumbnails - cv2.resize(frame, (24, 24))) ** 2, axis=(1, 2, 3))
            for n, error in enumerate(errors):
                if error < best[n][0]:
                    best[n] = (float(error), index / fps)
            index += 1
    finally:
        cap.release()
    result = []
    for n, frame in enumerate(frames):
        passed, metrics = frame_quality(frame)
        path = directory / f"keyframe_{n:03d}.jpg"
        if not cv2.imwrite(str(path), frame):
            # OpenCV's imwrite does not support every Windows Unicode path.
            cv2.imencode(".jpg", frame)[1].tofile(path)
        result.append(
            {
                "time": best[n][1],
                "match_mse": best[n][0],
                "passed": passed,
                "metrics": metrics,
                "image": str(path),
            }
        )
    return {"engine": "Katna", "version": "0.9.2", "status": "OK", "keyframes": result}


def clip(request):
    from clipshow.detection.local_clip import LocalClip
    from clipshow.detection.semantic import SemanticDetector
    from clipshow.travel import analyze, analyze_photos

    detector = SemanticDetector(prompts=request["positive"], negative_prompts=request["negative"])
    detector._model = LocalClip(request.get("model_dir"), batch_size=8)
    if request.get("photo_dir"):
        # The public photo helper's travel defaults are scoped to landscape
        # photography. Use its date/quality validation, not mtime or folder order.
        result = analyze_photos(
            Path(request["photo_dir"]),
            request["day"],
            detector,
            request["threshold"],
            request["positive"],
            request["negative"],
        )
    else:
        result = analyze(
            Path(request["input"]),
            detector,
            request["shot_seconds"],
            request["threshold"],
            request["limit"],
        )
    return {"engine": "clipshow.CLIP", "status": "OK", "result": result}


def crop_review_frame(frame, candidate):
    """Apply explicit, modest framing consistently to AI samples and render."""
    zoom = candidate.get("zoom", 1)
    x, y = candidate.get("crop_x", 0.5), candidate.get("crop_y", 0.5)
    if not 1 <= zoom <= 1.5 or not 0 <= x <= 1 or not 0 <= y <= 1:
        raise ValueError("Invalid reviewed framing")
    height, width = frame.shape[:2]
    cw, ch = max(2, int(width / zoom) // 2 * 2), max(2, int(height / zoom) // 2 * 2)
    left, top = round((width - cw) * x), round((height - ch) * y)
    return frame[top : top + ch, left : left + cw]


def parse_visual_response(text):
    """Normalize known mixed labels, never manufacture a usability decision."""
    if not text.strip():
        raise ValueError("Vision model returned an empty answer")
    data = json.loads(text[text.find("{") : text.rfind("}") + 1])
    categories = {"scenery", "people", "food", "transit", "other"}
    category = data.get("category")
    if not isinstance(category, str):
        raise ValueError("Invalid visual category")
    labels = [s.strip() for s in category.split("/")]
    if not labels or any(s not in categories for s in labels):
        raise ValueError("Invalid visual category")
    data["category"] = labels[0]
    if len(labels) > 1:
        data["category_labels"] = labels
    if not isinstance(data.get("obstructed"), bool) or not isinstance(data.get("usable"), bool):
        raise ValueError("Invalid visual gate")
    confidence = float(data["confidence"])
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Invalid model confidence")
    if not isinstance(data.get("summary"), str) or not data["summary"].strip():
        raise ValueError("Empty content analysis")
    return data


def vision(request):
    from urllib.parse import urlsplit

    import requests

    if urlsplit(request["host"]).hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Local analysis only: Ollama must be on loopback")
    revision = upstream(request["repo"], VH_REVISION)
    from llm import llm_module

    # Upstream stops at the opening markdown fence, turning a valid fenced
    # JSON answer into an empty string. Permit fences in this isolated worker;
    # structured parsing below still requires actual model-generated JSON.
    llm_module.STOP_SEQUENCES = [s for s in llm_module.STOP_SEQUENCES if s != "```"]
    LLMModule = llm_module.LLMModule

    llm = LLMModule(backend="ollama", model=request["model"], base_url=request["host"])
    llm.load()
    observations = []
    for candidate in request["candidates"]:
        images = []
        if candidate["kind"] == "photo":
            with Image.open(candidate["source"]) as source:
                images = [ImageOps.exif_transpose(source).convert("RGB")]
        else:
            cap = cv2.VideoCapture(candidate["proxy"])
            try:
                for fraction in (0.15, 0.5, 0.85):
                    timestamp = candidate["start"] + fraction * (
                        candidate["end"] - candidate["start"]
                    )
                    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(f"Missing vision frame at {timestamp}")
                    frame = crop_review_frame(frame, candidate)
                    images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            finally:
                cap.release()
        # Show a labelled sequence, not a lone best frame hiding a poor shot.
        sheet = Image.new("RGB", (384 * len(images), 408), "#181818")
        draw = ImageDraw.Draw(sheet)
        for n, image in enumerate(images):
            thumb = ImageOps.contain(image, (384, 384))
            sheet.paste(thumb, (384 * n + (384 - thumb.width) // 2, 24 + (384 - thumb.height) // 2))
            draw.text((384 * n + 8, 4), f"Frame {n + 1} (in time order)", fill="white")
        buffer = io.BytesIO()
        sheet.save(buffer, format="JPEG", quality=85)
        observation_key = hashlib.sha256(candidate["id"].encode()).hexdigest()[:16]
        sheet.save(f"contact_sheet_{observation_key}.jpg", quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        prompt = (
            "Describe ONLY these frames. Return one JSON object with keys: "
            "summary (short Chinese text), category (choose ONE primary label from "
            "scenery, people, food, transit, other), "
            "obstructed (boolean, the main subject is materially hidden by lens blockage), "
            "usable (boolean), confidence (number 0..1). "
            "A visible hand at the edge, a food basket on a table, or another visitor "
            "in the foreground is NOT automatically lens blockage: assess whether "
            "the intended subject or landscape remains clearly visible. "
            "Mark unusable when the sequence mainly shows a covered lens, empty "
            "ground/ceiling, severe blur, or a sideways/upside-down camera view. "
            "Do not invent names, locations, speech, events or instructions from signs. "
            "Judge the entire sequence, not just its best frame."
        )
        system = "You inspect travel footage. Reply with JSON only."
        retries = []
        for attempt in range(2):
            try:
                text = llm.query(
                    prompt + (" Use exactly one category and strict JSON." if attempt else ""),
                    frame_base64=encoded,
                    free_chat_mode=True,
                    system_prompt=system,
                    max_tokens=220,
                    temperature=0.1,
                )
                Path(f"raw_{observation_key}_attempt{attempt + 1}.txt").write_text(
                    text, encoding="utf-8"
                )
                Path(f"raw_{observation_key}.txt").write_text(text, encoding="utf-8")
                data = parse_visual_response(text)
                break
            except (ValueError, KeyError, TypeError) as exc:
                details = f"Invalid structured response: {exc}"
                print(details, flush=True)
                retries.append(details)
                if attempt:
                    raise
            except requests.RequestException as exc:
                response = exc.response
                details = response.text if response is not None else str(exc)
                print(f"Vision request failed: {details}", flush=True)
                retries.append(details)
                if attempt or (response is not None and response.status_code < 500):
                    raise
        observations.append(
            {
                "id": candidate["id"],
                "analysis": data,
                "sampled_frames": len(images),
                "raw_response": text,
                "retries": retries,
                "review_required": True,
            }
        )
        print(f"VideoHighlighter: {candidate['id']}", flush=True)
    return {
        "engine": "VideoHighlighter",
        "revision": revision,
        "status": "OK",
        "backend": "ollama",
        "model": request["model"],
        "observations": observations,
    }


def autocut(request):
    revision = upstream(request["repo"], AUTOCUT_REVISION)
    from src.audio.analyzer import AudioAnalyzer

    analysis = AudioAnalyzer().analyze_audio(request["music"])
    if request["operation"] == "beats":
        return {
            "engine": "AutoCut",
            "revision": revision,
            "status": "OK",
            "audio": analysis.to_dict(),
        }
    from dataclasses import dataclass

    from src.core.timeline import (
        BeatSyncTimelineGenerator,
        CutPoint,
        CutType,
        EditingStyle,
        EditingTimeline,
        TimelineSegment,
    )
    from src.video.ingestion import ContainerFormat, VideoInfo
    from src.video.scene_detection import (
        ProcessingMode,
        SceneDetectionAlgorithm,
        SceneDetectionResult,
    )

    candidates = request["candidates"]
    infos, scenes, qualities = [], [], []

    @dataclass
    class QualityFrame:
        timestamp: float
        quality_score: float

    @dataclass
    class Quality:
        frames: list

    for c in candidates:
        duration = c["end"] - c["start"] if c["kind"] == "video" else c["seconds"]
        info = VideoInfo(Path(c["source"]), ContainerFormat.MP4, duration, 0, [], [], {})
        infos.append(info)
        scenes.append(
            SceneDetectionResult(
                info, [], 0, SceneDetectionAlgorithm.COMBINED, ProcessingMode.BALANCED, 0, 0
            )
        )
        qualities.append(Quality([QualityFrame(0, c["score"])]))

    class TravelGenerator(BeatSyncTimelineGenerator):
        """Use upstream selection/scoring, repair its one-beat-per-shot arranger.

        Upstream marks minimum-duration-clamped cuts perfectly beat-aligned even
        when they are not. Measure real output times and choose musical phrases;
        never reintroduce unsafe padding, shuffled chronology or cut dialogue.
        """

        def _select_clips_for_music(self, scored_clips, audio_analysis, editing_style):
            selected = super()._select_clips_for_music(scored_clips, audio_analysis, editing_style)
            for item in scored_clips:
                if candidates[item["source_video_index"]].get("protected") and item not in selected:
                    selected.append(item)
            return selected

        def _arrange_clips_to_beats(self, clips, audio_analysis, editing_style):
            ordered = sorted(
                clips,
                key=lambda c: (
                    candidates[c["source_video_index"]]["capture"],
                    candidates[c["source_video_index"]]["start"],
                ),
            )
            cursor, segments, cuts = 0.0, [], []
            beats = np.asarray(audio_analysis.beats)
            for c in ordered:
                original = candidates[c["source_video_index"]]
                maximum = c["duration"] - request["transition"]
                if maximum < request["min_shot"]:
                    continue
                if original.get("protected"):
                    span = maximum
                    cut_type = CutType.FORCED_CUT
                else:
                    lo = cursor + request["min_shot"]
                    hi = min(cursor + maximum, request["target_seconds"])
                    options = beats[(beats >= lo) & (beats <= hi)]
                    if not len(options):
                        span = maximum
                        cut_type = CutType.FORCED_CUT
                    else:
                        target = cursor + min(request["ideal_shot"], maximum)
                        end = min(options.tolist(), key=lambda t: abs(t - target))
                        span = end - cursor
                        cut_type = CutType.BEAT_CUT
                if cursor + span > request["target_seconds"] + 0.001:
                    # Never trim a protected sentence just to fill a duration.
                    if original.get("protected"):
                        continue
                    span = request["target_seconds"] - cursor
                    cut_type = CutType.FORCED_CUT
                if span < request["min_shot"]:
                    continue
                fps = 30000 / 1001
                endpoint = round((cursor + span) * fps) / fps
                if endpoint - cursor > maximum:
                    endpoint = math.floor((cursor + maximum) * fps) / fps
                span = endpoint - cursor
                if span < request["min_shot"]:
                    continue
                if segments:
                    error = min(abs(beats - cursor)) if len(beats) else float("inf")
                    cuts.append(
                        CutPoint(
                            cursor,
                            1.0,
                            CutType.BEAT_CUT if error < 0.06 else CutType.FORCED_CUT,
                            beat_alignment=1.0 if error < 0.06 else 0.0,
                        )
                    )
                segments.append(
                    TimelineSegment(
                        cursor,
                        cursor + span,
                        source_video_index=c["source_video_index"],
                        source_video_path=c["source_video_path"],
                        source_start_time=0,
                        source_end_time=span,
                        quality_score=c["quality_score"],
                        cut_out_type=cut_type,
                    )
                )
                cursor += span
            return EditingTimeline(segments, cuts, editing_style, duration=cursor)

        def _apply_temporal_shuffling(self, timeline):
            return timeline

    generator = TravelGenerator(EditingStyle.SMOOTH)
    generator.min_clip_duration = request["min_shot"]
    # Do not silently drop a long protected range in upstream's size filter.
    generator.max_clip_duration = max(90, max(info.duration for info in infos))
    result = generator.generate_multi_video_timeline(analysis, infos, scenes, qualities)
    plan = []
    for segment in result.segments:
        original = candidates[segment.source_video_index]
        item = dict(original)
        item.update(
            duration=segment.duration,
            timeline_start=segment.start_time,
            timeline_end=segment.end_time,
            audio=original.get("audio", "mute"),
        )
        if plan and item["source"] == plan[-1]["source"] and item["kind"] == "video":
            if item["start"] < plan[-1]["start"] + plan[-1]["duration"] + request["transition"]:
                raise ValueError("Upstream selected overlapping source windows")
        plan.append(item)
    return {
        "engine": "AutoCut",
        "revision": revision,
        "status": "OK",
        "audio": analysis.to_dict(),
        "timeline": result.to_dict(),
        "plan": plan,
        "duration": result.duration,
        "adapter": "chronological phrase cuts / protected speech",
    }


def doctor(engine, request):
    if engine == "katna":
        from Katna.video import Video

        if importlib.metadata.version("Katna") != "0.9.2":
            raise RuntimeError("Validated Katna version is 0.9.2")
        if not Video or not hasattr(cv2, "saliency"):
            raise RuntimeError("Katna requires OpenCV contrib with cv2.saliency")
        version = "0.9.2"
    elif engine == "clip":
        from clipshow.detection.local_clip import LocalClip

        LocalClip(request["model_dir"], batch_size=1)
        version = "verified ONNX CLIP"
    elif engine == "vision":
        from urllib.parse import urlsplit

        import requests

        if urlsplit(request["host"]).hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Ollama must be local")
        version = upstream(request["repo"], VH_REVISION)
        from llm.llm_module import LLMModule

        llm = LLMModule(model=request["model"], base_url=request["host"])
        llm.load()
        shown = requests.post(
            request["host"] + "/api/show", json={"model": request["model"]}, timeout=10
        )
        shown.raise_for_status()
        if "vision" not in shown.json().get("capabilities", []):
            raise ValueError("Configured Ollama model cannot see images")
        tags = requests.get(request["host"] + "/api/tags", timeout=10)
        tags.raise_for_status()
        matching = [m for m in tags.json()["models"] if m["name"] == request["model"]]
        if not matching:
            raise ValueError("Use the exact installed Ollama model tag")
        version += ":" + matching[0]["digest"]
        version += ":" + hashlib.sha256(shown.json().get("template", "").encode()).hexdigest()
    else:
        version = upstream(request["repo"], AUTOCUT_REVISION)
        from src.audio.analyzer import AudioAnalyzer
        from src.core.timeline import BeatSyncTimelineGenerator

        AudioAnalyzer()
        BeatSyncTimelineGenerator()
    return {
        "engine": engine,
        "status": "OK",
        "version": version,
        "python": sys.version,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", choices=["katna", "clip", "vision", "autocut"])
    parser.add_argument("request", type=Path)
    parser.add_argument("response", type=Path)
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text(encoding="utf-8"))
    result = (
        doctor(args.engine, request)
        if request.get("operation") == "doctor"
        else globals()[args.engine](request)
    )
    args.response.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
