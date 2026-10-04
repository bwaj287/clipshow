"""Local travel Vlog compiler: Katna -> CLIP -> VideoHighlighter -> AutoCut.

All four engines are required by default. Failed/missing stages fail closed;
cached responses retain their engine/version and are not substitute algorithms.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from clipshow.vlog_worker import AUTOCUT_REVISION, VH_REVISION

DEFAULT_POSITIVE = [
    "a beautiful mountain landscape, turquoise alpine lake or river in a forest",
    "people exploring a scenic travel destination and smiling together",
    "a picturesque town street, local people and historic architecture",
    "friends enjoying food and talking at a cafe or restaurant",
    "a peaceful nature trail or wooden mountain boardwalk",
]
DEFAULT_NEGATIVE = [
    "a hand or finger covering the camera lens",
    "only shoes, legs, pavement or the ground with no visible scenery",
    "a blurry image, a whip pan, or an empty wall or ceiling",
    "a screenshot of a phone screen instead of a travel photograph",
]


def json_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def file_identity(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def finite(value, name, minimum, maximum):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def local_ollama(host):
    parts = urlsplit(host)
    if parts.scheme != "http" or parts.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("VideoHighlighter must use local, loopback Ollama")
    if (
        parts.username
        or parts.password
        or parts.query
        or parts.fragment
        or parts.path not in {"", "/"}
    ):
        raise ValueError("Use a plain local Ollama base URL")
    return host.rstrip("/")


@dataclass
class VlogConfig:
    day: str
    catalog: Path
    output_dir: Path
    ffmpeg: str
    ffprobe: str
    python: str
    autocut_repo: Path
    videohighlighter_repo: Path
    photo_dir: Path | None = None
    model_dir: Path | None = None
    music: Path | None = None
    ollama_model: str = "qwen3.8:27b"
    ollama_host: str = "http://127.0.0.1:11434"
    target_seconds: float = 600
    shot_seconds: float = 10
    min_shot: float = 3
    ideal_shot: float = 7
    photo_seconds: float = 8
    threshold: float = 0.55
    transition: float = 0.3
    katna_frames: int = 8
    candidates_per_video: int = 3
    vision_limit: int = 48
    keep_dialogue: tuple = ()
    positive: tuple = tuple(DEFAULT_POSITIVE)
    negative: tuple = tuple(DEFAULT_NEGATIVE)
    encoder: str = "libx264"
    width: int = 1080
    height: int = 1920
    video_layout: str = "preserve"
    worker_timeout: int = 900
    clip_ids: tuple = ()

    @classmethod
    def load(cls, path):
        path = Path(path).resolve(strict=True)
        values = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        if not isinstance(values, dict):
            raise ValueError("Config must be a YAML mapping")
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown Vlog config keys: {sorted(unknown)}")
        for name in (
            "catalog",
            "output_dir",
            "autocut_repo",
            "videohighlighter_repo",
            "photo_dir",
            "model_dir",
            "music",
        ):
            if values.get(name) is not None:
                item = Path(values[name]).expanduser()
                values[name] = (
                    (path.parent / item).resolve() if not item.is_absolute() else item.resolve()
                )
        values.setdefault("python", sys.executable)
        result = cls(**values)
        date.fromisoformat(result.day)
        result.ollama_host = local_ollama(result.ollama_host)
        for name, lo, hi in (
            ("target_seconds", 4, 7200),
            ("shot_seconds", 4, 60),
            ("min_shot", 0.5, 30),
            ("ideal_shot", 0.5, 60),
            ("photo_seconds", 2, 30),
            ("transition", 0, 1),
            ("threshold", 0, 1),
        ):
            setattr(result, name, finite(getattr(result, name), name, lo, hi))
        if result.min_shot + result.transition >= min(result.shot_seconds, result.photo_seconds):
            raise ValueError("Shot/photo duration must leave room for minimum shot and transition")
        result.transition = round(result.transition * 30000 / 1001) * 1001 / 30000
        for name in ("katna_frames", "candidates_per_video", "vision_limit", "worker_timeout"):
            value = getattr(result, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if result.encoder not in {"libx264", "h264_nvenc"}:
            raise ValueError("Supported encoders: libx264, h264_nvenc")
        if result.video_layout not in {"preserve", "crop"}:
            raise ValueError("video_layout must be preserve or crop")
        if (result.width, result.height) not in {(1080, 1920), (360, 640)}:
            raise ValueError("Use 1080x1920 or the 360x640 testing profile")
        for name in ("positive", "negative"):
            terms = getattr(result, name)
            if (
                not isinstance(terms, (tuple, list))
                or not terms
                or not all(isinstance(term, str) and term.strip() for term in terms)
            ):
                raise ValueError(f"{name} must contain nonempty text prompts")
        return result


def day_media(config, limit=None):
    with config.catalog.open(encoding="utf-8-sig", newline="") as source:
        rows = [row for row in csv.DictReader(source) if row["Day"] == config.day]
    if not rows:
        raise ValueError(f"No catalog videos for {config.day}")
    rows.sort(key=lambda row: (row["LogicalCapture"], row["ClipID"]))
    if config.clip_ids:
        known = {row["ClipID"] for row in rows}
        if set(config.clip_ids) - known:
            raise ValueError("Requested test clips not found in this day")
        rows = [row for row in rows if row["ClipID"] in config.clip_ids]
    ids = set()
    for row in rows:
        if row["ClipID"] in ids:
            raise ValueError("Duplicate catalog ClipID")
        ids.add(row["ClipID"])
        date.fromisoformat(row["LogicalCapture"][:10])
        if row["LogicalCapture"][:10] != config.day or not row["DateBasis"].strip():
            raise ValueError("Invalid capture date evidence")
        source = Path(row["OriginalPath"]).resolve(strict=True)
        # DJI capture filenames provide independent date evidence, not mtime.
        if source.name.startswith("DJI_") and source.name[4:12].isdigit():
            captured = datetime.strptime(source.name[4:12], "%Y%m%d").date().isoformat()
            if captured != config.day:
                raise ValueError("Original filename disagrees with catalog day")
        row["OriginalPath"] = str(source)
        finite(row["DurationSec"], "video duration", 0.01, 86400)
        if source.is_relative_to(config.output_dir):
            raise ValueError("Output directory must not contain source media")
    if limit is not None:
        if limit < 1:
            raise ValueError("Input test limit must be positive")
        rows = rows[:limit]
    return rows


def gate_candidates(candidates, observations):
    """Missing vision results never count as analyzed/approved footage."""
    by_id = {}
    for item in observations:
        if item["id"] in by_id:
            raise ValueError("Duplicate vision response ID")
        by_id[item["id"]] = item["analysis"]
    result = []
    for candidate in candidates:
        data = by_id.get(candidate["id"])
        if data is None or not data["usable"] or data["obstructed"]:
            continue
        item = dict(candidate, content=data, review_required=True)
        # Similarity/model confidence are heuristic scores, not probabilities.
        item["score"] = 0.8 * candidate["score"] + 0.2 * data["confidence"]
        result.append(item)
    return result


def validate_plan(plan, config):
    cursor = 0.0
    seen = {}
    for shot in plan:
        if shot["day"] != config.day:
            raise ValueError("Other-day source in timeline")
        span = finite(shot["duration"], "shot duration", 0.01, 86400)
        start = finite(shot["start"], "source start", 0, 86400)
        source_end = finite(shot["end"], "source end", start + 0.01, 86400)
        timeline_start = finite(shot["timeline_start"], "timeline start", 0, 86400)
        timeline_end = finite(shot["timeline_end"], "timeline end", 0.01, 86400)
        if abs(timeline_start - cursor) > 0.001:
            raise ValueError("Timeline gaps or overlap")
        if abs(timeline_end - cursor - span) > 0.001:
            raise ValueError("Invalid timeline end")
        if shot["kind"] == "video":
            if start + span + config.transition > source_end + 0.001:
                raise ValueError("Transition extends beyond reviewed window")
            key = shot["source"]
            end = start + span + config.transition
            if any(start < b and a < end for a, b in seen.get(key, [])):
                raise ValueError("Repeated/overlapping source windows")
            seen.setdefault(key, []).append((start, end))
        elif shot["kind"] != "photo":
            raise ValueError("Unknown timeline media kind")
        if shot.get("protected") and shot.get("audio") != "dialogue":
            raise ValueError("Protected dialogue has been muted")
        cursor += span
    if not plan or cursor > config.target_seconds + 0.001:
        raise ValueError("Empty or over-length plan; do not pad rejected footage")
    return cursor


def analysis_signature(config):
    """Settings that alter candidate selection, not music/render preferences."""
    names = (
        "day",
        "catalog",
        "photo_dir",
        "clip_ids",
        "model_dir",
        "ollama_host",
        "ollama_model",
        "shot_seconds",
        "photo_seconds",
        "min_shot",
        "transition",
        "threshold",
        "katna_frames",
        "candidates_per_video",
        "vision_limit",
        "keep_dialogue",
        "positive",
        "negative",
    )
    return fingerprint({name: str(getattr(config, name)) for name in names})


def validate_analysis(analysis, config):
    if analysis.get("schema") != 2 or analysis.get("day") != config.day:
        raise ValueError("Stale analysis from another day/version; rerun analyze")
    if analysis.get("config_signature") != analysis_signature(config):
        raise ValueError("Selection settings changed; rerun analyze")
    if analysis.get("catalog_identity") != file_identity(config.catalog):
        raise ValueError("Catalog changed since analysis; rerun analyze")
    for identity in analysis["sources"]:
        if identity != file_identity(identity["path"]):
            raise ValueError("Source changed since analysis; rerun analyze")
    allowed = {row["OriginalPath"]: row for row in day_media(config)}
    identities = {i["path"] for i in analysis["sources"]}
    ids = set()
    for candidate in analysis["candidates"]:
        if candidate["id"] in ids:
            raise ValueError("Duplicate analyzed candidate ID")
        ids.add(candidate["id"])
        source = Path(candidate["source"]).resolve(strict=True)
        if str(source) not in identities or candidate["day"] != config.day:
            raise ValueError("Candidate has no analyzed same-day source identity")
        if candidate["kind"] == "video":
            if str(source) not in allowed:
                raise ValueError("Video outside this day's catalog whitelist")
            end = float(candidate["end"])
            if (
                not 0
                <= candidate["start"]
                < end
                <= float(allowed[str(source)]["DurationSec"]) + 0.05
            ):
                raise ValueError("Candidate outside original source duration")
        elif candidate["kind"] == "photo":
            from clipshow.travel import photo_capture_day

            if not config.photo_dir or not source.is_relative_to(config.photo_dir):
                raise ValueError("Photo outside configured photo directory")
            if str(photo_capture_day(source)[0]) != config.day:
                raise ValueError("Photo is not from this day")
            with source.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != candidate["sha256"]:
                    raise ValueError("Photo changed since analysis")
        else:
            raise ValueError("Unknown analyzed media kind")


def validate_timeline(timeline, analysis, config):
    """Bind every output shot to the exact AI-reviewed candidate and source."""
    validate_analysis(analysis, config)
    if timeline.get("analysis_identity") != fingerprint(analysis):
        raise ValueError("Timeline no longer matches analysis; rerun plan")
    if timeline.get("day") != config.day:
        raise ValueError("Timeline belongs to a different day")
    if not config.music or timeline.get("music_identity") != file_identity(config.music):
        raise ValueError("BGM changed since planning; rerun plan")
    bed = Path(timeline["music_bed"])
    if not bed.resolve().is_relative_to(config.output_dir):
        raise ValueError("Music bed outside output directory")
    if timeline.get("bed_identity") != file_identity(bed):
        raise ValueError("Planned music bed changed; rerun plan")
    candidates = {c["id"]: c for c in analysis["candidates"]}
    protected_ids = {c["id"] for c in analysis["candidates"] if c.get("protected")}
    if not protected_ids.issubset({shot["id"] for shot in timeline["plan"]}):
        raise ValueError("Protected dialogue missing from timeline; increase target duration")
    seen = set()
    for shot in timeline["plan"]:
        candidate = candidates.get(shot["id"])
        if not candidate or shot["id"] in seen:
            raise ValueError("Unreviewed/repeated candidate in timeline")
        seen.add(shot["id"])
        if any(shot.get(k) != v for k, v in candidate.items()):
            raise ValueError("Timeline changed an analyzed source/window")
        if candidate.get("protected"):
            expected = candidate["end"] - candidate["start"] - config.transition
            if abs(shot["duration"] - expected) > 1001 / 30000 + 0.001:
                raise ValueError("Protected dialogue was shortened")
    return validate_plan(timeline["plan"], config)


class Compiler:
    def __init__(self, config):
        self.config = config
        self.out = config.output_dir
        self.out.mkdir(parents=True, exist_ok=True)
        self.code = Path(__file__).resolve().parent.parent
        self.audit = []
        self.preflight = {}

    def run_ffmpeg(self, args, log):
        result = subprocess.run(
            [self.config.ffmpeg, "-hide_banner", "-v", "error", *args], capture_output=True
        )
        (self.out / log).write_bytes(result.stderr)
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors="replace")[-2500:])
        return result.stdout

    def engine(self, name, request, key, cache=True):
        config = self.config
        # Source-code-aware cache: adapter changes cannot silently reuse old answers.
        identity = fingerprint(
            {
                "request": request,
                "runtime": self.preflight.get(name),
                "adapter": [
                    fingerprint((self.code / item).read_text(encoding="utf-8"))
                    for item in (
                        "clipshow/vlog_worker.py",
                        "clipshow/travel.py",
                        "clipshow/detection/semantic.py",
                        "clipshow/detection/local_clip.py",
                    )
                ],
                "pins": [AUTOCUT_REVISION, VH_REVISION, "Katna==0.9.2"],
            }
        )
        folder = self.out / "engines" / name / fingerprint(key)[:16]
        folder.mkdir(parents=True, exist_ok=True)
        response = folder / f"response_{identity[:16]}.json"
        stamp = folder / "stamp.json"
        if cache and response.exists() and stamp.exists():
            old = json.loads(stamp.read_text(encoding="utf-8"))
            if old.get("identity") == identity:
                result = json.loads(response.read_text(encoding="utf-8"))
                if result.get("status") == "OK":
                    self.audit.append(
                        {
                            "engine": name,
                            "key": key,
                            "status": "CACHED_OK",
                            "evidence": str(response),
                            "identity": identity,
                        }
                    )
                    json_write(self.out / "engine_audit.json", self.audit)
                    return result
        request_path = folder / "request.json"
        json_write(request_path, request)
        env = dict(
            os.environ,
            PYTHONIOENCODING="utf-8",
            OMP_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            NUMBA_NUM_THREADS="2",
        )
        env["PYTHONPATH"] = str(self.code)
        command = [
            config.python,
            "-m",
            "clipshow.vlog_worker",
            name,
            str(request_path),
            str(response),
        ]
        print(f"Running {name}: {key}", flush=True)
        with (folder / "run.log").open("wb") as log:
            try:
                process = subprocess.run(
                    command,
                    env=env,
                    cwd=folder,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=config.worker_timeout,
                )
                if process.returncode:
                    raise RuntimeError(
                        f"{name} failed ({process.returncode}); see {folder / 'run.log'}"
                    )
            except Exception:
                self.audit.append({"engine": name, "key": key, "status": "FAILED"})
                json_write(self.out / "engine_audit.json", self.audit)
                raise
        result = json.loads(response.read_text(encoding="utf-8"))
        if result.get("status") != "OK":
            raise RuntimeError(f"{name} did not produce a successful response")
        json_write(stamp, {"identity": identity})
        self.audit.append(
            {
                "engine": name,
                "key": key,
                "status": "OK",
                "evidence": str(response),
                "identity": identity,
            }
        )
        json_write(self.out / "engine_audit.json", self.audit)
        return result

    def doctor(self):
        for executable in (self.config.ffmpeg, self.config.ffprobe):
            subprocess.run([executable, "-version"], check=True, capture_output=True)
        for name in ("katna", "clip", "vision", "autocut"):
            request = {
                "operation": "doctor",
                "model_dir": str(self.config.model_dir) if self.config.model_dir else None,
                "repo": str(
                    self.config.videohighlighter_repo
                    if name == "vision"
                    else self.config.autocut_repo
                ),
                "host": self.config.ollama_host,
                "model": self.config.ollama_model,
            }
            self.preflight[name] = self.engine(name, request, "preflight", cache=False)
        return self.audit

    def proxy(self, row):
        original = Path(row["OriginalPath"])
        proxy_path = Path(row.get("ProxyPath") or original)
        if not proxy_path.exists():
            proxy_path = original
        identity = fingerprint(file_identity(proxy_path))
        output = self.out / "proxies" / f"{identity[:16]}.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.exists():
            # Autorotate first and discard DJI's secondary thumbnail stream.
            self.run_ffmpeg(
                [
                    "-i",
                    str(proxy_path),
                    "-map",
                    "0:v:0",
                    "-an",
                    "-vf",
                    "scale=384:384:force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1",
                    "-r",
                    "30000/1001",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-crf",
                    "24",
                    "-y",
                    str(output),
                ],
                f"proxy_{identity[:12]}.log",
            )
        return output

    def analyze(self, limit=None):
        config = self.config
        rows = day_media(config, limit)
        candidates, reports = [], []
        source_identities = [file_identity(r["OriginalPath"]) for r in rows]
        for row in rows:
            proxy = self.proxy(row)
            common = {
                "input": str(proxy),
                "identity": file_identity(row["OriginalPath"]),
                "proxy_identity": file_identity(proxy),
            }
            keyframes = self.engine(
                "katna",
                {
                    **common,
                    "work": str(self.out / "katna" / fingerprint(row["ClipID"])[:12]),
                    "ffmpeg": config.ffmpeg,
                    "frames": config.katna_frames,
                },
                row["ClipID"],
            )
            report = self.engine(
                "clip",
                {
                    **common,
                    "model_dir": str(config.model_dir) if config.model_dir else None,
                    "positive": config.positive,
                    "negative": config.negative,
                    "shot_seconds": config.shot_seconds,
                    "threshold": config.threshold,
                    "limit": config.candidates_per_video,
                },
                row["ClipID"],
            )
            anchors = [k["time"] for k in keyframes["keyframes"] if k["passed"]]
            for n, window in enumerate(report["result"]["candidates"]):
                if not any(window["start"] <= t < window["end"] for t in anchors):
                    continue
                candidates.append(
                    {
                        "id": f"{row['ClipID']}:{n}",
                        "kind": "video",
                        "day": config.day,
                        "source": row["OriginalPath"],
                        "proxy": str(proxy),
                        "start": window["start"],
                        "end": window["end"],
                        "capture": row["LogicalCapture"],
                        "date_basis": row["DateBasis"],
                        "score": window["mean_semantic"],
                        "katna_anchor": True,
                    }
                )
            reports.append({"id": row["ClipID"], "katna": keyframes, "clip": report})
        if config.photo_dir:
            from clipshow.travel import photo_capture_datetime

            photo_identities = [
                file_identity(p)
                for p in config.photo_dir.rglob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ]
            source_identities.extend(photo_identities)
            report = self.engine(
                "clip",
                {
                    "photo_dir": str(config.photo_dir),
                    "day": config.day,
                    "model_dir": str(config.model_dir) if config.model_dir else None,
                    "identities": photo_identities,
                    "threshold": config.threshold,
                    "positive": config.positive,
                    "negative": config.negative,
                },
                "photos",
            )
            hashes = set()
            for photo in report["result"]:
                path = Path(photo["source"])
                with path.open("rb") as source:
                    digest = hashlib.file_digest(source, "sha256").hexdigest()
                if digest in hashes:
                    continue
                hashes.add(digest)
                captured, _basis = photo_capture_datetime(path)
                assert captured.date().isoformat() == config.day
                candidates.append(
                    {
                        "id": f"photo:{digest[:16]}",
                        "kind": "photo",
                        "day": config.day,
                        "source": str(path),
                        "start": 0,
                        "end": config.photo_seconds,
                        "capture": captured.isoformat(sep=" "),
                        "date_basis": photo["date_basis"],
                        "score": photo["semantic_score"],
                        "seconds": config.photo_seconds,
                        "sha256": digest,
                    }
                )
        # Explicit, source-bounded dialogue windows are protected from beat cuts.
        by_id = {row["ClipID"]: row for row in rows}
        for n, dialogue in enumerate(config.keep_dialogue):
            row = by_id[dialogue["clip_id"]]
            start = finite(dialogue["start"], "dialogue start", 0, float(row["DurationSec"]))
            end = finite(
                dialogue["end"],
                "dialogue end",
                start + config.min_shot + config.transition,
                float(row["DurationSec"]),
            )
            candidates = [
                c
                for c in candidates
                if c["source"] != row["OriginalPath"] or c["end"] <= start or c["start"] >= end
            ]
            candidates.append(
                {
                    "id": f"dialogue:{n}",
                    "kind": "video",
                    "source": row["OriginalPath"],
                    "day": config.day,
                    "proxy": str(self.proxy(row)),
                    "start": start,
                    "end": end,
                    "capture": row["LogicalCapture"],
                    "date_basis": row["DateBasis"],
                    "protected": True,
                    "audio": "dialogue",
                    "score": 1.0,
                }
            )
        # Give each source a first look before spending the vision budget on its
        # second/third candidate. Protected speech always gets a vision check.
        protected = [c for c in candidates if c.get("protected")]
        remaining = sorted(
            [c for c in candidates if not c.get("protected")],
            key=lambda c: c["score"],
            reverse=True,
        )
        first, rest, sources = [], [], set()
        for c in remaining:
            if c["source"] in sources:
                rest.append(c)
            else:
                first.append(c)
                sources.add(c["source"])
        chosen = (protected + first + rest)[: config.vision_limit]
        if len(protected) > config.vision_limit:
            raise ValueError("Increase vision_limit to cover all protected dialogue")
        vision = self.engine(
            "vision",
            {
                "repo": str(config.videohighlighter_repo),
                "host": config.ollama_host,
                "model": config.ollama_model,
                "candidates": chosen,
                "source_identities": [file_identity(c["source"]) for c in chosen],
                "proxy_identities": [
                    file_identity(c["proxy"]) for c in chosen if c["kind"] == "video"
                ],
            },
            "day-content",
            cache=True,
        )
        accepted = gate_candidates(chosen, vision["observations"])
        rejected_protected = {c["id"] for c in protected} - {c["id"] for c in accepted}
        if rejected_protected:
            raise ValueError(
                f"Protected dialogue rejected by vision gate: {sorted(rejected_protected)}"
            )
        result = {
            "schema": 2,
            "day": config.day,
            "sources": source_identities,
            "config_signature": analysis_signature(config),
            "catalog_identity": file_identity(config.catalog),
            "videos_analyzed": len(rows),
            "candidates": accepted,
            "vision": vision,
            "review_required": True,
            "audit": self.audit,
            "limitations": [
                "CLIP and model confidence are not probabilities",
                "Sampled frames cannot certify a whole shot",
                "Review framing, brief obstructions and dialogue before publication",
            ],
        }
        json_write(self.out / "analysis.json", result)
        json_write(self.out / "coarse_engine_reports.json", reports)
        print(
            f"Analyzed {len(rows)} videos; {len(accepted)} video/photo candidates passed",
            flush=True,
        )
        return result

    def plan(self):
        from clipshow.vlog_render import music_bed

        config = self.config
        if not config.music:
            raise ValueError(
                "BGM not supplied: analysis is saved, but no timeline/render is fabricated"
            )
        analysis = json.loads((self.out / "analysis.json").read_text(encoding="utf-8"))
        validate_analysis(analysis, config)
        candidates = analysis["candidates"]
        if not candidates:
            raise ValueError("No approved candidates; do not force filler")
        bed = music_bed(self, config.music, config.target_seconds)
        result = self.engine(
            "autocut",
            {
                "repo": str(config.autocut_repo),
                "operation": "plan",
                "music": str(bed),
                "music_identity": file_identity(bed),
                "candidates": candidates,
                "target_seconds": config.target_seconds,
                "transition": config.transition,
                "min_shot": config.min_shot,
                "ideal_shot": config.ideal_shot,
            },
            "highlight-timeline",
        )
        validate_plan(result["plan"], config)
        result.update(
            day=config.day,
            music_bed=str(bed),
            music_source=str(config.music),
            requested_seconds=config.target_seconds,
            review_required=True,
            analysis_identity=fingerprint(analysis),
            music_identity=file_identity(config.music),
            bed_identity=file_identity(bed),
        )
        validate_timeline(result, analysis, config)
        json_write(self.out / "timeline.json", result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--stage", choices=["doctor", "analyze", "plan", "render", "all"], default="all"
    )
    parser.add_argument(
        "--limit-inputs", type=int, help="Explicit smoke-test subset; recorded in report"
    )
    args = parser.parse_args(argv)
    try:
        compiler = Compiler(VlogConfig.load(args.config))
        compiler.doctor()
        if args.stage in {"analyze", "all"}:
            compiler.analyze(args.limit_inputs)
        if args.stage in {"plan", "all"}:
            compiler.plan()
        if args.stage in {"render", "all"}:
            from clipshow.vlog_render import render

            render(compiler)
        return 0
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f"Vlog compiler stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
