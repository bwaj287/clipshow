"""Real FFmpeg regression using synthetic media, not private travel footage.

Set CLIPSHOW_TEST_FFMPEG/FFPROBE if the executables are not on PATH.
This covers >8 shots, photos, protected audio, BGM loops and full decode QC.
"""

import csv
import json
import os
import shutil
import subprocess

import pytest
from PIL import Image

from clipshow.vlog import (
    Compiler,
    VlogConfig,
    analysis_signature,
    file_identity,
    fingerprint,
    json_write,
)
from clipshow.vlog_render import FPS, music_bed, render


def test_photo_dialogue_chunk_boundaries_and_long_music(tmp_path):
    ffmpeg = os.environ.get("CLIPSHOW_TEST_FFMPEG") or shutil.which("ffmpeg")
    ffprobe = os.environ.get("CLIPSHOW_TEST_FFPROBE") or shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("Real renderer test needs FFmpeg and ffprobe")
    config = VlogConfig(
        day="2026-08-20",
        catalog=tmp_path / "catalog.csv",
        output_dir=tmp_path / "out",
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        python="unused",
        autocut_repo=tmp_path / "AutoCut",
        videohighlighter_repo=tmp_path / "VideoHighlighter",
        photo_dir=tmp_path / "photos",
        music=tmp_path / "music.wav",
        target_seconds=40,
        width=360,
        height=640,
        transition=9 / FPS,
    )
    compiler = Compiler(config)
    original = tmp_path / "DJI_20260820075121_fixture.MP4"
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=128x72:rate=30000/1001:duration=30",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=330:sample_rate=48000:duration=30",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(original),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:sample_rate=48000:duration=8",
            "-c:a",
            "pcm_s16le",
            str(config.music),
        ],
        check=True,
        capture_output=True,
    )
    row = dict(
        Day=config.day,
        LogicalCapture="2026-08-20 07:51:21",
        ClipID="TEST-VIDEO",
        DateBasis="Filename timestamp",
        OriginalPath=str(original),
        DurationSec="30",
    )
    with config.catalog.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    config.photo_dir.mkdir()
    photos = []
    for n in range(2):
        path = config.photo_dir / f"photo{n}.jpg"
        exif = Image.Exif()
        exif[36867] = f"2026:08:20 08:0{n}:00"
        Image.new("RGB", (120, 80), (80 + 40 * n, 130, 190)).save(path, exif=exif)
        photos.append(path)
    candidates, plan = [], []
    span = 102 / FPS
    for n in range(9):
        candidate = dict(id=f"synthetic:{n}", day=config.day, score=0.9, start=0)
        if n in (1, 8):
            source = photos[0 if n == 1 else 1]
            with source.open("rb") as stream:
                import hashlib

                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            candidate.update(
                kind="photo",
                source=str(source),
                end=span + config.transition,
                seconds=span + config.transition,
                sha256=digest,
            )
        else:
            source_start = (n - (n > 1)) * 4
            candidate.update(
                kind="video",
                source=str(original),
                start=source_start,
                end=source_start + span + config.transition,
            )
            if n == 4:
                candidate.update(protected=True, audio="dialogue")
        candidates.append(candidate)
        plan.append(
            dict(
                candidate,
                duration=span,
                timeline_start=n * span,
                timeline_end=(n + 1) * span,
                audio=candidate.get("audio", "mute"),
            )
        )
    analysis = dict(
        schema=2,
        day=config.day,
        config_signature=analysis_signature(config),
        catalog_identity=file_identity(config.catalog),
        sources=[file_identity(p) for p in [original, *photos]],
        candidates=candidates,
    )
    json_write(compiler.out / "analysis.json", analysis)
    bed = music_bed(compiler, config.music, config.target_seconds)
    timeline = dict(
        day=config.day,
        plan=plan,
        analysis_identity=fingerprint(analysis),
        music_identity=file_identity(config.music),
        bed_identity=file_identity(bed),
        music_bed=str(bed),
    )
    json_write(compiler.out / "timeline.json", timeline)
    final = render(compiler)
    qc = json.loads(final.with_suffix(".qc.json").read_text())
    assert qc["status"] == qc["full_decode"] == "PASS"
    assert abs(qc["video_seconds"] - 9 * span) < 0.1
    assert qc["audio_windows"][-1]["start"] >= 20
    assert (compiler.out / "render/chunk_001.mp4").exists()
    assert original.exists() and all(photo.exists() for photo in photos)
