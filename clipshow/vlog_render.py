"""Bounded portrait FFmpeg renderer; no source mutation, full-length audio QC."""

from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from clipshow.vlog import file_identity, fingerprint, finite, json_write, validate_timeline

FPS = 30000 / 1001


def probe(config, path):
    return json.loads(
        subprocess.check_output(
            [
                config.ffprobe,
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ]
        )
    )


def music_bed(compiler, source, duration):
    config = compiler.config
    source = Path(source).resolve(strict=True)
    key = fingerprint({"source": file_identity(source), "duration": duration, "schema": 1})[:16]
    directory = compiler.out / "audio"
    directory.mkdir(parents=True, exist_ok=True)
    bed = directory / f"bed_{key}.wav"
    if bed.exists():
        if abs(float(probe(config, bed)["format"]["duration"]) - duration) < 0.05:
            return bed
    # Decode the entire provided BGM before using it, including files whose
    # source directory calls them corrupt. No KGM decryption is performed here.
    compiler.run_ffmpeg(
        ["-xerror", "-i", str(source), "-map", "0:a:0", "-f", "null", "-"], "bgm_decode.log"
    )
    normalized = directory / f"normalized_{key}.wav"
    compiler.run_ffmpeg(
        [
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-af",
            "aresample=48000,loudnorm=I=-18:TP=-3:LRA=10,aresample=48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s24le",
            "-y",
            str(normalized),
        ],
        "bgm_normalize.log",
    )
    track = float(probe(config, normalized)["format"]["duration"])
    if track < 4:
        raise ValueError("BGM must be at least four seconds")
    overlap = min(3.0, track / 4)
    count = max(1, math.ceil((duration - overlap) / (track - overlap)))
    args, graph = [], []
    for n in range(count):
        args += ["-i", str(normalized)]
        graph.append(f"[{n}:a:0]asetpts=PTS-STARTPTS[m{n}]")
    label = "m0"
    for n in range(1, count):
        graph.append(f"[{label}][m{n}]acrossfade=d={overlap}:c1=qsin:c2=qsin[j{n}]")
        label = f"j{n}"
    graph.append(f"[{label}]atrim=duration={duration},afade=t=in:d=0.35[bed]")
    compiler.run_ffmpeg(
        [
            *args,
            "-filter_complex",
            ";".join(graph),
            "-map",
            "[bed]",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s24le",
            "-y",
            str(bed),
        ],
        "bgm_bed.log",
    )
    if abs(float(probe(config, bed)["format"]["duration"]) - duration) > 0.05:
        raise ValueError("BGM does not cover requested duration")
    return bed


def encoding(config):
    if config.encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "23", "-b:v", "0"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21"]


def assemble(compiler, paths, target, final_duration=None):
    config = compiler.config
    args, graph, total = [], [], 0.0
    if not paths:
        raise ValueError("Cannot assemble an empty timeline")
    video, audio = "v0", "a0"
    transition = config.transition
    for n, path in enumerate(paths):
        info = probe(config, path)
        length = float(next(s for s in info["streams"] if s["codec_type"] == "video")["duration"])
        args += ["-i", str(path)]
        graph += [
            f"[{n}:v:0]setpts=PTS-STARTPTS,fps=30000/1001,settb=AVTB[v{n}]",
            f"[{n}:a:0]aresample=48000,asetpts=PTS-STARTPTS[a{n}]",
        ]
        if n:
            if transition:
                graph += [
                    f"[{video}][v{n}]xfade=duration={transition}:"
                    f"offset={total - transition},fps=30000/1001,settb=AVTB[xv{n}]",
                    f"[{audio}][a{n}]acrossfade=d={transition}[xa{n}]",
                ]
            else:
                graph += [f"[{video}][{audio}][v{n}][a{n}]concat=n=2:v=1:a=1[xv{n}][xa{n}]"]
            video, audio = f"xv{n}", f"xa{n}"
            total -= transition
        else:
            video, audio = "v0", "a0"
        total += length
    if final_duration is not None:
        graph += [
            f"[{video}]tpad=stop_mode=clone:stop_duration=2,trim=duration={final_duration},"
            f"fade=t=out:st={max(0, final_duration - 1.2)}:d=1.2[vout]",
            f"[{audio}]apad,atrim=duration={final_duration}[aout]",
        ]
        video, audio = "vout", "aout"
    compiler.run_ffmpeg(
        [
            *args,
            "-filter_complex_threads",
            "2",
            "-filter_complex",
            ";".join(graph),
            "-map",
            f"[{video}]",
            "-map",
            f"[{audio}]",
            *encoding(config),
            "-pix_fmt",
            "yuv420p",
            "-r",
            "30000/1001",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            "-y",
            str(target),
        ],
        target.stem + ".log",
    )


def render(compiler):
    config = compiler.config
    timeline = json.loads((compiler.out / "timeline.json").read_text(encoding="utf-8"))
    analysis = json.loads((compiler.out / "analysis.json").read_text(encoding="utf-8"))
    duration = validate_timeline(timeline, analysis, config)
    directory = compiler.out / "render"
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    width, height = config.width, config.height
    for n, shot in enumerate(timeline["plan"]):
        raw = shot["duration"] + (config.transition if n < len(timeline["plan"]) - 1 else 0)
        output = directory / f"segment_{n:04d}.mp4"
        source = Path(shot["source"])
        if shot["kind"] == "photo":
            from clipshow.travel import photo_capture_day

            if str(photo_capture_day(source)[0]) != config.day:
                raise ValueError("Photo date changed before render")
            with source.open("rb") as stream:
                import hashlib

                if hashlib.file_digest(stream, "sha256").hexdigest() != shot["sha256"]:
                    raise ValueError("Photo changed before render")
            with Image.open(source) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
                bg = ImageOps.fit(image, (width, height)).filter(
                    ImageFilter.GaussianBlur(width / 30)
                )
                bg = Image.blend(bg, Image.new("RGB", bg.size, "black"), 0.25)
                foreground = ImageOps.contain(image, (int(width * 0.95), int(height * 0.82)))
                bg.paste(
                    foreground, ((width - foreground.width) // 2, (height - foreground.height) // 2)
                )
                layout = directory / f"photo_{n:04d}.jpg"
                bg.save(layout, quality=95)
            args = ["-i", str(layout)]
            vf = (
                f"scale={2 * width}:{2 * height},zoompan=z='1+0.035*on/{raw * FPS}':"
                f"x='(iw-iw/zoom)/2':y='(ih-ih/zoom)/2':d={round(raw * FPS)}:"
                f"s={width}x{height}:fps=30000/1001,setsar=1"
            )
        else:
            args = ["-ss", str(shot["start"]), "-i", str(source)]
            vf = (
                f"scale={width}:{height}:force_original_aspect_ratio=increase,"
                f"crop={width}:{height},setsar=1,setpts=PTS-STARTPTS"
            )
            if config.video_layout == "preserve":
                vf = (
                    f"split=2[b][f];[b]scale={width}:{height}:force_original_aspect_ratio=increase,"
                    f"crop={width}:{height},gblur=sigma=20[blur];"
                    f"[f]scale={width}:{height}:force_original_aspect_ratio=decrease[sharp];"
                    "[blur][sharp]overlay=(W-w)/2:(H-h)/2,setsar=1,setpts=PTS-STARTPTS"
                )
            zoom = finite(shot.get("zoom", 1), "reviewed zoom", 1, 1.5)
            x = finite(shot.get("crop_x", 0.5), "reviewed crop_x", 0, 1)
            y = finite(shot.get("crop_y", 0.5), "reviewed crop_y", 0, 1)
            if zoom > 1:
                vf = (
                    f"crop=w='trunc(iw/{zoom}/2)*2':h='trunc(ih/{zoom}/2)*2':"
                    f"x='(iw-ow)*{x}':y='(ih-oh)*{y}'," + vf
                )
        if shot["audio"] == "dialogue":
            audio = [
                "-map",
                "0:a:0",
                "-af",
                "aresample=48000,asetpts=PTS-STARTPTS,"
                "highpass=f=150:poles=2,lowpass=f=7500,afftdn=nr=14:nf=-38:tn=1,"
                "acompressor=threshold=.1:ratio=2.5:attack=15:release=200,"
                "loudnorm=I=-17:TP=-3:LRA=7,aresample=48000",
            ]
        else:
            args += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
            audio = ["-map", "1:a:0"]
        compiler.run_ffmpeg(
            [
                *args,
                "-map",
                "0:v:0",
                *audio,
                "-vf",
                vf,
                "-t",
                str(raw),
                *encoding(config),
                "-pix_fmt",
                "yuv420p",
                "-r",
                "30000/1001",
                "-c:a",
                "aac",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-y",
                str(output),
            ],
            output.stem + ".log",
        )
        paths.append(output)
    chunks = []
    for n in range(0, len(paths), 8):
        chunk = directory / f"chunk_{n // 8:03d}.mp4"
        assemble(compiler, paths[n : n + 8], chunk)
        chunks.append(chunk)
    picture = directory / "picture_and_dialogue.mp4"
    assemble(compiler, chunks, picture, duration)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    final = compiler.out / f"Vlog_{config.day}_{stamp}.mp4"
    # Independent bed inputs were assembled before this point. Never use an
    # exhausted asplit branch as though it were a fresh music loop.
    fade = min(6, duration / 4)
    graph = (
        f"[0:a:0]aresample=48000,apad,atrim=duration={duration},asplit=2[voice][key];"
        f"[1:a:0]atrim=duration={duration},afade=t=out:st={duration - fade}:d={fade}[bgm];"
        "[bgm][key]sidechaincompress=threshold=.018:ratio=8:attack=25:release=650[ducked];"
        "[voice][ducked]amix=inputs=2:normalize=0:duration=first,"
        f"loudnorm=I=-14:TP=-2:LRA=10,aresample=48000,apad,atrim=duration={duration}[mix]"
    )
    compiler.run_ffmpeg(
        [
            "-i",
            str(picture),
            "-i",
            timeline["music_bed"],
            "-filter_complex",
            graph,
            "-map",
            "0:v:0",
            "-map",
            "[mix]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-t",
            str(duration),
            "-movflags",
            "+faststart",
            "-n",
            str(final),
        ],
        "final_mix.log",
    )
    result = verify_audio(compiler, final, fade)
    json_write(final.with_suffix(".qc.json"), result)
    print(f"Verified portrait preview: {final}", flush=True)
    return final


def verify_audio(compiler, path, closing_fade=6):
    config = compiler.config
    info = probe(config, path)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    audio = next(s for s in info["streams"] if s["codec_type"] == "audio")
    vd, ad = float(video["duration"]), float(audio["duration"])
    if abs(vd - ad) > 0.1 or (video["width"], video["height"]) != (config.width, config.height):
        raise ValueError("Truncated audio or wrong portrait dimensions")
    data = compiler.run_ffmpeg(
        ["-xerror", "-i", str(path), "-vn", "-ac", "1", "-ar", "8000", "-f", "f32le", "-"],
        "audio_decode.log",
    )
    samples = np.frombuffer(data, dtype="<f4")
    if abs(len(samples) / 8000 - vd) > 0.15:
        raise ValueError("Decoded audio ends early")
    windows = []
    for start in range(0, max(0, int(vd - closing_fade)), 5):
        window = samples[start * 8000 : min((start + 5) * 8000, len(samples))]
        rms = float(20 * np.log10(max(float(np.sqrt(np.mean(window**2))), 1e-10)))
        if rms < -55:
            raise ValueError(f"Unexpected silent interval at {start}s")
        windows.append({"start": start, "rms_dbfs": rms})
    compiler.run_ffmpeg(
        ["-xerror", "-i", str(path), "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
        "full_decode.log",
    )
    return {
        "file": str(path),
        "status": "PASS",
        "video_seconds": vd,
        "audio_seconds": ad,
        "decoded_audio_seconds": len(samples) / 8000,
        "audio_windows": windows,
        "portrait": [video["width"], video["height"]],
        "full_decode": "PASS",
    }
