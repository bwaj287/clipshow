"""Fail-closed compiler tests; real engine smoke tests are opt-in separately."""

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from clipshow.vlog import (
    Compiler,
    VlogConfig,
    analysis_signature,
    day_media,
    file_identity,
    fingerprint,
    gate_candidates,
    local_ollama,
    validate_analysis,
    validate_plan,
    validate_timeline,
)


@pytest.fixture
def config_file(tmp_path):
    values = dict(
        day="2026-08-20",
        catalog="catalog.csv",
        output_dir="out",
        ffmpeg="ffmpeg",
        ffprobe="ffprobe",
        autocut_repo="AutoCut",
        videohighlighter_repo="VideoHighlighter",
    )
    path = tmp_path / "vlog.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return path


def change(path, **kwargs):
    data = yaml.safe_load(path.read_text())
    data.update(kwargs)
    path.write_text(yaml.safe_dump(data))


def test_defaults_and_paths(config_file):
    config = VlogConfig.load(config_file)
    assert config.day == "2026-08-20"
    assert config.target_seconds == 600
    assert config.video_layout == "preserve"
    assert config.music is None
    assert config.catalog == config_file.parent / "catalog.csv"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True])
def test_invalid_timing(config_file, value):
    change(config_file, shot_seconds=value)
    with pytest.raises(ValueError):
        VlogConfig.load(config_file)


@pytest.mark.parametrize("value", [0, False, 1.5])
def test_invalid_vision_budget(config_file, value):
    change(config_file, vision_limit=value)
    with pytest.raises(ValueError):
        VlogConfig.load(config_file)


def test_unknown_config_is_not_silently_ignored(config_file):
    change(config_file, vidio_layout="crop")
    with pytest.raises(ValueError, match="Unknown"):
        VlogConfig.load(config_file)


@pytest.mark.parametrize(
    "host",
    [
        "https://example.com",
        "http://192.168.1.5:11434",
        "http://localhost:11434/api/generate",
        "http://user:password@localhost:11434",
    ],
)
def test_footage_cannot_be_sent_to_remote_ollama(host):
    with pytest.raises(ValueError):
        local_ollama(host)


def test_loopback_allowed():
    assert local_ollama("http://127.0.0.1:11434/") == "http://127.0.0.1:11434"


def catalog(config, day="2026-08-20", filename="DJI_20260820075121_0055_D.MP4"):
    original = config.catalog.parent / filename
    original.write_bytes(b"fixture-not-a-render-input")
    rows = [
        dict(
            Day=day,
            LogicalCapture=day + " 07:51:21",
            ClipID="BANFF-D02-003",
            DateBasis="Filename timestamp",
            OriginalPath=str(original),
            DurationSec="20",
        )
    ]
    with config.catalog.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return original


def test_dji_original_date_crosscheck(config_file):
    config = VlogConfig.load(config_file)
    catalog(config, filename="DJI_20260821075121_0055_D.MP4")
    with pytest.raises(ValueError, match="filename"):
        day_media(config)


def test_same_day_catalog(config_file):
    config = VlogConfig.load(config_file)
    catalog(config)
    assert len(day_media(config)) == 1


def test_wrong_day_not_included(config_file):
    config = VlogConfig.load(config_file)
    catalog(config, day="2026-08-21", filename="DJI_20260821075121_0055_D.MP4")
    with pytest.raises(ValueError, match="No catalog"):
        day_media(config)


def test_unanalyzed_frames_never_approved():
    assert gate_candidates([dict(id="a", score=0.9)], []) == []


@pytest.mark.parametrize("usable,obstructed", [(False, False), (True, True)])
def test_bad_vision_sequence_veto(usable, obstructed):
    observation = {
        "id": "a",
        "analysis": {"usable": usable, "obstructed": obstructed, "confidence": 0.99},
    }
    assert gate_candidates([dict(id="a", score=0.9)], [observation]) == []


def test_gate_preserves_provenance_and_review():
    observation = {"id": "a", "analysis": {"usable": True, "obstructed": False, "confidence": 0.8}}
    result = gate_candidates([dict(id="a", score=0.9, source="original.mp4")], [observation])
    assert result[0]["source"] == "original.mp4"
    assert result[0]["review_required"]


def shot(source="source.mp4", start=0):
    return dict(
        day="2026-08-20",
        kind="video",
        source=source,
        start=start,
        end=start + 12,
        duration=6,
        timeline_start=0,
        timeline_end=6,
        audio="mute",
    )


def test_transition_handle_not_added_outside_reviewed_window(config_file):
    config = VlogConfig.load(config_file)
    value = shot()
    value["end"] = 6.1
    with pytest.raises(ValueError, match="Transition"):
        validate_plan([value], config)


def test_other_day_cannot_be_rendered(config_file):
    config = VlogConfig.load(config_file)
    value = shot()
    value["day"] = "2026-08-21"
    with pytest.raises(ValueError, match="Other-day"):
        validate_plan([value], config)


def test_no_duplicate_source_windows(config_file):
    config = VlogConfig.load(config_file)
    a, b = shot(), shot(start=4)
    b.update(timeline_start=6, timeline_end=12)
    with pytest.raises(ValueError, match="overlapping"):
        validate_plan([a, b], config)


def test_protected_dialogue_cannot_be_muted(config_file):
    config = VlogConfig.load(config_file)
    value = shot()
    value["protected"] = True
    with pytest.raises(ValueError, match="muted"):
        validate_plan([value], config)


def test_no_music_no_fabricated_timeline(config_file):
    compiler = Compiler(VlogConfig.load(config_file))
    with pytest.raises(ValueError, match="BGM not supplied"):
        compiler.plan()
    assert not (compiler.out / "timeline.json").exists()


def test_all_four_engines_preflight_by_default(config_file, monkeypatch):
    compiler = Compiler(VlogConfig.load(config_file))
    monkeypatch.setattr("subprocess.run", lambda *a, **k: None)
    engines = []

    def engine(name, request, key, cache):
        engines.append(name)
        return {"status": "OK"}

    monkeypatch.setattr(compiler, "engine", engine)
    compiler.doctor()
    assert engines == ["katna", "clip", "vision", "autocut"]


def test_missing_engine_stops_pipeline(config_file, monkeypatch):
    compiler = Compiler(VlogConfig.load(config_file))
    monkeypatch.setattr("subprocess.run", lambda *a, **k: None)

    def engine(*args, **kwargs):
        raise RuntimeError("engine missing")

    monkeypatch.setattr(compiler, "engine", engine)
    with pytest.raises(RuntimeError, match="missing"):
        compiler.doctor()


def test_failed_worker_cannot_use_existing_response(config_file, monkeypatch):
    compiler = Compiler(VlogConfig.load(config_file))

    class Failed:
        returncode = 1

    monkeypatch.setattr("subprocess.run", lambda *a, **k: Failed())
    with pytest.raises(RuntimeError, match="failed"):
        compiler.engine("clip", {"input": "unavailable.mp4"}, "fixture")
    audit = json.loads((compiler.out / "engine_audit.json").read_text())
    assert audit[-1]["status"] == "FAILED"


def test_cache_carries_real_engine_evidence(config_file, monkeypatch):
    compiler = Compiler(VlogConfig.load(config_file))
    called = []

    class Success:
        returncode = 0

    def worker(command, **kwargs):
        called.append(command)
        Path(command[-1]).write_text(json.dumps({"status": "OK", "engine": "clipshow.CLIP"}))
        return Success()

    monkeypatch.setattr("subprocess.run", worker)
    compiler.engine("clip", {"input": "fixture"}, "fixture")
    compiler.engine("clip", {"input": "fixture"}, "fixture")
    assert len(called) == 1
    assert compiler.audit[-1]["status"] == "CACHED_OK"
    changed = copy.deepcopy(compiler.preflight)
    changed["clip"] = {"version": "different runtime"}
    compiler.preflight = changed
    compiler.engine("clip", {"input": "fixture"}, "fixture")
    assert len(called) == 2


@pytest.fixture
def analyzed_fixture(config_file):
    config = VlogConfig.load(config_file)
    original = catalog(config)
    config.music = config_file.parent / "music.wav"
    config.music.write_bytes(b"synthetic-music-identity")
    compiler = Compiler(config)
    bed = compiler.out / "bed.wav"
    bed.write_bytes(b"synthetic-bed-identity")
    candidate = shot(source=str(original))
    for name in ("duration", "timeline_start", "timeline_end", "audio"):
        candidate.pop(name)
    candidate["id"] = "fixture:0"
    candidate["score"] = 0.8
    analysis = dict(
        schema=2,
        day=config.day,
        config_signature=analysis_signature(config),
        catalog_identity=file_identity(config.catalog),
        sources=[file_identity(original)],
        candidates=[candidate],
    )
    planned = dict(candidate, duration=6, timeline_start=0, timeline_end=6, audio="mute")
    timeline = dict(
        day=config.day,
        analysis_identity=fingerprint(analysis),
        music_identity=file_identity(config.music),
        bed_identity=file_identity(bed),
        music_bed=str(bed),
        plan=[planned],
    )
    return compiler, analysis, timeline


def test_plan_source_is_bound_to_reviewed_candidate(analyzed_fixture):
    compiler, analysis, timeline = analyzed_fixture
    assert validate_timeline(timeline, analysis, compiler.config) == 6
    timeline["plan"][0]["source"] = "unreviewed.mp4"
    with pytest.raises(ValueError, match="changed an analyzed"):
        validate_timeline(timeline, analysis, compiler.config)


def test_stale_selection_settings_need_new_analysis(analyzed_fixture):
    compiler, analysis, _ = analyzed_fixture
    compiler.config.threshold = 0.9
    with pytest.raises(ValueError, match="Selection settings"):
        validate_analysis(analysis, compiler.config)


def test_original_changed_after_analysis(analyzed_fixture):
    compiler, analysis, _ = analyzed_fixture
    Path(analysis["sources"][0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Source changed"):
        validate_analysis(analysis, compiler.config)


def test_music_changed_after_planning(analyzed_fixture):
    compiler, analysis, timeline = analyzed_fixture
    compiler.config.music.write_bytes(b"changed-music")
    with pytest.raises(ValueError, match="BGM changed"):
        validate_timeline(timeline, analysis, compiler.config)


def test_protected_dialogue_cannot_be_silently_excluded(analyzed_fixture):
    compiler, analysis, timeline = analyzed_fixture
    analysis["candidates"][0].update(protected=True, audio="dialogue")
    timeline["analysis_identity"] = fingerprint(analysis)
    timeline["plan"] = []
    with pytest.raises(ValueError, match="Protected dialogue missing"):
        validate_timeline(timeline, analysis, compiler.config)


@pytest.mark.parametrize("key", ["timeline_start", "timeline_end", "end"])
def test_nonfinite_timeline_rejected(config_file, key):
    value = shot()
    value[key] = float("nan")
    with pytest.raises(ValueError):
        validate_plan([value], VlogConfig.load(config_file))


def test_vlog_cli_routes_without_starting_gui(monkeypatch):
    from clipshow.__main__ import main

    received = []
    monkeypatch.setattr("clipshow.vlog.main", lambda args: received.append(args) or 0)
    assert main(["--vlog-config", "vlog.yaml", "--vlog-stage", "analyze"]) == 0
    assert received == [["vlog.yaml", "--stage", "analyze"]]


def test_explicit_dialogue_crop_removes_only_reviewed_edges():
    from clipshow.vlog_worker import crop_review_frame

    frame = np.zeros((200, 100, 3), dtype=np.uint8)
    frame[-16:] = 255  # Synthetic bottom-edge lens obstruction.
    result = crop_review_frame(frame, {"zoom": 1.14, "crop_x": 0.5, "crop_y": 0})
    assert result.shape == (174, 86, 3)
    assert not result.any()


def test_invalid_review_framing_is_not_applied():
    from clipshow.vlog_worker import crop_review_frame

    with pytest.raises(ValueError):
        crop_review_frame(np.zeros((20, 20, 3), dtype=np.uint8), {"zoom": 0})
