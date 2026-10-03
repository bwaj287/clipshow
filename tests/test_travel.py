import numpy as np
import pytest
from PIL import Image

from clipshow.travel import frame_quality, photo_capture_day, select_windows


def test_irrelevant_video_not_promoted_to_highlight():
    assert select_windows(np.full(200, 0.12), np.ones(200, bool)) == []


def test_single_good_frame_cannot_choose_bad_shot():
    scores = np.full(100, 0.2)
    scores[50] = 1
    assert select_windows(scores, np.ones(100, bool)) == []


def test_quality_veto_cannot_be_reintroduced_by_padding():
    quality = np.ones(200, bool)
    quality[40:80] = False
    candidates = select_windows(np.full(200, 0.9), quality)
    assert candidates
    assert all(not (c.start < 8 and c.end > 4) for c in candidates)


def test_best_window_is_not_always_beginning():
    scores = np.full(200, 0.1)
    scores[100:180] = 0.9
    candidates = select_windows(scores, np.ones(200, bool), limit=1)
    assert len(candidates) == 1
    assert candidates[0].start >= 10
    assert candidates[0].review_required


def test_no_duplicate_overlapping_windows():
    candidates = select_windows(np.full(240, 0.9), np.ones(240, bool))
    assert all(a.end <= b.start for a, b in zip(candidates, candidates[1:]))


def test_technical_filter_rejects_blank_and_black():
    assert not frame_quality(np.zeros((120, 160, 3), np.uint8))[0]
    assert not frame_quality(np.full((120, 160, 3), 255, np.uint8))[0]


def test_invalid_timeline_fails_closed():
    with pytest.raises(ValueError):
        select_windows(np.ones(10), np.ones(9))
    with pytest.raises(ValueError):
        select_windows(np.array([np.nan]), np.ones(1))


def test_photo_exif_has_priority_over_filename(tmp_path):
    path = tmp_path / "DJI_20260822120000_0001_D.JPG"
    exif = Image.Exif()
    exif[36867] = "2026:08:21 10:30:00"
    Image.new("RGB", (32, 32)).save(path, exif=exif)
    day, basis = photo_capture_day(path)
    assert str(day) == "2026-08-21"
    assert basis == "EXIF"


def test_unknown_photo_date_not_inferred_from_copy_time(tmp_path):
    path = tmp_path / "undated.jpg"
    Image.new("RGB", (32, 32)).save(path)
    assert photo_capture_day(path) == (None, "unknown")


def test_photo_dji_date_fallback(tmp_path):
    path = tmp_path / "DJI_20260821103000_0001_D.JPG"
    Image.new("RGB", (32, 32)).save(path)
    assert str(photo_capture_day(path)[0]) == "2026-08-21"
