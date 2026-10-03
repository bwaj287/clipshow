from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from clipshow.detection.semantic import SemanticDetector


def run_detector(margin, cancel_after=None):
    frames = [np.zeros((8, 8, 3), np.uint8) for _ in range(60)]
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = [30.0, 60]
    cap.read.side_effect = [(True, f) for f in frames] + [(False, None)]
    model = MagicMock()
    model.get_text_embeddings.side_effect = [np.array([[1.0, 0.0]]),
                                             np.array([[0.0, 1.0]])]
    model.get_image_embeddings.return_value = np.array([[margin, 0.0]])
    detector = SemanticDetector()
    detector._model = model
    with patch("clipshow.detection.semantic.cv2.VideoCapture", return_value=cap):
        result = detector.detect("mock.mp4", cancel_flag=cancel_after)
    assert cap.release.called
    return result.scores


def test_constant_good_video_has_no_holes_between_samples():
    scores = run_detector(0.2)
    assert len(scores) == 20
    assert np.all(scores > 0.9)
    assert np.ptp(scores) < 1e-6


def test_negative_only_video_not_scaled_up_to_one():
    scores = run_detector(-0.2)
    assert np.all(scores < 0.02)


def test_cancelled_tail_remains_unknown_not_good():
    calls = iter([False] * 16 + [True])
    scores = run_detector(0.2, lambda: next(calls))
    assert np.all(scores[6:] == 0)


def test_invalid_time_step():
    with pytest.raises(ValueError):
        SemanticDetector(time_step=0)
