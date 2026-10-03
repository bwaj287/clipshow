"""Semantic content detection via CLIP (ONNX Runtime, no PyTorch).

Lazy-loads onnx_clip on first use. Downloads CLIP ViT-B/32 model (~338MB)
to ~/.clipshow/models/ on first use. Samples frames at 2 FPS and scores
against user-configurable text prompts, with negative prompt contrast and
temporal smoothing.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d

from clipshow.detection.base import Detector, DetectorResult

DEFAULT_PROMPTS = [
    "exciting moment",
    "people laughing",
    "beautiful scenery",
    "action scene",
]
DEFAULT_NEGATIVE_PROMPTS = [
    "boring static shot",
    "blank wall",
    "empty room",
    "black screen",
]
MODEL_DIR = Path.home() / ".clipshow" / "models"
SAMPLE_FPS = 2  # Sample 2 frames per second

# Sigmoid normalization parameters for CLIP cosine similarities.
# Raw scores are typically 0.15-0.35 for real matches; this sigmoid
# stretches that range to fill [0, 1] with good contrast.
# This is a POSITIVE MINUS NEGATIVE margin, not a raw cosine similarity.
# Keep the scale absolute so a uniformly irrelevant clip cannot score 1.0.
_SIGMOID_CENTER = 0.03
_SIGMOID_SCALE = 20.0

# Temporal smoothing window size in score samples (not seconds).
# Reduces noise from single-frame outliers.
_SMOOTH_WINDOW = 5


class SemanticDetector(Detector):
    """CLIP-based semantic content scoring.

    Uses onnx_clip (ONNX Runtime) to score video frames against text prompts.
    Scores positive prompts against negative prompts for better contrast,
    applies sigmoid normalization, and smooths temporally.
    """

    name = "semantic"

    def __init__(
        self,
        prompts: list[str] | None = None,
        negative_prompts: list[str] | None = None,
        time_step: float = 0.1,
    ):
        self._prompts = prompts or DEFAULT_PROMPTS
        self._negative_prompts = negative_prompts or DEFAULT_NEGATIVE_PROMPTS
        self._time_step = time_step
        if time_step <= 0:
            raise ValueError("time_step must be positive")
        self._model = None

    def _load_model(self):
        """Lazy-load the CLIP model via onnx_clip."""
        try:
            importlib.import_module("onnx_clip")
        except ImportError:
            raise RuntimeError(
                "onnx_clip is not installed. Install with: "
                "pip install onnx_clip onnxruntime"
            )
        # Suppress noisy CoreML warnings on macOS (CLIP only supports 30/889
        # nodes via CoreML, so ORT falls back to CPU anyway).
        import onnxruntime as ort

        ort.set_default_logger_severity(3)  # 3 = ERROR only
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        # onnx_clip's original S3 weights bucket is no longer available.
        # Keep dependency validation above, but use checksum-verified exports.
        from clipshow.detection.local_clip import LocalClip
        self._model = LocalClip(batch_size=8)
        return self._model

    def detect(
        self,
        video_path: str,
        progress_callback: callable | None = None,
        cancel_flag: callable | None = None,
    ) -> DetectorResult:
        model = self._model or self._load_model()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0
        frame_interval = max(1, int(fps / SAMPLE_FPS))

        num_samples = max(1, int(np.ceil(duration / self._time_step)))
        scores = np.zeros(num_samples, dtype=float)

        # Encode text prompts once
        from PIL import Image

        pos_embeddings = model.get_text_embeddings(self._prompts)
        neg_embeddings = model.get_text_embeddings(self._negative_prompts)

        frame_idx = 0
        sample_times = []
        sample_scores = []
        while True:
            if cancel_flag and cancel_flag():
                break

            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                # Convert BGR to RGB PIL Image
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)

                image_embeddings = model.get_image_embeddings([img])

                # Positive vs negative contrast: best positive minus best negative
                pos_sim = float(np.max(image_embeddings @ pos_embeddings.T))
                neg_sim = float(np.max(image_embeddings @ neg_embeddings.T))
                raw = pos_sim - neg_sim

                # Sigmoid normalization for better dynamic range
                score = 1.0 / (1.0 + np.exp(-_SIGMOID_SCALE * (raw - _SIGMOID_CENTER)))
                score = float(np.clip(score, 0.0, 1.0))

                t = frame_idx / fps
                sample_times.append(t)
                sample_scores.append(score)

                if progress_callback:
                    progress_callback(frame_idx / max(total_frames, 1))

            frame_idx += 1

        cap.release()

        # Fill the timeline between measured samples. The old implementation
        # left four zero-valued holes between every pair of 2 FPS samples.
        if sample_times:
            target_times = np.arange(num_samples) * self._time_step
            scores = np.interp(target_times, sample_times, sample_scores)
            # Cancellation/decode failure must not extrapolate the final good
            # frame through the entire unobserved remainder of the video.
            observed_end = min(duration, frame_idx / fps)
            scores[target_times >= observed_end] = 0.0

        # Temporal smoothing to reduce single-frame noise
        if len(scores) >= _SMOOTH_WINDOW:
            scores = uniform_filter1d(scores, size=_SMOOTH_WINDOW)

        # Do not normalize by this video's maximum: that promotes the best
        # frame of a completely irrelevant video into a guaranteed highlight.
        if sample_times:
            scores[np.arange(num_samples) * self._time_step >= observed_end] = 0.0

        if progress_callback:
            progress_callback(1.0)

        return DetectorResult(
            name=self.name,
            scores=scores,
            time_step=self._time_step,
            source_path=video_path,
        )
