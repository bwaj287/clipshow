"""Local CLIP using pinned, SHA256-verified Xenova ONNX exports.

The upstream onnx_clip weights bucket returns 404. Reuse its MIT tokenizer
and preprocessing, but load available ONNX exports without running remote code.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

REVISION = "d15189d7028b43f1d3e65039190477f6af591c2a"
FILES = {
    "vision_model_quantized.onnx":
        "583fd1110a514667812fee7d684952aaf82a99b959760c8d7dca7e0ab9839299",
    "text_model_quantized.onnx":
        "73baab855d406190da9faa498cfedf65f15cf309f4cc7385b7b032e6d08e5c3a",
}


def verified_model(directory, filename):
    if filename not in FILES:
        raise ValueError("unknown model file")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    if not path.exists():
        import requests

        partial = path.with_suffix(path.suffix + ".part")
        url = (f"https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/"
               f"{REVISION}/onnx/{filename}")
        with requests.get(url, stream=True, timeout=(15, 60)) as response:
            response.raise_for_status()
            with partial.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
        with partial.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != FILES[filename]:
                raise RuntimeError(f"Model checksum failed: {partial}")
        partial.replace(path)
    with path.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != FILES[filename]:
            raise RuntimeError(f"Model checksum failed: {path}")
    return path


class LocalClip:
    def __init__(self, directory=None, batch_size=8):
        import onnxruntime as ort
        from onnx_clip.preprocessor import Preprocessor
        from onnx_clip.tokenizer import Tokenizer

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        directory = Path(directory or os.environ.get(
            "CLIPSHOW_MODEL_DIR", str(Path.home() / ".clipshow" / "models")))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 8
        options.inter_op_num_threads = 1
        self.image = ort.InferenceSession(
            str(verified_model(directory, "vision_model_quantized.onnx")),
            sess_options=options, providers=["CPUExecutionProvider"])
        self.text = ort.InferenceSession(
            str(verified_model(directory, "text_model_quantized.onnx")),
            sess_options=options, providers=["CPUExecutionProvider"])
        self.preprocessor = Preprocessor()
        self.tokenizer = Tokenizer()
        self.batch_size = batch_size

    @staticmethod
    def normalize(vectors):
        return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)

    def get_image_embeddings(self, images):
        images = list(images)
        results = []
        for start in range(0, len(images), self.batch_size):
            batch = np.concatenate([self.preprocessor.encode_image(image)
                                    for image in images[start:start + self.batch_size]])
            results.append(self.image.run(["image_embeds"], {"pixel_values": batch})[0])
        return self.normalize(np.concatenate(results)) if results else np.empty((0, 512))

    def get_text_embeddings(self, texts):
        tokens = self.tokenizer.encode_text(list(texts)).astype(np.int64)
        inputs = {"input_ids": tokens}
        if any(node.name == "attention_mask" for node in self.text.get_inputs()):
            inputs["attention_mask"] = (tokens != 0).astype(np.int64)
        vectors = self.text.run(["text_embeds"], inputs)[0]
        return self.normalize(vectors)
