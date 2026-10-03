import hashlib

import numpy as np
import pytest

from clipshow.detection.local_clip import FILES, LocalClip, verified_model


def test_embeddings_normalized_before_cosine_similarity():
    embeddings = LocalClip.normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.isclose(np.linalg.norm(embeddings[0]), 1)
    assert np.all(np.isfinite(embeddings))
    assert np.all(embeddings[1] == 0)


def test_cached_model_checksum_fails_closed(tmp_path):
    filename = next(iter(FILES))
    (tmp_path / filename).write_bytes(b"invalid model")
    with pytest.raises(RuntimeError, match="checksum failed"):
        verified_model(tmp_path, filename)


def test_cached_verified_model_needs_no_network(tmp_path, monkeypatch):
    filename = next(iter(FILES))
    content = b"model fixture"
    (tmp_path / filename).write_bytes(content)
    monkeypatch.setitem(FILES, filename, hashlib.sha256(content).hexdigest())
    assert verified_model(tmp_path, filename) == tmp_path / filename


def test_unknown_model_filename_rejected(tmp_path):
    with pytest.raises(ValueError):
        verified_model(tmp_path, "../untrusted.onnx")
