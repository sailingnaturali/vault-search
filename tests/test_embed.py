import pytest

from vault_search.embed import Embedder


def test_embed_dim_and_determinism():
    emb = Embedder()                       # default bge-small-en-v1.5
    vecs = emb.encode(["sailing vessel lights", "sailing vessel lights"])
    assert len(vecs) == 2
    assert len(vecs[0]) == emb.dim == 384
    # identical text -> identical vector (ONNX is deterministic)
    assert vecs[0] == vecs[1]


def test_unknown_model_dim_raises():
    with pytest.raises(ValueError):
        Embedder("not-a-real-model").dim
