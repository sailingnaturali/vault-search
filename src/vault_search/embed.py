"""In-process embeddings via fastembed (ONNX). No torch, no daemon."""

from __future__ import annotations

from functools import cached_property

from fastembed import TextEmbedding

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
_DIMS = {"BAAI/bge-small-en-v1.5": 384}


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name

    @cached_property
    def _model(self) -> TextEmbedding:
        return TextEmbedding(model_name=self.model_name)

    @property
    def dim(self) -> int:
        return _DIMS.get(self.model_name, 384)

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(list(texts))]
