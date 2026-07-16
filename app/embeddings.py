"""Embedding providers.

Two implementations, one interface:

- LocalEmbedder: sentence-transformers, runs on CPU, no API key, free forever. This is the
  default and the one that matters. Paying an API per embedding for a document corpus is a
  choice, not a requirement.
- FakeEmbedder: deterministic hash-based vectors. Not semantically meaningful, but stable and
  instant, which is exactly what the ACL tests need. Those tests are about who can see what, and
  they should not need a 90MB model download or a network call to run.

The interface is one method so swapping providers is a config change, not a refactor.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

DIMS = 384  # all-MiniLM-L6-v2. If you change the model, change the schema's vector(384) too.


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FakeEmbedder:
    """Deterministic, offline, instant. For tests.

    Hashes text into a fixed-dimension unit vector. Same text always gives the same vector, and
    different text gives a different one, which is all a retrieval test needs. It is NOT semantic:
    "cat" and "kitten" are unrelated here. Any test that depends on semantic similarity should use
    the real embedder or, better, not be a unit test.
    """

    dims = DIMS

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        # Expand a digest into DIMS floats by rehashing with a counter.
        vec: list[float] = []
        counter = 0
        while len(vec) < DIMS:
            h = hashlib.sha256(f"{text}:{counter}".encode()).digest()
            vec.extend(b / 255.0 - 0.5 for b in h)
            counter += 1
        vec = vec[:DIMS]
        return _normalize(vec)


class LocalEmbedder:
    """sentence-transformers on CPU. Free, no key, no network after the first download.

    Loaded lazily so that importing this module (which the tests do) does not pull in torch.
    """

    dims = DIMS

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # imported lazily on purpose

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        # normalize_embeddings=True so cosine distance in pgvector behaves, and so the schema's
        # vector_cosine_ops index is the right choice.
        arr = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, row)) for row in arr]


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def get_embedder(kind: str = "local", model: str = "sentence-transformers/all-MiniLM-L6-v2") -> Embedder:
    if kind == "fake":
        return FakeEmbedder()
    if kind == "local":
        return LocalEmbedder(model)
    raise ValueError(f"unknown embedder: {kind!r} (expected 'local' or 'fake')")
