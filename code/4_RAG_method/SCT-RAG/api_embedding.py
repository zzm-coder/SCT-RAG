"""Small OpenAI-compatible embedding client used by indexing and retrieval."""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import requests


class APIEmbeddingModel:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        self.model = model
        self.base_url = (base_url or os.environ.get("SCT_EMBEDDING_API_BASE_URL") or os.environ.get("SCT_API_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("SCT_EMBEDDING_API_KEY") or os.environ.get("SCT_API_KEY", "")
        if not self.base_url or not self.model:
            raise ValueError("SCT_EMBEDDING_API_BASE_URL and SCT_EMBEDDING_MODEL are required")
        self._dimension: int | None = None

    def get_sentence_embedding_dimension(self) -> int:
        if self._dimension is None:
            self._dimension = int(self.encode(["dimension probe"]).shape[1])
        return self._dimension

    def encode(self, texts, batch_size: int = 64, normalize_embeddings: bool = True,
               convert_to_numpy: bool = True, show_progress_bar: bool = False):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        rows: list[list[float]] = []
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for start in range(0, len(items), max(1, batch_size)):
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json={"model": self.model, "input": items[start:start + batch_size]},
                timeout=180,
            )
            response.raise_for_status()
            data = sorted(response.json()["data"], key=lambda row: row.get("index", 0))
            rows.extend(row["embedding"] for row in data)
        array = np.asarray(rows, dtype="float32")
        if array.ndim == 2 and array.shape[1]:
            self._dimension = int(array.shape[1])
        if normalize_embeddings and array.size:
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            array = array / np.maximum(norms, 1e-12)
        result = array[0] if single else array
        return result if convert_to_numpy else result.tolist()

    def embed_query(self, text: str):
        return self.encode(text).tolist()

    def embed_documents(self, texts: list[str]):
        return self.encode(texts).tolist()

    async def aembed_query(self, text: str):
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]):
        return self.embed_documents(texts)
