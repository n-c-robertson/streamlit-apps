"""Local in-memory semantic search over fetched classroom content.

Uses OpenAI text-embedding-3-small + numpy cosine similarity. The index is
rebuilt per program and cached by the Streamlit session.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
EMBED_BATCH = 200


class CorpusIndex:
    def __init__(self, chunks: list[dict[str, Any]], vectors: np.ndarray):
        self.chunks = chunks
        self.vectors = vectors  # shape (n, d), L2-normalized

    def search(
        self,
        query_vec: np.ndarray,
        *,
        project_key: str | None = None,
        k: int = 8,
    ) -> list[dict[str, Any]]:
        if self.vectors.size == 0:
            return []
        q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
        sims = self.vectors @ q
        order = np.argsort(-sims)
        results: list[dict[str, Any]] = []
        for idx in order:
            chunk = self.chunks[idx]
            if project_key is not None and chunk.get("project_key") not in (None, project_key):
                continue
            results.append({**chunk, "score": float(sims[idx])})
            if len(results) >= k:
                break
        return results


def _embed(client: OpenAI, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    all_vecs: list[np.ndarray] = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start : start + EMBED_BATCH]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_vecs.extend(np.asarray(d.embedding, dtype=np.float32) for d in resp.data)
    mat = np.stack(all_vecs)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def embed_query(client: OpenAI, query: str) -> np.ndarray:
    return _embed(client, [query])[0]


def build_index(client: OpenAI, chunks: list[dict[str, Any]]) -> CorpusIndex:
    texts = [c["text"] for c in chunks]
    vectors = _embed(client, texts)
    return CorpusIndex(chunks, vectors)
