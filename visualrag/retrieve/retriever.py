"""Retriever: encode a query, search per-modality, late-fuse. Shared by the eval
harness and the query CLI so fusion logic lives in one place.

Modalities:
  - "visual" / "text": single-collection cosine search.
  - "fused": query both, combine per-segment scores as
        alpha * visual + (1 - alpha) * text
    (late fusion — the RQ2-friendly default; early fusion can be added later).
"""

from __future__ import annotations

from typing import Optional
import numpy as np


def fuse(visual_hits: list[dict], text_hits: list[dict], alpha: float) -> list[dict]:
    """Sum weighted per-segment scores across modalities, re-rank descending."""
    scores: dict[str, float] = {}
    meta: dict[str, dict] = {}
    for h in visual_hits:
        sid = h["segment_id"]
        scores[sid] = scores.get(sid, 0.0) + alpha * h["score"]
        meta[sid] = h["metadata"]
    for h in text_hits:
        sid = h["segment_id"]
        scores[sid] = scores.get(sid, 0.0) + (1 - alpha) * h["score"]
        meta.setdefault(sid, h["metadata"])
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [{"segment_id": sid, "score": sc, "metadata": meta[sid]} for sid, sc in ranked]


class Retriever:
    def __init__(self, cfg, encoder=None, store=None):
        if encoder is None:
            from visualrag.embed.encoder import CLIPEncoder
            encoder = CLIPEncoder(cfg)
        if store is None:
            from visualrag.index.chroma_store import ChromaStore
            store = ChromaStore(cfg)
        self.encoder = encoder
        self.store = store
        # over-fetch per modality before fusion so fused top-k isn't starved.
        self.fusion_overfetch = 3

    def search_vec(self, qvec: np.ndarray, modality: str = "fused", k: int = 10,
                   alpha: float = 0.5, where: Optional[dict] = None) -> list[dict]:
        """Search with a pre-computed query vector (skips encoding — used in eval
        where all questions are batch-encoded once)."""
        if modality in ("visual", "text"):
            return self.store.query(modality, qvec, k, where)
        if modality == "fused":
            m = k * self.fusion_overfetch
            v = self.store.query("visual", qvec, m, where)
            t = self.store.query("text", qvec, m, where)
            return fuse(v, t, alpha)[:k]
        raise ValueError(f"unknown modality {modality!r}")

    def search(self, query: str, modality: str = "fused", k: int = 10,
               alpha: float = 0.5, where: Optional[dict] = None) -> list[dict]:
        return self.search_vec(self.encoder.encode_query(query), modality, k, alpha, where)
