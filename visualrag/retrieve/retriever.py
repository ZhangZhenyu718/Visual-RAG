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


def rrf(hit_lists: list[list[dict]], k_rrf: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion across several rankings of the same collection.

    score(seg) = sum over lists of 1/(k_rrf + rank). Rank-based, so it is robust
    to the (incomparable) score scales of different sub-queries — the standard
    choice for multi-query retrieval (W5 query decomposition)."""
    scores: dict[str, float] = {}
    meta: dict[str, dict] = {}
    docs: dict[str, str] = {}
    for hits in hit_lists:
        for rank, h in enumerate(hits):
            sid = h["segment_id"]
            scores[sid] = scores.get(sid, 0.0) + 1.0 / (k_rrf + rank + 1)
            meta.setdefault(sid, h["metadata"])
            if h.get("document"):
                docs.setdefault(sid, h["document"])
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [{"segment_id": sid, "score": sc, "metadata": meta[sid],
             "document": docs.get(sid, "")} for sid, sc in ranked]


def fuse(visual_hits: list[dict], text_hits: list[dict], alpha: float) -> list[dict]:
    """Sum weighted per-segment scores across modalities, re-rank descending."""
    scores: dict[str, float] = {}
    meta: dict[str, dict] = {}
    docs: dict[str, str] = {}
    for h in visual_hits:
        sid = h["segment_id"]
        scores[sid] = scores.get(sid, 0.0) + alpha * h["score"]
        meta[sid] = h["metadata"]
        if h.get("document"):
            docs.setdefault(sid, h["document"])
    for h in text_hits:
        sid = h["segment_id"]
        scores[sid] = scores.get(sid, 0.0) + (1 - alpha) * h["score"]
        meta.setdefault(sid, h["metadata"])
        if h.get("document"):
            docs.setdefault(sid, h["document"])
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [{"segment_id": sid, "score": sc, "metadata": meta[sid],
             "document": docs.get(sid, "")} for sid, sc in ranked]


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

    def search_vecs_rrf(self, qvecs, modality: str = "visual", k: int = 10,
                        alpha: float = 0.5, where: Optional[dict] = None) -> list[dict]:
        """Multi-query retrieval (W5): search each pre-encoded sub-query vector,
        RRF-fuse the rankings, return top-k."""
        hit_lists = [self.search_vec(v, modality, k * self.fusion_overfetch, alpha, where)
                     for v in qvecs]
        return rrf(hit_lists)[:k]

    def search_multi(self, queries: list[str], modality: str = "visual", k: int = 10,
                     alpha: float = 0.5, where: Optional[dict] = None) -> list[dict]:
        return self.search_vecs_rrf(self.encoder.encode_texts(queries), modality, k, alpha, where)
