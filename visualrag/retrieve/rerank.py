"""W6: cross-encoder re-ranking over retrieved candidates.

W5 showed query decomposition lifts deep-rank recall but not top-1: the right
segment reaches the top-10 more often, but CLIP's coarse similarity doesn't
order it first. A cross-encoder reads (question, transcript) jointly and
re-scores the candidate list.

Fusion design: the CE only sees text, and 40% of segments have no transcript —
so the CE ranking is weighted-RRF-fused with the retrieval ranking instead of
replacing it. Text-less segments simply don't appear in the CE ranking and
keep the score their retrieval rank earned; they are never demoted for lacking
speech.

Default model is bge-reranker-v2-m3 (multilingual — NExT-QA transcripts are
mixed-language while questions are English). `BAAI/bge-reranker-base` (EN/ZH,
the plan's original pick) stays available via `rerank.model` as an ablation.

RESULT NOTE (W6, val subset): the text CE *hurts* on NExT-QA at every tested
weight — transcripts are chatter, largely uncorrelated with the visual events
the questions ask about, so CE reorderings inject noise. Kept for the
dissertation ablation; `rerank.method: visual` (VisualReranker) is the default.
"""

from __future__ import annotations

import glob
import os
from typing import Optional

import numpy as np

from visualrag.utils.device import resolve_device


def make_reranker(cfg):
    method = cfg.get_path("rerank.method", "visual")
    if method == "visual":
        return VisualReranker(cfg)
    if method in ("cross_encoder", "ce"):
        return CrossEncoderReranker(cfg)
    raise ValueError(f"unknown rerank.method {method!r}")


class VisualReranker:
    """Second-stage visual scoring with a stronger CLIP backbone (default
    ViT-L-14) over precomputed segment embeddings (`scripts/embed_backbone.py`).

    Every segment has keyframes, so unlike the text cross-encoder there is no
    missing-modality problem — and on NExT-QA the visual channel is the one
    that carries signal (W3). Candidate score = max over sub-query vectors of
    cosine vs the segment's pooled ViT-L embedding; the ViT-L ranking is
    weighted-RRF-fused with the first-stage retrieval ranking."""

    def __init__(self, cfg):
        self.backbone = cfg.get_path("rerank.visual_backbone", "ViT-L-14")
        self.pretrained = cfg.get_path("rerank.visual_pretrained", "laion2b_s32b_b82k")
        self.emb_dir = cfg.get_path("rerank.visual_embeddings", "artifacts/embeddings_vitl")
        self.candidates = int(cfg.get_path("rerank.candidates", 30))
        self.retrieval_weight = float(cfg.get_path("rerank.retrieval_weight", 0.5))
        self.k_rrf = 60
        self._cfg = cfg
        self._encoder = None
        self._vecs: Optional[dict] = None

    def _ensure(self):
        if self._encoder is None:
            from visualrag.embed.encoder import CLIPEncoder
            from visualrag.utils.config import Config
            sub = Config({**self._cfg, "embed": {
                "backbone": self.backbone, "pretrained": self.pretrained, "batch_size": 16}})
            self._encoder = CLIPEncoder(sub)
        if self._vecs is None:
            self._vecs = {}
            for p in glob.glob(os.path.join(self.emb_dir, "*.npz")):
                z = np.load(p, allow_pickle=True)
                for sid, vec, has in zip(z["segment_ids"], z["visual"], z["has_visual"]):
                    if has:
                        self._vecs[str(sid)] = vec.astype(np.float32)
            if not self._vecs:
                raise FileNotFoundError(
                    f"no visual-reranker embeddings in {self.emb_dir} — "
                    f"run scripts/embed_backbone.py first")
            print(f"[rerank] visual reranker: {len(self._vecs)} segment vectors "
                  f"({self.backbone}/{self.pretrained})")

    def rerank(self, question: str, hits: list[dict], k: int = 10,
               queries: Optional[list[str]] = None) -> list[dict]:
        if not hits:
            return hits
        self._ensure()

        qvecs = self._encoder.encode_texts(queries or [question])  # [n, D], normalized
        sims: dict[int, float] = {}
        for i, h in enumerate(hits):
            vec = self._vecs.get(h["segment_id"])
            if vec is not None:
                sims[i] = float(np.max(qvecs @ vec))
        order = sorted(sims, key=lambda i: -sims[i])
        vis_rank = {i: r for r, i in enumerate(order)}

        w = self.retrieval_weight
        fused = []
        for i, h in enumerate(hits):
            r_vis = vis_rank.get(i, i)  # no precomputed vec -> defer to retrieval rank
            s = w / (self.k_rrf + i + 1) + (1 - w) / (self.k_rrf + r_vis + 1)
            out = dict(h)
            out["score"] = s
            if i in sims:
                out["rerank_sim"] = sims[i]
            fused.append(out)
        fused.sort(key=lambda h: -h["score"])
        return fused[:k]


class CrossEncoderReranker:
    def __init__(self, cfg):
        self.model_name = cfg.get_path("rerank.model", "BAAI/bge-reranker-v2-m3")
        self.candidates = int(cfg.get_path("rerank.candidates", 30))
        self.retrieval_weight = float(cfg.get_path("rerank.retrieval_weight", 0.5))
        self.batch_size = int(cfg.get_path("rerank.batch_size", 32))
        self.k_rrf = 60
        self.device = resolve_device(cfg.get_path("device", "auto"))
        self._model = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            print(f"[rerank] loading {self.model_name} on {self.device}")
            self._model = CrossEncoder(self.model_name, device=self.device,
                                       max_length=512)

    def rerank(self, question: str, hits: list[dict], k: int = 10,
               queries: Optional[list[str]] = None) -> list[dict]:
        """Re-order retrieval hits by weighted RRF of (retrieval rank, CE rank).

        `hits` must be retrieval-ordered and carry `document` (the segment's
        transcript+OCR text). `queries` is accepted for interface parity with
        VisualReranker; the CE always scores against the original question.
        """
        if not hits:
            return hits
        self._ensure()

        with_text = [(i, h) for i, h in enumerate(hits) if (h.get("document") or "").strip()]
        ce_rank: dict[int, int] = {}
        ce_score: dict[int, float] = {}
        if with_text:
            pairs = [(question, h["document"]) for _, h in with_text]
            scores = self._model.predict(pairs, batch_size=self.batch_size,
                                         show_progress_bar=False)
            order = sorted(range(len(with_text)), key=lambda j: -float(scores[j]))
            for rank, j in enumerate(order):
                idx = with_text[j][0]
                ce_rank[idx] = rank
                ce_score[idx] = float(scores[j])

        w = self.retrieval_weight
        fused = []
        for i, h in enumerate(hits):
            # Text-less hits: the CE has no opinion, so its term defers to the
            # retrieval rank. Keeps both hit classes on the same score scale —
            # otherwise every CE-liked text hit outranks ALL visual-only hits.
            r_ce = ce_rank.get(i, i)
            s = w / (self.k_rrf + i + 1) + (1 - w) / (self.k_rrf + r_ce + 1)
            out = dict(h)
            out["score"] = s
            if i in ce_score:
                out["ce_score"] = ce_score[i]
            fused.append(out)
        fused.sort(key=lambda h: -h["score"])
        return fused[:k]
