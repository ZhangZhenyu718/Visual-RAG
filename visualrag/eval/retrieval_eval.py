"""Run retrieval evaluation over a NExT-GQA split.

For each grounded question: encode the question, retrieve top-K segments (per
modality), and score against the ground-truth moment(s). Aggregates overall and
by question type, for each of visual / text / fused — directly answering RQ1
(does retrieval work) and RQ2 (which modalities help).

Scope:
  - "corpus" (default): search across all indexed videos — the RQ1 setting.
  - "video": restrict to the GT video's own segments — pure temporal localization.
Only questions whose video has been indexed are evaluated (others are skipped and
reported), so partial indexes still give clean numbers.
"""

from __future__ import annotations

import glob
import os

from visualrag.schema import read_jsonl
from visualrag.data.nextqa import load_qa, load_grounding, grounding_key
from visualrag.retrieve.retriever import Retriever
from visualrag.eval.metrics import evaluate_question, aggregate, aggregate_by_type


def indexed_video_ids(cfg) -> set[str]:
    """Videos that have been embedded (npz present) — i.e. present in the index."""
    emb_dir = cfg.get_path("paths.embeddings", "artifacts/embeddings")
    return {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(emb_dir, "*.npz"))}


class _SegmentCache:
    """Lazily load a video's segment [start, end] intervals for ideal-DCG."""
    def __init__(self, cfg):
        self.seg_dir = os.path.join(cfg.get_path("paths.artifacts"), "segments")
        self._cache: dict[str, list] = {}

    def intervals(self, video_id: str) -> list:
        if video_id not in self._cache:
            path = os.path.join(self.seg_dir, f"{video_id}.jsonl")
            rows = read_jsonl(path) if os.path.exists(path) else []
            self._cache[video_id] = [(r["start"], r["end"]) for r in rows]
        return self._cache[video_id]


def run_eval(cfg, split="val", modalities=("visual", "text", "fused"),
             scope="corpus", k=10, alpha=0.5, tau=0.5, limit=0,
             decompose=False) -> dict:
    ann = cfg.get_path("paths.annotations")
    rows = load_qa(ann, split)
    gnd = load_grounding(ann, split)
    indexed = indexed_video_ids(cfg)

    # Evaluable = indexed video AND has a grounding label.
    evaluable = [r for r in rows
                 if r["video_id"] in indexed and grounding_key(r["video_id"], r["qid"]) in gnd]
    skipped = len(rows) - len(evaluable)
    if limit:
        evaluable = evaluable[:limit]

    ks = tuple(sorted({1, 5, 10, k}))
    seg_cache = _SegmentCache(cfg)
    retriever = Retriever(cfg)

    # Per-question query list: [question] baseline, or W5 LLM decomposition
    # (sub-queries + original, retrieved independently and RRF-fused).
    if decompose:
        from visualrag.retrieve.decompose import QueryDecomposer
        dec = QueryDecomposer(cfg, cache_name=split)
        sub = dec.decompose_batch([r["question"] for r in evaluable])
        qlists = []
        for r in evaluable:
            qs = list(sub.get(r["question"], []))
            if dec.include_original or not qs:
                qs.append(r["question"])
            qlists.append(qs)
    else:
        qlists = [[r["question"]] for r in evaluable]

    # Batch-encode all queries once; reuse vectors across modalities.
    flat = [q for qs in qlists for q in qs]
    flat_vecs = retriever.encoder.encode_texts(flat)
    offsets, pos = [], 0
    for qs in qlists:
        offsets.append((pos, pos + len(qs)))
        pos += len(qs)

    results: dict[str, dict] = {}
    for mod in modalities:
        per_q = []
        for i, r in enumerate(evaluable):
            where = {"video_id": r["video_id"]} if scope == "video" else None
            lo, hi = offsets[i]
            if hi - lo == 1:
                hits = retriever.search_vec(flat_vecs[lo], mod, k=max(ks), alpha=alpha, where=where)
            else:
                hits = retriever.search_vecs_rrf(flat_vecs[lo:hi], mod, k=max(ks),
                                                 alpha=alpha, where=where)
            g = gnd[grounding_key(r["video_id"], r["qid"])]
            m = evaluate_question(hits, r["video_id"], g["locations"],
                                  seg_cache.intervals(r["video_id"]), ks=ks, tau=tau)
            m["type"] = r["qtype"]
            per_q.append(m)
        results[mod] = {"overall": aggregate(per_q), "by_type": aggregate_by_type(per_q)}

    return {
        "split": split, "scope": scope, "tau": tau, "alpha": alpha,
        "decompose": decompose,
        "n_evaluated": len(evaluable), "n_skipped_not_indexed": skipped,
        "n_indexed_videos": len(indexed), "ks": list(ks),
        "results": results,
    }
