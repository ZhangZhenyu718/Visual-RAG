"""Retrieval metrics for temporal-grounded video search.

Relevance is graded by temporal IoU (tIoU) between a retrieved segment's
[start, end] and the question's ground-truth moment(s):

    gain(segment) = max tIoU over GT intervals   if segment is from the GT video
                  = 0                             otherwise

From these graded gains we derive:
  - Recall@K / MRR : binary hit at a tIoU threshold τ (default 0.5)
  - nDCG@K         : graded, ideal-normalized against the GT video's own segments
  - tIoU@1         : tIoU of the top-1 hit — the plan's "temporal accuracy"

Requiring the hit to be both the right video AND overlap the right moment is what
ties this to RQ1 (retrieve the relevant segment, not merely the right video).
"""

from __future__ import annotations

import math


def tiou(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Temporal Intersection-over-Union of two [start, end] intervals."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def max_tiou(interval: tuple[float, float], gt_intervals: list) -> float:
    return max((tiou(interval, (g[0], g[1])) for g in gt_intervals), default=0.0)


def _gain(hit: dict, gt_video: str, gt_intervals: list) -> float:
    m = hit.get("metadata", {})
    if str(m.get("video_id")) != str(gt_video):
        return 0.0
    return max_tiou((float(m.get("start", 0.0)), float(m.get("end", 0.0))), gt_intervals)


def evaluate_question(ranked_hits: list, gt_video: str, gt_intervals: list,
                      gt_video_segments: list, ks=(1, 5, 10), tau: float = 0.5) -> dict:
    """Per-question metrics. `gt_video_segments` = all indexed segments of the GT
    video (as [start, end] pairs), used to compute the ideal DCG for nDCG."""
    gains = [_gain(h, gt_video, gt_intervals) for h in ranked_hits]
    rels = [1.0 if g >= tau else 0.0 for g in gains]

    out: dict[str, float] = {}
    for k in ks:
        out[f"R@{k}"] = 1.0 if any(rels[:k]) else 0.0

    # MRR (over the returned ranking; first relevant at threshold τ)
    rr = 0.0
    for i, r in enumerate(rels):
        if r > 0:
            rr = 1.0 / (i + 1)
            break
    out["MRR"] = rr

    # nDCG@K with graded (tIoU) gains, normalized by the best achievable ordering.
    K = max(ks)
    dcg = sum(gains[i] / math.log2(i + 2) for i in range(min(K, len(gains))))
    ideal = sorted((max_tiou((s[0], s[1]), gt_intervals) for s in gt_video_segments), reverse=True)
    idcg = sum(ideal[i] / math.log2(i + 2) for i in range(min(K, len(ideal))))
    out[f"nDCG@{K}"] = (dcg / idcg) if idcg > 0 else 0.0

    # Temporal accuracy of the top result.
    out["tIoU@1"] = gains[0] if gains else 0.0
    return out


def aggregate(per_question: list[dict]) -> dict:
    """Mean of each metric over a list of per-question metric dicts (ignores the
    non-numeric 'type' field)."""
    if not per_question:
        return {}
    keys = [k for k, v in per_question[0].items() if isinstance(v, (int, float))]
    return {k: sum(q[k] for q in per_question) / len(per_question) for k in keys}


def aggregate_by_type(per_question: list[dict]) -> dict[str, dict]:
    """Same aggregation, grouped by question `type` (CW/TN/CH/TC/TP...)."""
    groups: dict[str, list] = {}
    for q in per_question:
        groups.setdefault(q.get("type", "?"), []).append(q)
    return {t: {**aggregate(qs), "n": len(qs)} for t, qs in groups.items()}
