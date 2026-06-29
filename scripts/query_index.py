#!/usr/bin/env python3
"""Quick retrieval sanity check: encode a text query and print top-k segments.

    python scripts/query_index.py "a person opening a present" --modality visual -k 5
    python scripts/query_index.py "someone talking about dogs" --modality text

Late fusion (`--modality fused`) combines visual + text scores per segment.
A fuller retrieval/re-ranking module lands in W3; this is a thin smoke test.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualrag.utils.config import load_config
from visualrag.embed.encoder import CLIPEncoder
from visualrag.index.chroma_store import ChromaStore


def fuse(visual_hits, text_hits, alpha: float):
    """Late fusion: score = alpha*visual + (1-alpha)*text, summed per segment_id."""
    scores: dict[str, float] = {}
    meta: dict[str, dict] = {}
    for h in visual_hits:
        scores[h["segment_id"]] = scores.get(h["segment_id"], 0.0) + alpha * h["score"]
        meta[h["segment_id"]] = h["metadata"]
    for h in text_hits:
        scores[h["segment_id"]] = scores.get(h["segment_id"], 0.0) + (1 - alpha) * h["score"]
        meta.setdefault(h["segment_id"], h["metadata"])
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [{"segment_id": sid, "score": sc, "metadata": meta[sid]} for sid, sc in ranked]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--modality", choices=["visual", "text", "fused"], default="visual")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5, help="visual weight for fused")
    args = ap.parse_args()

    cfg = load_config(args.config)
    enc = CLIPEncoder(cfg)
    store = ChromaStore(cfg)
    q = enc.encode_query(args.query)

    if args.modality == "fused":
        hits = fuse(store.query("visual", q, args.k * 2),
                    store.query("text", q, args.k * 2), args.alpha)[:args.k]
    else:
        hits = store.query(args.modality, q, args.k)

    print(f"\nQuery: {args.query!r}  (modality={args.modality})\n")
    for r in hits:
        m = r["metadata"]
        print(f"  {r['score']:.3f}  {r['segment_id']}  "
              f"[{m.get('start')}-{m.get('end')}s, video {m.get('video_id')}]")


if __name__ == "__main__":
    main()
