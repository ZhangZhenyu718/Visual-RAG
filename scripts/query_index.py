#!/usr/bin/env python3
"""Quick retrieval sanity check: encode a text query and print top-k segments.

    python scripts/query_index.py "a person opening a present" --modality visual -k 5
    python scripts/query_index.py "someone talking about dogs" --modality text
    python scripts/query_index.py "a baby on a sofa" --modality fused --alpha 0.5

Thin CLI over visualrag.retrieve.Retriever (same code path as the eval harness).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualrag.utils.config import load_config
from visualrag.retrieve.retriever import Retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--modality", choices=["visual", "text", "fused"], default="visual")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5, help="visual weight for fused")
    args = ap.parse_args()

    cfg = load_config(args.config)
    retriever = Retriever(cfg)
    hits = retriever.search(args.query, modality=args.modality, k=args.k, alpha=args.alpha)

    print(f"\nQuery: {args.query!r}  (modality={args.modality})\n")
    for r in hits:
        m = r["metadata"]
        print(f"  {r['score']:.3f}  {r['segment_id']}  "
              f"[{m.get('start')}-{m.get('end')}s, video {m.get('video_id')}]")


if __name__ == "__main__":
    main()
