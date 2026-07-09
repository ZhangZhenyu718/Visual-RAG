#!/usr/bin/env python3
"""Precompute segment embeddings with an alternative CLIP backbone.

Used by the W6 visual reranker (second-stage scoring with a stronger model)
and the W8 backbone ablation. Writes per-video npz to --out; the main index
(embed.backbone in the config) is untouched.

    python scripts/embed_backbone.py --backbone ViT-L-14 \\
        --pretrained laion2b_s32b_b82k --out artifacts/embeddings_vitl
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualrag.utils.config import load_config, Config
from visualrag.schema import Segment, read_jsonl
from visualrag.embed.encoder import CLIPEncoder
from visualrag.embed.segment_embedder import embed_video_segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--backbone", default="ViT-L-14")
    ap.add_argument("--pretrained", default="laion2b_s32b_b82k")
    ap.add_argument("--out", default="artifacts/embeddings_vitl")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    base = load_config(args.config)
    cfg = Config({**base,
                  "embed": {"backbone": args.backbone, "pretrained": args.pretrained,
                            "batch_size": args.batch_size},
                  "paths": {**base["paths"], "embeddings": args.out}})

    files = sorted(glob.glob(os.path.join(cfg.get_path("paths.artifacts"), "segments", "*.jsonl")))
    if args.limit:
        files = files[: args.limit]
    print(f"[embed-backbone] {len(files)} videos -> {args.out} ({args.backbone}/{args.pretrained})")

    encoder = CLIPEncoder(cfg)
    for i, f in enumerate(files, 1):
        video_id = os.path.splitext(os.path.basename(f))[0]
        segments = [Segment(**r) for r in read_jsonl(f)]
        if segments:
            embed_video_segments(video_id, segments, encoder, cfg)
        if i % 25 == 0 or i == len(files):
            print(f"[embed-backbone] {i}/{len(files)}")


if __name__ == "__main__":
    main()
