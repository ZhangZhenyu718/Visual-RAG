#!/usr/bin/env python3
"""Embed ingested segments and load them into ChromaDB.

Reads per-video Segment JSONL from artifacts/segments/, encodes visual + text
vectors (cached to artifacts/embeddings/<vid>.npz, resumable), and upserts both
modality collections. This is the second half of the one-time GPU job (after
ingest); the resulting Chroma dir is portable back to the local machine.

Usage:
    python scripts/build_index.py --split val
    python scripts/build_index.py --split val --limit 5     # quick smoke test
    python scripts/build_index.py --overwrite                # re-embed everything
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualrag.utils.config import load_config, ensure_dirs
from visualrag.schema import Segment, read_jsonl
from visualrag.embed.encoder import CLIPEncoder
from visualrag.embed.segment_embedder import embed_video_segments
from visualrag.index.chroma_store import ChromaStore


def _segment_files(cfg, split: str, limit: int) -> list[str]:
    seg_dir = os.path.join(cfg.get_path("paths.artifacts"), "segments")
    try:
        from visualrag.data.nextqa import list_video_ids
        vids = list_video_ids(cfg.get_path("paths.annotations"), split)
        files = [os.path.join(seg_dir, f"{v}.jsonl") for v in vids]
        files = [f for f in files if os.path.exists(f)]
    except FileNotFoundError:
        files = sorted(glob.glob(os.path.join(seg_dir, "*.jsonl")))
    return files[:limit] if limit else files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true", help="re-embed (ignore npz cache)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    files = _segment_files(cfg, args.split, args.limit)
    if not files:
        print("[index] no segment files found — run scripts/ingest_dataset.py first.")
        return
    print(f"[index] {len(files)} videos to embed/index")

    encoder = CLIPEncoder(cfg)
    store = ChromaStore(cfg)

    n_visual, n_text = 0, 0
    for i, f in enumerate(files, 1):
        video_id = os.path.splitext(os.path.basename(f))[0]
        segments = [Segment(**r) for r in read_jsonl(f)]
        if not segments:
            continue
        emb = embed_video_segments(video_id, segments, encoder, cfg, overwrite=args.overwrite)

        seg_ids = list(emb["segment_ids"])
        by_id = {s.segment_id: s for s in segments}
        metas = []
        for sid in seg_ids:
            s = by_id[sid]
            metas.append({"video_id": s.video_id, "start": s.start, "end": s.end,
                          "n_frames": len(s.keyframe_paths)})

        v_mask = emb["has_visual"]
        t_mask = emb["has_text"]
        if v_mask.any():
            n_visual += store.upsert(
                "visual",
                [seg_ids[j] for j in range(len(seg_ids)) if v_mask[j]],
                emb["visual"][v_mask],
                [metas[j] for j in range(len(seg_ids)) if v_mask[j]],
                documents=[by_id[seg_ids[j]].text for j in range(len(seg_ids)) if v_mask[j]],
            )
        if t_mask.any():
            n_text += store.upsert(
                "text",
                [seg_ids[j] for j in range(len(seg_ids)) if t_mask[j]],
                emb["text"][t_mask],
                [metas[j] for j in range(len(seg_ids)) if t_mask[j]],
                documents=[by_id[seg_ids[j]].text for j in range(len(seg_ids)) if t_mask[j]],
            )
        print(f"[{i}/{len(files)}] {video_id}: +{int(v_mask.sum())} visual, +{int(t_mask.sum())} text")

    print(f"\n[index] done. collections: visual={store.count('visual')}, text={store.count('text')} "
          f"(this run: +{n_visual} visual, +{n_text} text)")


if __name__ == "__main__":
    main()
