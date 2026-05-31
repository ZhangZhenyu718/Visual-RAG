#!/usr/bin/env python3
"""Batch ingest entry point — the one-time heavy job to run on a (cloud) GPU.

Loads the Whisper model once, iterates over the videos referenced by a NExT-QA
split, and produces per-video Segment JSONL under artifacts/segments/. Idempotent
and resumable (skips already-processed videos unless --overwrite).

Usage:
    python scripts/ingest_dataset.py --split val
    python scripts/ingest_dataset.py --split val --limit 20        # quick local smoke test
    python scripts/ingest_dataset.py --config configs/default.yaml --overwrite
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

from visualrag.utils.config import load_config, ensure_dirs
from visualrag.utils.device import resolve_device, describe_device
from visualrag.ingest.pipeline import ingest_video
from visualrag.ingest.transcribe import Transcriber
from visualrag.ingest.ocr import OCRReader


def find_video(videos_dir: str, video_id: str) -> str | None:
    for ext in (".mp4", ".mkv", ".webm", ".avi"):
        p = os.path.join(videos_dir, video_id + ext)
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(videos_dir, f"{video_id}.*"))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0, help="cap #videos (0 = all)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-asr", action="store_true", help="skip transcription (frames only)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    os.makedirs(os.path.join(cfg.get_path("paths.artifacts"), "segments"), exist_ok=True)

    device = resolve_device(cfg.get_path("device", "auto"))
    print(f"[ingest] device: {describe_device(device)}")

    videos_dir = cfg.get_path("paths.videos")
    ann_dir = cfg.get_path("paths.annotations")

    # Determine which videos to process: those referenced by the split, else all on disk.
    try:
        from visualrag.data.nextqa import list_video_ids
        video_ids = list_video_ids(ann_dir, args.split)
        print(f"[ingest] {len(video_ids)} videos referenced by split '{args.split}'")
    except FileNotFoundError as e:
        print(f"[ingest] no annotations ({e}); falling back to all videos in {videos_dir}")
        video_ids = sorted({os.path.splitext(os.path.basename(p))[0]
                             for p in glob.glob(os.path.join(videos_dir, "*.*"))})

    if args.limit:
        video_ids = video_ids[:args.limit]

    transcriber = None if args.no_asr else Transcriber(cfg)
    ocr_reader = OCRReader(cfg) if cfg.get_path("ocr.enabled", False) else None

    done, missing, total_segs = 0, 0, 0
    for i, vid in enumerate(video_ids, 1):
        vpath = find_video(videos_dir, vid)
        if vpath is None:
            missing += 1
            continue
        try:
            segs = ingest_video(vpath, cfg, transcriber, ocr_reader, overwrite=args.overwrite)
            total_segs += len(segs)
            done += 1
            print(f"[{i}/{len(video_ids)}] {vid}: {len(segs)} segments")
        except Exception as e:
            print(f"[{i}/{len(video_ids)}] {vid}: FAILED ({type(e).__name__}: {e})", file=sys.stderr)

    print(f"\n[ingest] done: {done} videos, {total_segs} segments, {missing} videos missing on disk")


if __name__ == "__main__":
    main()
