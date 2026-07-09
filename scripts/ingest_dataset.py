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
import json
import os
import sys

# Allow running as `python scripts/ingest_dataset.py` without `pip install -e .`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualrag.utils.config import load_config, ensure_dirs
from visualrag.utils.device import resolve_device, describe_device
from visualrag.ingest.pipeline import ingest_video
from visualrag.ingest.transcribe import Transcriber
from visualrag.ingest.ocr import OCRReader

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".avi", ".mov")


def build_video_index(videos_dir: str) -> dict[str, str]:
    """Recursively map basename-without-extension -> full path.

    NExT-QA videos live at <videos_dir>/NExTVideo/<group>/<vidorID>.mp4, and the
    file's basename equals the QA `video_id`, so a one-pass basename index resolves
    them regardless of the nested group dirs."""
    index: dict[str, str] = {}
    for root, _dirs, files in os.walk(videos_dir):
        for fn in files:
            stem, ext = os.path.splitext(fn)
            if ext.lower() in VIDEO_EXTS:
                index.setdefault(stem, os.path.join(root, fn))
    return index


def load_id_map(annotations_dir: str) -> dict[str, str]:
    """video_id -> '<group>/<vidorID>' (fallback resolver if basenames differ)."""
    p = os.path.join(annotations_dir, "map_vid_vidorID.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def resolve_video(video_id: str, index: dict[str, str], id_map: dict[str, str]) -> str | None:
    if video_id in index:
        return index[video_id]
    # Fallback: map to VidOR id, match on its basename.
    mapped = id_map.get(video_id)
    if mapped:
        return index.get(os.path.basename(mapped))
    return None


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

    video_index = build_video_index(videos_dir)
    id_map = load_id_map(ann_dir)
    print(f"[ingest] indexed {len(video_index)} video files on disk")

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
        vpath = resolve_video(vid, video_index, id_map)
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
