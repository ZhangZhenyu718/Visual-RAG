#!/usr/bin/env python3
"""Download NExT-QA / NExT-GQA benchmark assets.

Annotations (QA CSVs + temporal grounding JSON + id maps) live directly in the
NExT-GQA GitHub repo, so we fetch them straight from raw.githubusercontent.com —
no Google Drive, no git clone needed. The raw videos are a large Google-Drive
archive, downloaded via `gdown` (best-effort; Drive quota can require a manual
retry, in which case we print the link).

Usage:
    python scripts/download_nextqa.py                      # annotations (default)
    python scripts/download_nextqa.py --what annotations
    python scripts/download_nextqa.py --what videos        # large; needs gdown
    python scripts/download_nextqa.py --what all
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

# Files shipped in https://github.com/doc-doc/NExT-GQA/tree/main/datasets/nextgqa
RAW_BASE = "https://raw.githubusercontent.com/doc-doc/NExT-GQA/main/datasets/nextgqa"
ANNOTATION_FILES = [
    "train.csv",            # GQA QA subset
    "val.csv",
    "test.csv",
    "gsub_val.json",        # temporal grounding labels (seconds), keyed by video_id
    "gsub_test.json",
    "frame2time_val.json",  # frame-index -> timestamp (for grounding eval)
    "frame2time_test.json",
    "map_vid_vidorID.json", # video_id -> VidOR id (needed to match raw videos)
]

# Raw videos (NExTVideo) — Google Drive archive, see NExT-QA repo README.
VIDEO_DRIVE_ID = "1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH"
VIDEO_DRIVE_URL = f"https://drive.google.com/file/d/{VIDEO_DRIVE_ID}/view"


def _download(url: str, dest: str) -> bool:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        with open(dest, "wb") as f:
            f.write(data)
        print(f"  ok  {os.path.basename(dest)}  ({len(data)/1024:.0f} KB)")
        return True
    except Exception as e:
        print(f"  FAIL {os.path.basename(dest)}: {type(e).__name__}: {e}", file=sys.stderr)
        return False


def download_annotations(ann_dir: str) -> None:
    print(f"[annotations] -> {ann_dir}")
    ok = 0
    for fname in ANNOTATION_FILES:
        if _download(f"{RAW_BASE}/{fname}", os.path.join(ann_dir, fname)):
            ok += 1
    print(f"[annotations] {ok}/{len(ANNOTATION_FILES)} files downloaded")
    if ok < len(ANNOTATION_FILES):
        print(f"  (any failures: fetch manually from {RAW_BASE})")


def download_videos(videos_dir: str) -> None:
    os.makedirs(videos_dir, exist_ok=True)
    zip_path = os.path.join(videos_dir, "NExTVideo.zip")
    print(f"[videos] downloading archive via gdown -> {zip_path}")
    try:
        import gdown
    except ImportError:
        print("  gdown not installed. `pip install gdown`, then re-run.", file=sys.stderr)
        print(f"  Or download manually: {VIDEO_DRIVE_URL}")
        return
    try:
        gdown.download(id=VIDEO_DRIVE_ID, output=zip_path, quiet=False)
    except Exception as e:
        print(f"  gdown failed ({e}).", file=sys.stderr)
        print(f"  Download manually and unzip into {videos_dir}: {VIDEO_DRIVE_URL}")
        return
    print(f"  downloaded. Unzip into {videos_dir}/ (videos are named by VidOR id;")
    print("  use map_vid_vidorID.json to map QA video_id -> file).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["annotations", "videos", "all"], default="annotations")
    ap.add_argument("--annotations-dir", default="data/nextqa/annotations")
    ap.add_argument("--videos-dir", default="data/nextqa/videos")
    args = ap.parse_args()

    if args.what in ("annotations", "all"):
        download_annotations(args.annotations_dir)
    if args.what in ("videos", "all"):
        download_videos(args.videos_dir)


if __name__ == "__main__":
    main()
