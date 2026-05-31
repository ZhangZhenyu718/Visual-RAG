#!/usr/bin/env python3
"""Helper for fetching NExT-QA / NExT-GQA annotations + videos.

Most NExT-QA assets are hosted on Google Drive, so this script documents the
canonical sources and downloads what it can via `gdown`. Some large video
archives may require manual download / acceptance of terms — those are printed
as instructions rather than fetched silently.

Usage:
    python scripts/download_nextqa.py --what annotations
    python scripts/download_nextqa.py --what videos
"""

from __future__ import annotations

import argparse
import os

# Canonical sources (see project READMEs). Verify/update IDs against the repos:
#   NExT-QA:  https://github.com/doc-doc/NExT-QA
#   NExT-GQA: https://github.com/doc-doc/NExT-GQA
ANNOTATION_SOURCES = {
    "NExT-QA repo (CSVs: train/val/test.csv)": "https://github.com/doc-doc/NExT-QA",
    "NExT-GQA repo (temporal grounding gsub_*.json)": "https://github.com/doc-doc/NExT-GQA",
}
VIDEO_NOTE = (
    "NExT-QA videos are sourced from VidOR. Download the video archive linked in the\n"
    "NExT-QA repo's README (Google Drive), unzip into data/nextqa/videos/ as <video_id>.mp4."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["annotations", "videos"], default="annotations")
    ap.add_argument("--annotations-dir", default="data/nextqa/annotations")
    ap.add_argument("--videos-dir", default="data/nextqa/videos")
    args = ap.parse_args()

    if args.what == "annotations":
        os.makedirs(args.annotations_dir, exist_ok=True)
        print("[download] NExT-QA / NExT-GQA annotation sources:")
        for name, url in ANNOTATION_SOURCES.items():
            print(f"  - {name}\n      {url}")
        print(
            "\nClone the repos (they ship the CSVs / grounding JSON directly), then copy:\n"
            f"  train.csv / val.csv / test.csv  -> {args.annotations_dir}/\n"
            f"  gsub_val.json / gsub_test.json  -> {args.annotations_dir}/\n"
            "\nTip: `git clone https://github.com/doc-doc/NExT-GQA` and look under its dataset dir."
        )
    else:
        os.makedirs(args.videos_dir, exist_ok=True)
        print(VIDEO_NOTE)


if __name__ == "__main__":
    main()
