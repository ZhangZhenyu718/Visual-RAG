#!/usr/bin/env python3
"""YouCook2 speech-dense domain replication (ch6 §6.2 prediction test) — data prep.

Downloads N validation-split videos (the YouCook2 video id IS the YouTube id,
so ingest_dataset.py's basename matching works unchanged) and emits NExT-style
annotation files so the whole existing harness runs verbatim:

    annotations/val.csv       video_id, qid, type, question  (question = step sentence)
    annotations/gsub_val.json {video_id: {duration, location: {qid: [[s, e]]}}}

Resumable: already-downloaded videos are skipped; annotations are rewritten to
match whatever is on disk each run.

    python scripts/prep_youcook2.py --n 40
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YC2 = os.path.join(ROOT, "data", "youcook2")
ANN_JSON = os.path.join(YC2, "youcookii_annotations_trainval.json")

# Progressive formats only (audio+video in one file, no ffmpeg merge needed).
FORMAT = "18/best[height<=360][acodec!=none][vcodec!=none]/best[height<=480][acodec!=none][vcodec!=none]"


def load_val() -> dict[str, dict]:
    with open(ANN_JSON, encoding="utf-8") as f:
        db = json.load(f)["database"]
    return {k: v for k, v in sorted(db.items()) if v.get("subset") == "validation"}


def download(video_id: str, out_dir: str) -> bool:
    import yt_dlp
    out = os.path.join(out_dir, f"{video_id}.mp4")
    if os.path.exists(out) and os.path.getsize(out) > 1_000_000:
        return True
    opts = {
        "format": FORMAT,
        "outtmpl": out,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 3,
        "socket_timeout": 30,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        return os.path.exists(out) and os.path.getsize(out) > 1_000_000
    except Exception as e:
        print(f"  {video_id}: unavailable ({type(e).__name__})", flush=True)
        return False


def write_annotations(val: dict[str, dict], have: list[str]) -> int:
    ann_dir = os.path.join(YC2, "annotations")
    os.makedirs(ann_dir, exist_ok=True)
    n_q = 0
    gsub: dict[str, dict] = {}
    with open(os.path.join(ann_dir, "val.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "qid", "type", "question", "answer"])
        for vid in have:
            meta = val[vid]
            loc: dict[str, list] = {}
            for i, seg in enumerate(meta["annotations"]):
                s, e = float(seg["segment"][0]), float(seg["segment"][1])
                if e <= s:
                    continue
                w.writerow([vid, i, "YC", seg["sentence"], ""])
                loc[str(i)] = [[s, e]]
                n_q += 1
            gsub[vid] = {"duration": float(meta["duration"]), "location": loc}
    with open(os.path.join(ann_dir, "gsub_val.json"), "w", encoding="utf-8") as f:
        json.dump(gsub, f)
    return n_q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    val = load_val()
    print(f"[yc2] {len(val)} validation videos in annotations", flush=True)
    vid_dir = os.path.join(YC2, "videos")
    os.makedirs(vid_dir, exist_ok=True)

    have: list[str] = []
    for vid in val:
        if len(have) >= args.n:
            break
        print(f"[yc2] ({len(have)}/{args.n}) {vid} ...", flush=True)
        if download(vid, vid_dir):
            have.append(vid)

    n_q = write_annotations(val, have)
    print(f"[yc2] DONE: {len(have)} videos on disk, {n_q} step-queries "
          f"-> data/youcook2/annotations/{{val.csv, gsub_val.json}}", flush=True)
    if len(have) < args.n:
        print(f"[yc2] WARNING: only {len(have)}/{args.n} downloadable", file=sys.stderr)


if __name__ == "__main__":
    main()
