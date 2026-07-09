#!/usr/bin/env python3
"""W4 MVP CLI: ask a question, get a grounded answer with timestamp citations.

    python scripts/answer.py "what is the video about?" --video 4882821564
    python scripts/answer.py "why did the boy move to the sofa?" --video 4882821564
    python scripts/answer.py "a video explaining how snowflakes melt" --config configs/demo.yaml
    python scripts/answer.py "..." --dry-run          # retrieval context only, no LLM call

Requires ANTHROPIC_API_KEY (or an `ant auth login` profile) unless --dry-run.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualrag.utils.config import load_config
from visualrag.agent.answerer import VideoQA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--video", default=None, help="restrict search to one video_id")
    ap.add_argument("--modality", choices=["visual", "text", "fused"], default=None)
    ap.add_argument("-k", type=int, default=None)
    ap.add_argument("--model", default=None, help="override agent.model")
    ap.add_argument("--dry-run", action="store_true",
                    help="print retrieved context and exit (no API key needed)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    qa = VideoQA(cfg)
    if args.model:
        qa.model = args.model
    if args.modality:
        qa.modality = args.modality
    if args.k:
        qa.k = args.k

    if args.dry_run:
        hits = qa.search(args.question, video_id=args.video)
        print(f"\nRetrieved context for {args.question!r} (modality={qa.modality}):\n")
        n_img = 0
        for b in qa.hits_to_blocks(hits):
            if b["type"] == "text":
                print(b["text"])
            else:
                n_img += 1
                print(f"  <keyframe image #{n_img}: {len(b['source']['data']) * 3 // 4} bytes jpeg>")
        return

    result = qa.answer(args.question, video_id=args.video)

    print(f"\n{'=' * 70}\nQ: {args.question}\n{'=' * 70}\n")
    print(result["answer"])
    print(f"\n{'-' * 70}")
    for s in result["searches"]:
        print(f"searched [{s['modality']}]: {s['query']!r}")
        for h in s["hits"][:5]:
            m = h["metadata"]
            print(f"    {h['score']:.3f}  {m.get('video_id')} [{m.get('start')}-{m.get('end')}s]")
    u = result["usage"]
    print(f"tokens: {u['input_tokens']} in / {u['output_tokens']} out")


if __name__ == "__main__":
    main()
