#!/usr/bin/env python3
"""Retrieval evaluation CLI — RQ1 (retrieval works?) and RQ2 (which modalities?).

    python scripts/evaluate.py --split val                       # visual/text/fused
    python scripts/evaluate.py --split val --modalities visual    # one modality
    python scripts/evaluate.py --split val --scope video          # localization only
    python scripts/evaluate.py --split val --tau 0.3 --json out.json

Prints a modality comparison table plus a by-question-type breakdown, and
optionally dumps the full result dict to JSON for the dissertation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualrag.utils.config import load_config
from visualrag.eval.retrieval_eval import run_eval

METRIC_ORDER = ["R@1", "R@5", "R@10", "MRR", "nDCG@10", "tIoU@1"]


def _fmt_row(label, m, width=10):
    cells = "".join(f"{m.get(k, 0.0):>9.3f}" for k in METRIC_ORDER if k in m)
    return f"  {label:<{width}}{cells}"


def _header(width=10):
    cols = "".join(f"{k:>9}" for k in METRIC_ORDER)
    return f"  {'':<{width}}{cols}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--modalities", nargs="+", default=["visual", "text", "fused"],
                    choices=["visual", "text", "fused"])
    ap.add_argument("--scope", choices=["corpus", "video"], default="corpus")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.5, help="visual weight in fusion")
    ap.add_argument("--tau", type=float, default=0.5, help="tIoU threshold for a hit")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--decompose", action="store_true",
                    help="W5: LLM query decomposition + RRF multi-query retrieval")
    ap.add_argument("--json", default=None, help="dump full results to this path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    res = run_eval(cfg, split=args.split, modalities=tuple(args.modalities),
                   scope=args.scope, k=args.k, alpha=args.alpha, tau=args.tau,
                   limit=args.limit, decompose=args.decompose)

    print(f"\n=== Retrieval eval | split={res['split']} scope={res['scope']} "
          f"tIoU>={res['tau']} alpha={res['alpha']} decompose={res['decompose']} ===")
    print(f"evaluated {res['n_evaluated']} questions over {res['n_indexed_videos']} indexed videos "
          f"({res['n_skipped_not_indexed']} skipped: video not indexed)\n")

    print(_header())
    for mod in args.modalities:
        print(_fmt_row(mod, res["results"][mod]["overall"]))

    # By-type breakdown for the fused (or first) modality.
    focus = "fused" if "fused" in args.modalities else args.modalities[0]
    print(f"\n  by question type ({focus}):")
    print(_header(width=12))
    by_type = res["results"][focus]["by_type"]
    for t in sorted(by_type, key=lambda x: -by_type[x]["n"]):
        print(_fmt_row(f"{t} (n={by_type[t]['n']})", by_type[t], width=12))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\n[eval] full results -> {args.json}")


if __name__ == "__main__":
    main()
