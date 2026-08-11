# -*- coding: utf-8 -*-
"""Paired significance tests for the two central RQ3 claims (supervisor request).

Comparison A: simple loop vs LangGraph agent, same 150 grounded val questions.
Comparison B: text-only vs multimodal evidence, same 44 TN questions.

Both are paired on identical question sets, so we report:
  - exact McNemar test (two-sided binomial on the discordant pairs), and
  - a paired bootstrap 95% CI on the accuracy difference (10k resamples, seed 0).

Usage:  python scripts/stats_tests.py  [--results results/]
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def load(path: Path) -> dict[tuple[str, str], bool]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {(r["video_id"], str(r["qid"])): bool(r["correct"]) for r in data["results"]}


def mcnemar_exact(a: dict, b: dict) -> dict:
    """Exact two-sided McNemar on paired binary outcomes (binomial, p=0.5)."""
    keys = sorted(set(a) & set(b))
    b01 = sum(1 for k in keys if a[k] and not b[k])   # a right, b wrong
    b10 = sum(1 for k in keys if not a[k] and b[k])   # a wrong, b right
    n = b01 + b10
    if n == 0:
        p = 1.0
    else:
        lo = min(b01, b10)
        tail = sum(math.comb(n, i) for i in range(0, lo + 1)) / 2 ** n
        p = min(1.0, 2 * tail)
    return {"n_pairs": len(keys), "discordant_a_only": b01, "discordant_b_only": b10,
            "exact_p": p}


def bootstrap_diff_ci(a: dict, b: dict, iters: int = 10_000, seed: int = 0) -> dict:
    keys = sorted(set(a) & set(b))
    rng = random.Random(seed)
    n = len(keys)
    diffs = []
    for _ in range(iters):
        sample = [keys[rng.randrange(n)] for _ in range(n)]
        acc_a = sum(a[k] for k in sample) / n
        acc_b = sum(b[k] for k in sample) / n
        diffs.append(acc_b - acc_a)
    diffs.sort()
    return {"diff": sum(b[k] for k in keys) / n - sum(a[k] for k in keys) / n,
            "ci95": [diffs[int(0.025 * iters)], diffs[int(0.975 * iters) - 1]]}


def report(name: str, base: dict, treat: dict) -> None:
    mc = mcnemar_exact(base, treat)
    bs = bootstrap_diff_ci(base, treat)
    acc = lambda d: sum(v for k, v in d.items() if k in set(base) & set(treat)) / mc["n_pairs"]
    print(f"\n== {name} ==")
    print(f"  n paired questions : {mc['n_pairs']}")
    print(f"  accuracy           : {acc(base):.3f} -> {acc(treat):.3f}  (diff {bs['diff']:+.3f})")
    print(f"  discordant pairs   : base-only-right {mc['discordant_a_only']}, "
          f"treat-only-right {mc['discordant_b_only']}")
    print(f"  McNemar exact p    : {mc['exact_p']:.4f}")
    print(f"  bootstrap 95% CI   : [{bs['ci95'][0]:+.3f}, {bs['ci95'][1]:+.3f}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    rd = Path(args.results)

    simple = load(rd / "qa_simple_150.json")
    graph = load(rd / "qa_graph_150.json")
    claude_tn = load(rd / "qa_claude_tn44.json")

    report("A. Agent: simple loop -> LangGraph (150 questions, text-only LLM)",
           simple, graph)

    tn_keys = set(claude_tn)
    simple_tn = {k: v for k, v in simple.items() if k in tn_keys}
    graph_tn = {k: v for k, v in graph.items() if k in tn_keys}
    report("B1. Evidence: text-only simple agent -> multimodal (44 TN questions)",
           simple_tn, claude_tn)
    report("B2. Evidence: text-only graph agent -> multimodal (44 TN questions)",
           graph_tn, claude_tn)

    # C. Same model, prior only vs full best-config stack (both claude, 150 q).
    prior_c = rd / "qa_prior_claude_150.json"
    best = rd / "qa_best_150.json"
    if prior_c.exists() and best.exists():
        report("C. Evidence stack: claude prior-only -> best config (150 questions)",
               load(prior_c), load(best))

    # D. Same model, prior only vs the text-only agent stacks (deepseek, 150 q).
    prior_d = rd / "qa_prior_deepseek_150.json"
    if prior_d.exists():
        dp = load(prior_d)
        report("D1. Evidence stack: deepseek prior-only -> simple agent (150 questions)",
               dp, simple)
        report("D2. Evidence stack: deepseek prior-only -> graph agent (150 questions)",
               dp, graph)
        dp_tn = {k: v for k, v in dp.items() if k in tn_keys}
        report("D3. TN subset: deepseek prior-only -> graph agent (44 TN questions)",
               dp_tn, graph_tn)


if __name__ == "__main__":
    main()
