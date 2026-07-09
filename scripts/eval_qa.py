#!/usr/bin/env python3
"""NExT-QA multiple-choice accuracy for the answer agents (W4 simple / W7 graph).

Each question is posed with its 5 options; the agent gathers evidence with its
tools and must end with "FINAL: <index>". Accuracy overall and by question type
(CW/CH causal, TN/TC/TP temporal — the temporal types are where the W6/W7
temporal tool should pay off).

    python scripts/eval_qa.py --limit 150 --agent simple --json artifacts/qa_simple.json
    python scripts/eval_qa.py --limit 150 --agent graph  --json artifacts/qa_graph.json

Concurrent (--workers); DeepSeek cost ~ $0.2 / 150 questions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualrag.utils.config import load_config
from visualrag.data.nextqa import load_qa

MC_TEMPLATE = """Answer this multiple-choice question about the video by gathering \
evidence with your tools.

Question: {question}

Options:
{options}

After your evidence-based reasoning, the LAST line of your reply must be exactly
"FINAL: <option number>" (0-4). If the evidence is inconclusive, pick the most
plausible option — never refuse."""


def parse_choice(text: str) -> int | None:
    m = re.findall(r"FINAL:\s*([0-4])", text)
    if m:
        return int(m[-1])
    m = re.findall(r"\b([0-4])\b", text.strip().splitlines()[-1] if text.strip() else "")
    return int(m[-1]) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--agent", choices=["simple", "graph"], default="simple")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--types", default=None, help="comma list, e.g. TN,TC,TP")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--provider", choices=["deepseek", "claude"], default=None,
                    help="override agent.provider")
    ap.add_argument("--model", default=None, help="override agent.model")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.provider:
        cfg.setdefault("agent", {})["provider"] = args.provider
        cfg["agent"].pop("model", None)  # fall back to the provider's default model
    if args.model:
        cfg.setdefault("agent", {})["model"] = args.model
    rows = load_qa(cfg.get_path("paths.annotations"), args.split)
    if args.types:
        wanted = {t.strip() for t in args.types.split(",")}
        rows = [r for r in rows if r["qtype"] in wanted]
    rows = rows[: args.limit] if args.limit else rows

    if args.agent == "graph":
        from visualrag.agent.graph_agent import GraphVideoQA
        qa = GraphVideoQA(cfg)
    else:
        from visualrag.agent.answerer import VideoQA
        qa = VideoQA(cfg)

    def ask(r: dict) -> dict:
        options = "\n".join(f"{i}. {c}" for i, c in enumerate(r["choices"]))
        prompt = MC_TEMPLATE.format(question=r["question"], options=options)
        gt = r["choices"].index(r["answer"]) if r["answer"] in r["choices"] else -1
        try:
            res = qa.answer(prompt, video_id=r["video_id"])
            choice = parse_choice(res["answer"])
            usage = res["usage"]
        except Exception as e:
            return {**{k: r[k] for k in ("video_id", "qid", "qtype")},
                    "gt": gt, "choice": None, "correct": False, "error": str(e)}
        return {**{k: r[k] for k in ("video_id", "qid", "qtype")},
                "gt": gt, "choice": choice, "correct": choice == gt,
                "tokens_out": usage["output_tokens"]}

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(ask, r) for r in rows]
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 20 == 0:
                acc = sum(x["correct"] for x in results) / len(results)
                print(f"[qa] {i}/{len(rows)} acc so far {acc:.3f}")

    acc = sum(x["correct"] for x in results) / max(len(results), 1)
    errors = sum(1 for x in results if x.get("error"))
    print(f"\n=== QA accuracy | agent={args.agent} n={len(results)} "
          f"(errors {errors}) ===")
    print(f"overall: {acc:.3f}")
    by_type: dict[str, list] = {}
    for x in results:
        by_type.setdefault(x["qtype"], []).append(x)
    for t, xs in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        print(f"  {t} (n={len(xs)}): {sum(x['correct'] for x in xs) / len(xs):.3f}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"agent": args.agent, "n": len(results), "accuracy": acc,
                       "results": results}, f, ensure_ascii=False, indent=1)
        print(f"[qa] -> {args.json}")


if __name__ == "__main__":
    main()
