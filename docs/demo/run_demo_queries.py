"""Run demo queries against the demo index and dump rich results to JSON."""
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from visualrag.utils.config import load_config
from visualrag.retrieve.retriever import Retriever
from visualrag.schema import read_jsonl

cfg = load_config("configs/demo.yaml")
retriever = Retriever(cfg)

# segment_id -> {text, keyframes}
seg_info = {}
seg_dir = os.path.join(cfg.get_path("paths.artifacts"), "segments")
for fn in os.listdir(seg_dir):
    for r in read_jsonl(os.path.join(seg_dir, fn)):
        txt = (r.get("transcript", "") + " " + r.get("ocr_text", "")).strip()
        seg_info[r["segment_id"]] = {"text": txt, "keyframes": r.get("keyframe_paths", [])}

QUERIES = [
    ("a cartoon rabbit in a green forest", "visual"),
    ("jellyfish swimming underwater", "visual"),
    ("a snowy mountain landscape", "visual"),
    ("scientists explain how snowflakes melt", "fused"),
    ("a 3D computer model", "fused"),
    ("radar observations of precipitation", "text"),
]

out = []
for query, modality in QUERIES:
    hits = retriever.search(query, modality=modality, k=3)
    rows = []
    for h in hits:
        info = seg_info.get(h["segment_id"], {})
        kfs = info.get("keyframes", [])
        rows.append({
            "segment_id": h["segment_id"],
            "score": round(h["score"], 3),
            "video_id": h["metadata"].get("video_id"),
            "start": h["metadata"].get("start"),
            "end": h["metadata"].get("end"),
            "text": info.get("text", ""),
            "keyframe": kfs[len(kfs) // 2] if kfs else None,
        })
    out.append({"query": query, "modality": modality, "hits": rows})
    print(f"[{modality:6s}] {query!r}: top -> {rows[0]['video_id']} [{rows[0]['start']}-{rows[0]['end']}s] score {rows[0]['score']}")

dst = sys.argv[1] if len(sys.argv) > 1 else "demo_results.json"
with open(dst, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("saved ->", dst)
