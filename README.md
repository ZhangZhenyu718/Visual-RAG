# Visual RAG — Autonomous Visual Search Agent for Video Retrieval

MSc summer project. Multi-modal (visual + audio + OCR) retrieval-augmented
generation over video, with temporal-precise grounding. See
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the full design.

**Status:** W1 — ingestion skeleton + environment.

## Architecture (one line)

Offline heavy compute (Whisper + visual embeddings, on a cloud GPU) → embeddings +
ChromaDB index → online query time stays light (6GB local + LLM API).

## Layout

```
visualrag/
  schema.py        Segment / Keyframe / TranscriptChunk dataclasses
  utils/           device auto-detect (cuda/mps/cpu), config loader
  ingest/          frames (PySceneDetect), transcribe (faster-whisper), ocr, segment, pipeline
  data/            NExT-QA / NExT-GQA loaders
scripts/
  download_nextqa.py   fetch/locate benchmark annotations + videos
  ingest_dataset.py    batch ingest entry point (the one-time GPU job)
configs/default.yaml   all knobs (models, paths, windowing)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Quick start

```bash
# 1. Get benchmark (prints sources; clone the NExT-QA / NExT-GQA repos for CSV + grounding)
python scripts/download_nextqa.py --what annotations
python scripts/download_nextqa.py --what videos

# 2. Smoke-test ingest on a few videos locally (frames-only, no GPU needed)
python scripts/ingest_dataset.py --split val --limit 5 --no-asr

# 3. Full ingest with ASR (run on a GPU; device auto-detected)
python scripts/ingest_dataset.py --split val
```

Artifacts land under `artifacts/` (frames, transcripts, `segments/<video_id>.jsonl`),
both git-ignored.

## Benchmark

Primary: **NExT-QA** (causal/temporal multiple-choice QA) + **NExT-GQA**
(temporal grounding labels → drives the Temporal Accuracy metric).

## Roadmap

- **W1 (now):** ingest skeleton + env ✅
- **W2:** batch Whisper + visual embeddings on GPU → ChromaDB
- **W3:** retrieval baselines (Recall@K / MRR / nDCG), text-only vs multimodal
- **W4:** MVP vertical slice (query → retrieve → grounded answer)
- **W5–7:** query decomposition, re-ranking, LangGraph agent
- **W8–12:** evaluation, ablations, open-source LLM comparison, demo, dissertation
