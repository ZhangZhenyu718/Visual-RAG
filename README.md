# Visual RAG — Autonomous Visual Search Agent for Video Retrieval

MSc summer project. Multi-modal (visual + audio + OCR) retrieval-augmented
generation over video, with temporal-precise grounding. See
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the full design.

**Status:** W3 ✅ — full val evaluated (567 videos / 5725 segments; frames + Whisper large-v3 int8 on a local RTX 4050, CLIP ViT-B-32). Corpus scope: visual R@10 0.111 / MRR 0.047. Video scope (localization): visual R@5 0.394 / tIoU@1 0.203. Text modality is weak (multilingual transcripts vs English CLIP text encoder) and drags late fusion below visual-only at every alpha — fixing the text pipeline precedes fusion gains. Full numbers: `artifacts/eval_val_{corpus,video}.json`.

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

# 4. Embed segments + build the ChromaDB index (visual + text collections)
python scripts/build_index.py --split val            # caches embeddings to artifacts/embeddings/
python scripts/query_index.py "a baby on a sofa" --modality fused -k 5   # retrieval smoke test

# 5. Evaluate retrieval on NExT-GQA grounding (RQ1/RQ2)
python scripts/evaluate.py --split val --json artifacts/eval_val.json
```

Artifacts land under `artifacts/` (frames, transcripts, `segments/<video_id>.jsonl`),
both git-ignored.

## Benchmark

Primary: **NExT-QA** (causal/temporal multiple-choice QA) + **NExT-GQA**
(temporal grounding labels → drives the Temporal Accuracy metric).

## Roadmap

- **W1:** ingest skeleton + env ✅
- **W2:** visual + text embeddings (open_clip) → ChromaDB, late-fusion retrieval ✅
- **W3:** retrieval eval harness (Recall@K / MRR / nDCG / tIoU) on NExT-GQA grounding, visual vs text vs fused ✅ (full-val numbers in Status above; alpha sweep shows naive fusion never beats visual-only)
- **W4:** MVP vertical slice ✅ — `scripts/answer.py`: question → LLM function-calling
  (`search_video_segments` tool) → grounded answer with `[video_id @ start-end s]`
  citations. Providers: `deepseek` (text-only, default) / `claude` (multimodal keyframes).
  Verified on NExT-QA val (answers the benchmark's first causal question correctly).
- **W5:** LLM query decomposition + RRF multi-query retrieval ✅ (`--decompose` on
  `evaluate.py`; cached in `artifacts/decompositions/`). Lifts deep-rank recall —
  corpus R@10 0.111→0.122 (+10% rel), MRR +6% rel; R@1 unchanged → the missing top-1
  precision is W6 re-ranking's job.
- **W6:** second-stage re-ranking ✅ (`--rerank`). Text cross-encoder (bge-reranker)
  *hurts* on NExT-QA — transcripts are chatter, uncorrelated with the visual events
  questions ask about (kept as ablation, `rerank.method: cross_encoder`). The default
  **visual reranker** (ViT-L-14 re-scores top-30 candidates, precompute via
  `scripts/embed_backbone.py`) composes with W5 decomposition: corpus R@10
  0.111→0.128, MRR 0.047→0.055 (+17% rel), video R@10 crosses 0.5. Full ladder in
  `artifacts/eval_val_*_{decomp,rerank,decomp_rerank}.json`.
- **W6b+W7:** temporal tool + LangGraph agent ✅ — `get_segments_around` (anchor an
  event via search, then walk before/after) + ReAct/self-reflection state machine
  (`visualrag/agent/graph_agent.py`, `scripts/answer.py --agent graph`). NExT-QA
  multiple-choice accuracy (150 val questions, `scripts/eval_qa.py`, text-only
  DeepSeek): simple 0.447 → graph **0.547** (+10pt; causal CW +13pt, TC +18pt).
  TN stuck at 0.341 for both — temporal-next answers are visual actions the
  text-only provider cannot see: the measured ceiling that motivates the
  multimodal (`provider: claude`) path.
- **W8+ (next):** backbone ablations (ViT-L/SigLIP as index), open-source LLM
  comparison, tau sensitivity, demo UI
- **W8–12:** evaluation, ablations, open-source LLM comparison, demo, dissertation
