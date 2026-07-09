# Experiment Log

Every headline number in the README, with the exact command that produced it and
where the raw output lives. All runs: NExT-QA/NExT-GQA **val split** (3358
grounded questions, 567 videos, 5725 segments), local RTX 4050 (6GB).
Raw JSONs are committed under [`results/`](../results/) (copied from the
git-ignored `artifacts/`).

Environment: conda env `visualrag` (Python 3.11, torch 2.13+cu126). API keys via
env vars `DEEPSEEK_API_KEY` / `ANTHROPIC_API_KEY` (set with `setx`, read at run
time; never hardcode in scripts).

## Preprocessing (one-time, 2026-07-06/07)

```bash
python scripts/ingest_dataset.py --split val        # frames + Whisper large-v3 int8 (GPU via CTranslate2), ~2h
python scripts/build_index.py    --split val        # CLIP ViT-B-32 -> ChromaDB: 5725 visual + 3433 text vectors
python scripts/embed_backbone.py                    # ViT-L-14 segment vectors for the W6 visual reranker (~10 min GPU)
```

Long-running jobs on this machine: background shells get reaped — use the
detached watchdog pattern (`scripts/ingest_watchdog.ps1`, self-healing restart
loop; ingest/build are idempotent so restarts are safe).

## W3 — retrieval baselines (2026-07-07)

```bash
python scripts/evaluate.py --split val                --json artifacts/eval_val_corpus.json
python scripts/evaluate.py --split val --scope video  --json artifacts/eval_val_video.json
# alpha sweep (fusion weight): --modalities fused --alpha 0.7|0.8|0.9
```

Corpus scope, visual: R@1 .026 / R@5 .075 / R@10 .111 / MRR .047 / tIoU@1 .035.
Video scope, visual: R@1 .148 / R@5 .394 / R@10 .491 / MRR .250 / tIoU@1 .203.
Text weak everywhere (multilingual transcripts vs English CLIP text encoder);
fused ≤ visual at every alpha (0.5–0.9) in both scopes → naive late fusion is a
clean negative result (`results/eval_val_corpus_a0.*.json`).

## W5 — LLM query decomposition + RRF (2026-07-09)

```bash
python scripts/evaluate.py --split val --modalities visual --decompose --json ...
python scripts/evaluate.py --split val --modalities visual --decompose --scope video --json ...
```

Decompositions: deepseek-chat, all 3358 questions (~$0.15), cache committed at
`results/decompositions_val.json` (runtime location:
`artifacts/decompositions/val.json` — copy back to re-run for free).
Corpus: R@10 .111→.122 (+10% rel), MRR +6%; R@1 flat. Video: R@5 .394→.405.
Decomposition broadens deep-rank recall, not top-1.

## W6 — second-stage re-ranking (2026-07-09)

Text cross-encoder (bge-reranker-v2-m3, `rerank.method: cross_encoder`):
**hurts at every weight tested** (100-q subset: R@5 .070→.030 at w=0.5, still
below baseline at w=0.8) — transcripts are chatter, uncorrelated with the
visual events questions ask about. Implementation lesson: candidates the
reranker has no opinion on (no text / no vector) must have the reranker term
imputed from their retrieval rank, or they are structurally demoted.

Visual reranker (ViT-L-14 over top-30, `rerank.method: visual`, default):

```bash
python scripts/evaluate.py --split val --modalities visual [--decompose] --rerank [--scope video] --json ...
```

Best config (decompose + visual rerank), corpus: R@1 .030 / R@10 .128 / MRR .055
(+15–17% rel vs W3); video: R@1 .154 / R@10 .501. Components compose additively.

## W7 — agent QA accuracy (2026-07-09)

```bash
python scripts/eval_qa.py --limit 150 --agent simple --json artifacts/qa_simple_150.json
python scripts/eval_qa.py --limit 150 --agent graph  --json artifacts/qa_graph_150.json
```

5-choice accuracy, first 150 val questions, DeepSeek text-only:
simple .447 → graph (LangGraph ReAct + reflection + temporal tool) **.547**
(CW +13pt, TC +18pt, CH +19pt). TN flat at .341 for both.

## Vision experiment — same 44 TN questions, multimodal (2026-07-09)

```bash
python scripts/eval_qa.py --types TN --limit 44 --agent simple --provider claude \
    --workers 3 --json artifacts/qa_claude_tn44.json
```

claude-opus-4-8, keyframes in tool results, temporal tool enabled: TN
**.341 → .636** (+87% rel, 0 errors, ~$3). Same questions, same retrieval, same
tools — only the evidence channel changed. Qualitative flagship (video
2834146886, "what does the white dog do after going to the cushion", GT "smells
the black dog"): DeepSeek answered "segments are silent, cannot tell"; Claude
answered "sniffs/nuzzles the small black puppy [28–36s]".

## Demo assets

- 4-video smoke-test corpus (`configs/demo.yaml`, `data/demo/videos/`, git-ignored):
  - bunny/jellyfish/sintel: `https://test-videos.co.uk/vids/{bigbuckbunny,jellyfish,sintel}/mp4/h264/360/*_360_10s_1MB.mp4`
  - nasa_snowflake: `https://images-assets.nasa.gov/video/GSFC_20180329_3DModel_m12908_Snoflake_Melt/GSFC_20180329_3DModel_m12908_Snoflake_Melt~small.mp4`
- Static demo snapshot: [`docs/demo/visualrag_demo.html`](demo/visualrag_demo.html)
  (regenerate: `run_demo_queries.py` then `make_demo_page.py`); hosted copy:
  https://claude.ai/code/artifact/2186f18f-72f1-492c-8777-c1e7dcd2165c
- Interactive demo: `streamlit run ui/app.py`
