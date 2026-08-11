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

## W8 — ablations (2026-07-10)

**Backbone as index** (corpus scope, visual, tau 0.5; `configs/ablation_{vitl,siglip}.yaml`;
ViT-L reuses the W6 reranker embeddings, SigLIP embedded via
`scripts/embed_backbone.py --backbone ViT-SO400M-14-SigLIP-384 --pretrained webli`):

| index backbone | R@1 | R@5 | R@10 | MRR | tIoU@1 |
|---|---|---|---|---|---|
| ViT-B-32 (baseline) | .026 | .075 | .111 | .047 | .035 |
| ViT-B + ViT-L rerank (W6) | .029 | .087 | .121 | .053 | .041 |
| ViT-L-14 | .029 | .090 | .122 | .054 | .040 |
| SigLIP SO400M | **.044** | **.106** | **.143** | **.070** | **.054** |
| SigLIP + decompose (final best) | **.048** | **.110** | **.161** | **.076** | **.055** |

Findings: (1) SigLIP dominates for retrieval (R@1 +69% rel vs ViT-B; sigmoid-loss
pretraining is known to favour retrieval); (2) two-stage ViT-B+ViT-L-rerank ≈
ViT-L-as-index to within ±.003 — the cheap-index/expensive-rerank design recovers
the big model's quality at this corpus scale; (3) final best system (SigLIP index
+ W5 decomposition) nearly doubles corpus R@1 vs the W3 baseline (.026→.048) and
reaches video-scope R@1 .176 / tIoU@1 .230. Re-ranking with the same model as the
index is an identity op — the reranker must differ from the index backbone.
Default config stays ViT-B-32 for provenance of earlier numbers; switch
`configs/default.yaml` embed/paths to the SigLIP values for the best system.

**tau sensitivity** (ViT-B baseline and best ViT-B config, tau 0.3 vs 0.5):
baseline corpus R@10 .111→.186, video R@5 .394→.660 / R@10 .491→.774; best
config (decompose+rerank) video R@5 .677 / R@10 .780. tau=0.5 is a strict
criterion for 8s windows; gains from W5/W6 hold at both thresholds.

**Open-source LLM plumbing**: `agent.provider: local` targets any
OpenAI-compatible endpoint (`agent.base_url`, default Ollama at
`http://localhost:11434/v1`). QA comparison: `scripts/eval_qa.py --provider local
--model qwen2.5:7b-instruct` once Ollama is installed. Not yet run (no Ollama on
this machine). Encoder fix that unblocked SigLIP: `CLIPEncoder.dim` now probes
visual.output_dim → embed_dim → dummy text encode (SigLIP timm towers).

## Best-configuration QA run — SOTA comparison (2026-07-14)

```bash
python scripts/eval_qa.py --config configs/agent_best.yaml --agent graph \
    --limit 150 --workers 3 --json artifacts/qa_best_150.json
```

configs/agent_best.yaml = SigLIP index + ViT-L visual rerank inside the agent's
search tool (`agent.use_rerank`) + LangGraph agent + claude-opus-4-8 multimodal
(graph agent gained Anthropic support for this run). Result: **.727** overall
(95% CI [.655, .798], 0 errors, ≈$25): CW .852 / CH .762 / TC .682 / TN .568.
Published zero-shot anchors (full val): VideoAgent .713, LLoVi ≈.677, SeViLA-ZS
≈.636. Grounding anchors (NExT-GQA test, answer-grounding mIoU): SeViLA .217,
Temp[CLIP] NG+ .121 vs our val-set retrieval tIoU@1 .230 (SigLIP+decompose).
Claim discipline: "at, and in point estimate above, published zero-shot SOTA
on our 150-question grounded sample" — full-split run (~$250) would settle it.

## Paired significance tests (supervisor revision, 2026-08-05)

```bash
python scripts/stats_tests.py
```

Exact McNemar + paired bootstrap (10k resamples, seed 0) over the per-question
outcomes already in `results/`. Agent effect (simple→graph, 150 paired q):
29 flips up vs 14 down, p = .031, diff 95% CI [+.013, +.187]. Evidence-channel
effect (text-only graph→multimodal, same 44 TN q): 16 up vs 3 down, p = .004,
CI [+.114, +.455] (vs simple agent: p = .007, CI [+.114, +.477]). Evidence
stack within-model (claude prior-only→best config, 150 q): 37 up vs 12 down,
p = .0005, CI [+.080, +.253]. Reported in §5.4.1–§5.4.3; limitation wording
updated in §6.2.

## Prior-only baseline (supervisor revision, 2026-08-05)

```bash
python scripts/eval_qa.py --agent prior --provider claude --limit 150 \
    --workers 4 --json artifacts/qa_prior_claude_150.json
```

No tools, no evidence — question + 5 options only (`PRIOR_TEMPLATE`).
claude-opus-4-8: **.560** overall (CW .639 / TN .455 / TC .591 / CH .476,
0 errors) → Table 5.4 row 3, §5.4.3, §6.3. Within-model attribution for the
full stack: .560 → .727 (+.167). DeepSeek arm pending a valid
DEEPSEEK_API_KEY (stored key was rejected with 401 on 2026-08-05).

## Demo assets

- 4-video smoke-test corpus (`configs/demo.yaml`, `data/demo/videos/`, git-ignored):
  - bunny/jellyfish/sintel: `https://test-videos.co.uk/vids/{bigbuckbunny,jellyfish,sintel}/mp4/h264/360/*_360_10s_1MB.mp4`
  - nasa_snowflake: `https://images-assets.nasa.gov/video/GSFC_20180329_3DModel_m12908_Snoflake_Melt/GSFC_20180329_3DModel_m12908_Snoflake_Melt~small.mp4`
- Static demo snapshot: [`docs/demo/visualrag_demo.html`](demo/visualrag_demo.html)
  (regenerate: `run_demo_queries.py` then `make_demo_page.py`); hosted copy:
  https://claude.ai/code/artifact/2186f18f-72f1-492c-8777-c1e7dcd2165c
- Interactive demo: `streamlit run ui/app.py`
