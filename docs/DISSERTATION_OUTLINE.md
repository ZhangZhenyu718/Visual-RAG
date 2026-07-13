# Dissertation Outline — Visual RAG: An Autonomous Visual Search Agent for Video Retrieval

MSc dissertation, University of Bristol. Target ~12–15k words / 45–60 pages.
Every results claim below already exists in `results/*.json` + `docs/EXPERIMENTS.md`;
"Evidence" lines say exactly where each number comes from.

---

## Abstract (~300 words)
Problem → approach (multi-modal RAG over video with temporal grounding, agentic
answering) → headline results: corpus R@1 nearly doubled by SigLIP+decomposition
(.026→.048); two-stage retrieval recovers large-model quality; QA accuracy .447→.547
via LangGraph agent; visual evidence nearly doubles temporal-QA accuracy (.341→.636).

## 1. Introduction (~1,200 words)
- 1.1 Motivation: video is unsearchable at the moment level; keyword search fails on
  causal/temporal intent. Industrial relevance (video intelligence platforms).
- 1.2 Research questions (verbatim from the project plan):
  - **RQ1** retrieval effectiveness vs text-only baselines
  - **RQ2** optimal multi-modal embedding strategy (visual + audio + text)
  - **RQ3** agentic query decomposition and answer synthesis
- 1.3 Contributions (numbered; each maps to a chapter section):
  1. An end-to-end open pipeline (ingest → segment → dual-modality index → retrieve
     → rerank → agent) with temporal-precise grounding on NExT-GQA.
  2. Systematic modality/backbone ablations, incl. two documented **negative results**
     (naive late fusion; text cross-encoder re-ranking) with causal analysis.
  3. Two-stage retrieval equivalence finding (cheap index + expensive rerank ≈
     expensive index).
  4. A LangGraph ReAct+reflection agent with a temporal tool, and a controlled
     experiment isolating the value of visual evidence in answer generation.
  5. Reproducible artifacts: code, eval harness, result JSONs, Streamlit demo.
- 1.4 Dissertation structure.

## 2. Background and Related Work (~2,000 words)
- 2.1 Contrastive vision-language pretraining: CLIP; SigLIP's sigmoid loss and why
  it favours retrieval (grounds the W8 result).
- 2.2 Video moment retrieval & temporal grounding; NExT-QA / NExT-GQA benchmarks
  (why chosen: temporal labels → tIoU metrics).
- 2.3 Retrieval-augmented generation; multi-query retrieval, RRF, re-ranking
  (bi- vs cross-encoder).
- 2.4 Agentic frameworks: ReAct, self-reflection, LangGraph; tool-use for
  multimodal QA.
- 2.5 Gap: agentic RAG systems evaluated with temporal precision on video are rare;
  modality contributions at *both* retrieval and answering stages under-studied.

## 3. System Design (~2,500 words)
- 3.1 Architecture overview (figure: pipeline diagram). Design principle: heavy
  offline compute → light online queries (consumer-GPU constraint as a feature:
  reproducibility).
- 3.2 Ingestion & temporal segmentation: scene-detect + uniform keyframes; Whisper
  ASR; 8s/4s overlapping windows as the atomic retrieval unit (justify: window
  overlap vs boundary effects — cite the duplicate-hit observation in W4).
- 3.3 Dual-modality indexing: separate visual/text collections (enables clean
  ablations); L2-normalised CLIP space; segment metadata schema.
- 3.4 Retrieval: per-modality search; late fusion (α-weighted); **W5** LLM query
  decomposition (question → CLIP-friendly scene captions) + RRF; **W6** second-stage
  re-ranking — design iterations: text CE (failed, §5.3), visual reranker with
  rank-imputation for missing-modality candidates (methodological lesson).
- 3.5 Agentic answering (**RQ3**): tools (search, get_segments_around); W4 single-loop
  vs W7 LangGraph state machine (figure: graph diagram with agent/tools/reflect
  nodes); bounded self-reflection; provider abstraction (text-only vs multimodal
  evidence channels).
- 3.6 Demo UI (screenshot; timestamp-jump playback).

## 4. Implementation (~1,200 words)
Stack table; engineering contributions worth examiner attention: idempotent/resumable
batch pipeline; GPU ASR via CTranslate2 on 6GB VRAM; embedding caches (npz) enabling
backbone swaps; concurrency + watchdog patterns for long jobs; cost engineering
(decomposition cache: 3358 questions ≈ $0.15, cached forever). Honest DeepSeek/Claude
API quirks (DSML markup, forced-answer handling).

## 5. Evaluation (~3,500 words — the core chapter)
- 5.1 Setup: NExT-QA/GQA val (3358 grounded questions, 567 videos, 5725 segments);
  metrics (R@K, MRR, nDCG@10, tIoU@1; 5-choice QA accuracy); two scopes (corpus =
  RQ1 setting, video = pure localization); τ=0.5 primary + τ=0.3 sensitivity.
- 5.2 **RQ1 — retrieval vs text-only baseline**: W3 table (visual .111 R@10 vs text
  .011 corpus); the multilingual-transcript explanation; per-question-type breakdown.
  Evidence: `results/eval_val_{corpus,video}.json`.
- 5.3 **RQ2 — embedding strategy** (the ablation chapter):
  - Modality: visual ≫ text; **negative result 1**: α-swept late fusion never beats
    visual-only (table, α ∈ {.5,.7,.8,.9}) → fusion needs a stronger text channel,
    not better weights. Evidence: `eval_val_corpus_a0.*.json`.
  - Query decomposition (W5): deep-rank recall gains (+10% rel R@10 corpus), R@1 flat.
  - Re-ranking (W6): **negative result 2**: text cross-encoder hurts at every weight
    (chatter transcripts uncorrelated with visual events) + the rank-imputation
    lesson; visual ViT-L reranker composes additively with decomposition (+15–17%
    rel overall).
  - Backbone (W8): SigLIP ≫ ViT-L > ViT-B table; **two-stage equivalence** finding;
    final best system (SigLIP + decomposition) corpus R@1 .048 vs .026 baseline.
    Evidence: `eval_val_*_{vitl,siglip,siglip_decomp}.json`.
  - τ sensitivity: best config video R@5 .677 / R@10 .780 at τ=0.3.
- 5.4 **RQ3 — agentic QA**: 150-question 5-choice accuracy: simple .447 → graph .547
  (by-type table: CW +13, TC +18, CH +19); reflection example (caught fabricated
  citation). **Controlled vision experiment**: same 44 TN questions, text-only .341
  vs multimodal .636 (+87%) — evidence channel, not mechanism, was the bottleneck.
  Qualitative case study: the white-dog example (DeepSeek honest abstention vs
  Claude exact-match answer; figures with keyframes).
  Evidence: `qa_{simple,graph}_150.json`, `qa_claude_tn44.json`.
- 5.5 System metrics: index sizes, query latency (ViT-B vs SigLIP encode), ingest
  throughput (567 videos ≈ 2h on RTX 4050), API costs.

## 6. Discussion (~1,200 words)
- 6.1 Synthesis: recall problems (decomposition) vs precision problems (re-ranking)
  vs evidence problems (multimodal answering) are separable and compose.
- 6.2 Limitations: single benchmark; multilingual ASR vs English encoders confound
  (the text-modality result may not generalise to lecture/meeting video); tIoU vs 8s
  windows; graph agent DeepSeek-only; MC-QA as a proxy for open-ended answer quality;
  no user study yet (scoped out; note plan deviation honestly).
- 6.3 Threats to validity: NExT-QA answer priors (LLM may guess without evidence —
  quantify via the .341 floor vs .20 random); decomposition cache determinism.

## 7. Conclusion and Future Work (~800 words)
Answers to RQ1/2/3 in one paragraph each. Future: translate-then-embed text channel;
SigLIP-based reranker over a cheaper index at scale; multimodal graph agent; open-source
LLM comparison (plumbing ready — `agent.provider: local`); user study; streaming index.

## Appendices
A. Reproducibility: EXPERIMENTS.md command→number provenance; config files.
B. Full per-type result tables.
C. Demo: static snapshot (docs/demo) + Streamlit screenshots.
D. Prompt texts (system prompt, decomposition prompt, reflection prompt).

## Figure / table shortlist
1. Pipeline architecture (draw)
2. LangGraph state machine (draw)
3. RQ1 baseline table (have)
4. α-sweep line chart (from JSONs)
5. Ablation ladder bar chart: baseline→W5→W6→W8 (from JSONs)
6. Backbone table incl. two-stage equivalence (have)
7. QA accuracy by question type, grouped bars ×3 configs (have)
8. White-dog qualitative figure: keyframes + two answers (have images)
9. τ=0.3/0.5 sensitivity table (have)
10. Demo screenshot (have)
