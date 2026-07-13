# Chapter 4 — Implementation

Chapter 3 described *what* was built and why; this chapter records *how*, with
emphasis on the engineering that made a full benchmark evaluation feasible on a
single consumer laptop (RTX 4050, 6 GB VRAM) and a few dollars of API usage. The
codebase is a small Python package (`visualrag/`) with thin CLI entry points
(`scripts/`), one YAML configuration per experimental condition, and a Streamlit
interface (`ui/`).

## 4.1 Technology stack

| Concern | Choice | Notes |
|---|---|---|
| Video decoding / duration | PyAV | single decode pass per video |
| Shot detection | PySceneDetect (adaptive) | + 0.5 fps uniform fallback |
| Speech recognition | faster-whisper (large-v3, int8) | CTranslate2 runtime |
| Embeddings | open_clip: ViT-B-32, ViT-L-14, SigLIP SO400M | one interface, three backbones |
| Vector store | ChromaDB (embedded, HNSW, cosine) | two collections per backbone |
| Re-ranker (ablation) | sentence-transformers bge-reranker-v2-m3 | text cross-encoder, §5.3.3 |
| Agent framework | LangGraph | W7 state machine |
| LLM providers | DeepSeek, Anthropic Claude, any OpenAI-compatible endpoint | provider-agnostic answerer |
| UI | Streamlit | search + timestamp-jump playback |
| Evaluation | custom harness (`visualrag/eval`) | all metrics of §5.1 |

Two stack decisions deserve one sentence each. ChromaDB was chosen over
server-class vector databases because at 10³–10⁴ vectors, HNSW is effectively
exact and zero operational overhead beats theoretical scalability. LangGraph was
chosen over a heavier agent framework because the W7 design needed exactly one
thing — an explicit, bounded, inspectable state machine — and nothing else.

## 4.2 Fitting the pipeline into 6 GB

The plan's premise (§3.1) was that heavy computation happens offline; the
implementation's contribution was making "offline" affordable *locally* rather
than requiring a cloud GPU. Three techniques mattered.

**Decoupling ASR from the deep-learning runtime.** Whisper large-v3 nominally
exceeds a 6 GB budget under PyTorch, but faster-whisper executes it through
CTranslate2 with int8 weights in ≈3 GB. One integration subtlety cost a bug
fix: the pipeline's device auto-detection originally probed *PyTorch's* CUDA
availability, which is false under a CPU-only PyTorch build even when the GPU
is usable — so the transcriber now probes CTranslate2's own CUDA device count,
and ships the pip-installed cuBLAS/cuDNN libraries to the DLL search path on
Windows. With this in place the full 567-video corpus transcribed in ≈2 h
locally.

**Embedding caches as the unit of reuse.** Every backbone's segment embeddings
persist as per-video compressed arrays (`.npz`) keyed by segment ID. Indexing
reads only the cache; evaluation reads only the index. Re-running any experiment
therefore costs seconds; adding a backbone costs one embedding pass; and the
visual re-ranker (§3.4) is *free at query time* beyond a matrix product, because
its ViT-L vectors are the same cache the ViT-L index ablation used. A related
robustness detail: SigLIP's open_clip implementation exposes no
`visual.output_dim`, so the encoder wrapper resolves the embedding width by a
three-step probe (attribute → model field → dummy forward pass) — the kind of
small fix on which "one interface, three backbones" actually depends.

**Idempotence everywhere.** Every batch stage — ingestion, embedding, indexing,
decomposition — checks for its own output and skips work already done, at
per-video or per-question granularity. Long jobs on a laptop *will* be
interrupted (sleep, OOM, session resets); with idempotent stages, every restart
resumes where it stopped. During the project this was exercised repeatedly: the
full-corpus ingestion survived two process kills and completed under a
self-healing watchdog script that relaunches the batch until its completion
marker appears (`scripts/ingest_watchdog.ps1`).

## 4.3 Cost engineering for LLM-dependent components

Two pipeline components call LLM APIs; both were engineered so that *evaluation
scale* never multiplies *API cost*.

**Query decomposition** results are cached on disk keyed by the question string.
The full validation split (3,358 questions, 16 concurrent workers) cost
≈US$0.15 *once*; every subsequent experiment — the W5/W6/W8 ablations each rerun
decomposition-dependent retrieval several times — replayed the cache for free.
The cache is committed to the repository, making the retrieval experiments
deterministic and reproducible without any API key.

**Agent evaluation** batches questions across a thread pool (the retrieval
stack is read-only and thread-safe), parses the model's final option choice
from a fixed sentinel line, and records per-question token usage. The 150-question
agent comparison cost ≈US$0.35 (DeepSeek); the 44-question multimodal run
≈US$3 (Claude). Rate limits are respected by capping concurrency rather than
by retry storms.

## 4.4 Provider-agnostic answering and its pitfalls

The answerer runs unchanged against Anthropic's API, DeepSeek, or any
OpenAI-compatible endpoint (including a local Ollama server — the hook for the
open-source-model comparison left as future work). The abstraction is a thin
dispatch: one message-format adapter per provider family, one shared tool
schema, one shared evidence renderer that adds keyframe image blocks when the
provider supports vision. Field experience contributed three fixes worth
recording, since each silently produces *empty or corrupt answers* rather than
errors. (i) Models sometimes spend the entire round budget requesting more
searches; the loop must end with a forced-answer turn with tool use disabled.
(ii) Under a forced-answer turn, DeepSeek occasionally emits its intended tool
calls as literal markup in the message text; the answer extractor strips these.
(iii) When self-reflection can send the agent back for more evidence, the
revision path must check the remaining round budget, or the reflect/act cycle
consumes it and the graph terminates answerless.

## 4.5 Evaluation harness and provenance

The harness (`scripts/evaluate.py`, `scripts/eval_qa.py`) exposes every
experimental axis of Chapter 5 as a flag (`--scope`, `--modalities`, `--tau`,
`--decompose`, `--rerank`, `--agent`, `--provider`) over per-condition YAML
configurations, and emits a self-describing JSON per run. All result files are
committed under `results/`; `docs/EXPERIMENTS.md` maps every number reported in
this dissertation to the exact command, configuration, and date that produced
it; and the dissertation's data figures are generated from those JSONs by a
committed script (`docs/figures/make_figures.py`), so no reported value exists
only in prose. Figures 3.1–3.2 (architecture) are maintained as editable
draw.io sources alongside their exports.

## 4.6 Summary

The implementation's through-line is *reuse*: one decode pass per video, one
embedding pass per backbone, one LLM call per unique question, one code path
shared by harness, agent, and UI. That discipline — more than any individual
optimisation — is what let a dissertation-scale evaluation programme (twenty-odd
full-split retrieval runs, three agent studies, three backbones) execute on one
laptop within a single project week.
