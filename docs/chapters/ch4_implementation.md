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

## 4.2 Code organisation and the segment contract

The implementation is deliberately small: 2,162 lines across 19 modules in
`visualrag/`, 1,019 lines of command-line entry points in `scripts/` (eleven of
them), and a 234-line Streamlit application. The package's seven sub-packages
mirror the pipeline stages one-to-one — `ingest`, `embed`, `index`, `retrieve`,
`agent`, `eval`, `data` — so the chapter structure of this dissertation and the
directory structure of the repository are the same structure. Entry points hold
no logic: each parses arguments, loads a configuration, and calls one library
function, which is what allows the harness, the agent, and the UI to exercise
identical code paths (§3.4).

Where the line counts concentrate is informative. `agent/answerer.py` (398
lines) and `agent/graph_agent.py` (268) dominate, followed by `retrieve/rerank.py`
(171) and `retrieve/decompose.py` (152); every remaining module is under 130
lines. The asymmetry is the design working as intended — provider abstraction and
control flow are genuinely irreducible complexity, while the pipeline stages stay
thin because a single data contract does the coordinating between them.

**The segment contract.** That contract is the `Segment` dataclass in
`schema.py`: an identifier, a video identifier, `[start, end]` seconds, keyframe
paths, transcript text, and optional OCR text. Three properties of it carry more
weight than its size suggests. First, the identifier is derived rather than
allocated — `f"{video_id}::{start:.2f}-{end:.2f}"` — so it is reproducible from
the timestamps alone; this is what lets per-backbone embedding caches, two
ChromaDB collections, and the result JSONs be joined after the fact without a
database or a migration step. Second, a `text` property concatenates whichever
textual modalities are present, so retrieval code never branches on whether OCR
is enabled; toggling that modality is a configuration change, not a code change.
Third, segments persist as JSONL, which keeps every intermediate stage readable
with ordinary tools and lets any stage be re-run from disk alone.

**Configuration.** Six YAML files describe the experimental conditions
(`default`, `demo`, `ablation_vitl`, `ablation_siglip`, `agent_best`,
`youcook2`), with command-line flags overriding individual fields. The point of
this is provenance rather than convenience: a Chapter 5 run is identified by a
file that is committed alongside its results, not by a line of shell history.

## 4.3 Fitting the pipeline into 6 GB

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

## 4.4 Cost engineering for LLM-dependent components

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
≈US$3 and the 150-question multimodal best-configuration run ≈US$25 (Claude,
where keyframe image tokens dominate). Rate limits are respected by capping
concurrency rather than by retry storms.

## 4.5 Provider-agnostic answering and its pitfalls

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

## 4.6 Evaluation harness and provenance

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

## 4.7 Correctness practices, and the absence of a test suite

The project has **no automated test suite**. This is stated plainly because the
alternative practices it relied on are load-bearing for the credibility of
Chapter 5, and because their gaps are real.

Four properties substituted for tests. *Determinism*: given the committed
decomposition cache, a retrieval run is bit-identical across invocations, so
re-running an unchanged configuration is itself a regression check and any metric
movement is attributable to the change made. *Self-describing outputs*: each run
emits a JSON recording its own configuration alongside its metrics, so a result
cannot be silently mismatched with the settings that produced it. *Generated
figures*: every data figure is produced from those committed JSONs by a script
(§4.6), which makes it impossible for a figure and its prose to drift apart.
*Idempotence*: a stage that finds its own output and skips it (§4.3) is an
implicit consistency check on that output's existence and shape.

In practice these caught defects at the level of implausible *numbers* rather
than failing assertions. The re-ranker's missing-modality bug (§3.4, §5.3.3)
announced itself as an R@1 of exactly zero — a value no correct implementation
could produce — and the ASR device-detection bug appeared as a silent fallback to
CPU-speed transcription. Both were found because a monitored quantity moved, not
because a test failed.

The limitation is the converse: this regime cannot detect a logic error in any
path that no reported number exercises. The OCR ingestion path, disabled
throughout (§3.2), is the clearest example — it is implemented, never measured,
and therefore unverified. Unit tests over the segmenter's boundary arithmetic and
over the rank-fusion and imputation logic would be the highest-value addition to
the codebase, and are recommended in §7.2 accordingly.

## 4.8 Summary

The implementation's through-line is *reuse*: one decode pass per video, one
embedding pass per backbone, one LLM call per unique question, one code path
shared by harness, agent, and UI. That discipline — more than any individual
optimisation — is what let a dissertation-scale evaluation programme (twenty-odd
full-split retrieval runs, three agent studies, three backbones) execute on one
laptop within a single project week.
