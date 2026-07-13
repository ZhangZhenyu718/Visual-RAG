# Chapter 1 — Introduction

## 1.1 Motivation

Video has become the dominant medium in which everyday life, education, and work
are recorded, yet it remains the least searchable. Text search operates at the
granularity of the answer: a query returns the sentence or passage that contains
it. Video search, in practice, still operates at the granularity of the
*container*: a title, a description, at best a manually chaptered timeline. The
question a user actually has — *why did the boy carry the present to the sofa?
what did the dog do after it reached the cushion?* — is a question about a
particular few seconds of footage, phrased in causal and temporal language that
neither keyword matching over metadata nor speech-transcript search can resolve.
The gap is felt anywhere video accumulates faster than it can be watched:
personal libraries, lecture archives, meeting recordings, broadcast footage, and
the industrial video-intelligence platforms that motivated this project's
supervision.

Two research threads have recently made this gap addressable. Contrastive
vision-language models (CLIP and its successors) embed images and natural
language into a shared space, making *what is on screen* directly searchable by
description. And retrieval-augmented generation (RAG) shows how a large language
model can be grounded in retrieved evidence to answer questions it could not
answer alone — with the retrieval unit, in the video case, naturally being a
*time-stamped segment* rather than a document. Between these threads sits an
under-explored composition: an *agentic* system that decomposes a causal or
temporal question, searches a multi-modal index of video segments, reasons over
what it retrieves — including reasoning *along the timeline* — and synthesises an
answer whose every claim cites the seconds of video that support it.

This dissertation designs, builds, and evaluates such a system end-to-end, under
a deliberately modest compute budget (a single consumer laptop GPU plus
commodity LLM APIs), and subjects each of its design decisions to quantitative
test on a benchmark with ground-truth temporal annotations.

## 1.2 Research questions

The project plan fixed three research questions, which this dissertation answers
in Chapter 5:

- **RQ1** — How effectively can a Visual RAG pipeline retrieve relevant video
  segments from natural-language queries compared to text-only retrieval
  baselines?
- **RQ2** — What is the optimal embedding strategy for multi-modal video
  representation (visual + audio + text)?
- **RQ3** — How can an agentic framework autonomously decompose complex queries
  into sub-retrieval tasks and synthesise coherent answers?

Temporal precision is treated throughout as a first-class requirement rather
than an afterthought: retrieval is scored not on finding the right *video* but
on finding the right *moment* (temporal intersection-over-union against
ground-truth intervals), and generated answers must carry timestamp citations.

## 1.3 Contributions

The dissertation makes five contributions, each developed in the chapter noted
and empirically grounded in Chapter 5:

1. **An end-to-end, openly reproducible video RAG pipeline with temporal
   grounding** (Chapter 3): ingestion (keyframes, Whisper speech recognition),
   overlapping temporal segmentation, dual-modality indexing in a joint
   vision-language space, staged retrieval, and an answering agent with
   per-claim timestamp citations — evaluated on 3,358 temporally annotated
   questions over 567 videos (NExT-QA/NExT-GQA), and runnable end-to-end on one
   consumer GPU plus ≈US$5 of API usage.

2. **A systematic ablation of the multi-modal embedding strategy** (§5.3),
   including two documented *negative results* with causal analyses: weighted
   late fusion of visual and transcript embeddings never outperforms
   visual-only retrieval at any weight on this domain, and a text cross-encoder
   re-ranker strictly degrades ranking quality. Both trace to the same cause —
   sparse, multilingual, conversational speech is a poor retrieval key for
   everyday video — and both redirect effort toward the visual channel.

3. **A two-stage retrieval equivalence finding** (§5.3.4): re-ranking a cheap
   index's top-30 with a stronger visual backbone matches indexing the entire
   corpus with that backbone to within measurement noise, while exchanging the
   index for SigLIP nearly doubles top-1 moment retrieval — together giving a
   concrete, measured recipe for scaling quality and cost independently.

4. **An agentic answering layer with temporal tools and bounded
   self-reflection** (Chapter 3, §5.4): a LangGraph state machine that anchors
   events by search, walks the timeline around them, and audits its own draft
   citations against gathered evidence, improving five-way QA accuracy from
   .447 to .547 over a single-loop agent — and, via a controlled
   evidence-channel experiment holding all else fixed, a demonstration that
   supplying keyframes to the answering model nearly doubles accuracy on
   *what-happened-next* questions (.341 → .636). Modality, the experiment
   shows, matters independently at both ends of the pipeline.

5. **Reproducible research artefacts**: the complete codebase, evaluation
   harness, per-experiment result files, provenance log mapping every reported
   number to the command that produced it, generated figures, and an
   interactive demonstration interface whose controls map one-to-one onto the
   ablation dimensions.

## 1.4 Dissertation structure

Chapter 2 reviews the background: contrastive vision-language pretraining,
video moment retrieval and its benchmarks, retrieval-augmented generation, and
agentic LLM frameworks. Chapter 3 presents the system design with the rationale
for each decision. Chapter 4 records the implementation: the concrete stack and
the engineering required to run the full pipeline within the stated budget.
Chapter 5 — the core of the dissertation — evaluates the system against the
three research questions. Chapter 6 discusses what the results mean, their
limitations, and threats to validity. Chapter 7 concludes and outlines future
work.
