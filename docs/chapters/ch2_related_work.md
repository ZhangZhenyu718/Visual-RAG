# Chapter 2 — Background and Related Work

This chapter reviews the four research threads the system composes — contrastive
vision-language pretraining (§2.1), video moment retrieval and its benchmarks
(§2.2), retrieval-augmented generation (§2.3), and agentic LLM frameworks
(§2.4) — then surveys the emerging work that, like this project, combines them
into LLM-driven video understanding systems (§2.5), and states the gap this
dissertation addresses (§2.6). Citation keys refer to `references.bib`.

## 2.1 Contrastive vision-language pretraining

CLIP (Radford et al., 2021) established that a dual-encoder trained with a
contrastive objective on web-scale image–text pairs yields a joint embedding
space in which arbitrary natural-language descriptions can be matched against
images zero-shot. Two properties of this family underpin the present system.
First, the *dual-encoder* architecture separates image and text encoding, so a
corpus can be embedded once offline and queried later with only the (cheap)
text tower — the architectural premise of Chapter 3's offline/online split.
Second, because training pairs are alt-text captions, the encoders are
strongest at matching *literal scene descriptions*; abstract, causal, or
temporal language falls outside the training distribution, a mismatch this
project addresses at query time (§3.4) rather than by fine-tuning.

The open-source replication effort around open_clip demonstrated reproducible
scaling laws for this family and released the LAION-trained checkpoints used
here (Schuhmann et al., 2022; Cherti et al., 2023). SigLIP (Zhai et al., 2023)
replaced CLIP's softmax-based InfoNCE objective with a pairwise sigmoid loss,
removing the global normalisation over the batch and yielding markedly stronger
retrieval performance at comparable scale. The backbone ablation of §5.3.4 — in
which SigLIP SO400M nearly doubles top-1 moment retrieval over CLIP ViT-B —
provides an independent, temporally grounded replication of that advantage on
everyday video.

## 2.2 Video moment retrieval, video QA, and temporal grounding

Text-to-video retrieval initially operated at whole-video granularity, with
benchmarks such as MSR-VTT (Xu et al., 2016); CLIP4Clip (Luo et al., 2022)
showed that frame-level CLIP features, pooled, transfer to this task with
minimal video-specific machinery — the same frames-as-proxy assumption this
system makes at segment level. *Moment* retrieval sharpened the task to
localising an interval inside a video: Gao et al. (2017) introduced the
sliding-window formulation and Charades-STA, dense event captions came with
ActivityNet Captions (Krishna et al., 2017), and Moment-DETR (Lei et al., 2021)
reframed localisation as end-to-end set prediction. The present system inherits
the sliding-window formulation — its 8-second overlapping segments are windows
scored by embedding similarity — but performs *corpus-level* moment retrieval
(find the moment across 567 videos), which moment-retrieval work typically
assumes solved by an upstream video search.

On the question-answering side, NExT-QA (Xiao et al., 2021) moved video QA
beyond descriptive questions to causal and temporal ones — *why* and
*what-happened-next* — over everyday videos. Its extension NExT-GQA (Xiao et
al., 2024) added 10.5K temporal grounding labels and reported a sobering
finding: state-of-the-art video-language models answer well while attending to
the *wrong moments*, i.e. their answers are weakly grounded. This finding
motivates both the benchmark choice and the evaluation design of this
dissertation: NExT-GQA's interval labels are what allow retrieval to be scored
at moment granularity (tIoU) rather than video granularity, and answer
grounding — every claim citing its seconds of video — is treated as a
first-class output requirement rather than a diagnostic. SeViLA (Yu et al.,
2023) addressed grounding by chaining a localiser and an answerer within one
trained model; the present system reaches a similar decomposition —
localise-then-answer — with zero-shot components orchestrated by an agent
instead of end-to-end training.

## 2.3 Retrieval-augmented generation

RAG couples a parametric language model with a non-parametric retrieval index
(Lewis et al., 2020; Guu et al., 2020), classically over dense passage
embeddings (Karpukhin et al., 2020). Three RAG sub-problems recur in this
dissertation in video form.

**Query–document mismatch.** When queries and indexed content live in different
registers, rewriting the query helps: HyDE (Gao et al., 2023) generates a
hypothetical document and searches with its embedding; least-to-most prompting
(Zhou et al., 2023) decomposes complex questions into simpler sub-problems. The
W5 decomposer (§3.4) transfers this idea across modalities: the LLM rewrites a
causal question into the *scene-caption register that a contrastive visual
encoder can match*, which is HyDE's manoeuvre with "document" replaced by
"caption". Multiple rewrites are combined with reciprocal rank fusion (Cormack
et al., 2009), whose rank-based, score-free formulation is what makes fusing
differently phrased sub-queries safe.

**Precision at the top of the ranking.** Two-stage retrieve-then-re-rank
pipelines with a cross-encoder second stage are standard in text retrieval
(Nogueira and Cho, 2019), with the BGE family (Chen et al., 2024) among the
strongest open re-rankers. Chapter 5 shows the transfer to video is *not*
automatic: the text cross-encoder degrades ranking on everyday video because
transcripts are weak relevance evidence (§5.3.3), and the effective second
stage is instead a stronger *visual* encoder — same architecture-level pattern,
different modality.

**Self-critique.** Self-RAG (Asai et al., 2024) trains a model to critique its
own retrieval and generation with reflection tokens. The graph agent's reflect
node (§3.5) implements the same principle — audit the draft against the
retrieved evidence before answering — as a zero-shot prompted pass with an
explicit, bounded control-flow position, rather than a trained capability.

## 2.4 Agentic LLM frameworks

Chain-of-thought prompting (Wei et al., 2022) established that intermediate
reasoning improves LLM task performance; ReAct (Yao et al., 2023) interleaved
that reasoning with *actions* against external tools, defining the
observe–think–act loop that underlies most current agents, including both
agents in this project. Toolformer (Schick et al., 2023) demonstrated that tool
invocation can be learned self-supervised, foreshadowing the native
function-calling interfaces that commercial LLM APIs now expose and on which
the W4 agent is built directly. Reflexion (Shinn et al., 2023) added verbal
self-feedback across episodes, improving agents without weight updates — the
within-episode analogue of which is the reflect node above. This project
implements its multi-step agent on LangGraph (a state-machine orchestration
library from the LangChain project), chosen because it makes the control flow —
nodes, bounded loops, termination conditions — explicit and inspectable
(§3.5), in contrast to prompt-encoded loops whose behaviour is implicit in
model outputs.

## 2.5 LLM-driven video understanding and video RAG

A rapidly growing line of work applies LLM orchestration to video. LLoVi (Zhang
et al., 2024) shows a deliberately simple recipe — densely caption short clips,
then let an LLM aggregate the captions — is strong on long-video QA including
NExT-QA. Two contemporaneous systems named VideoAgent make the LLM an active
controller: Wang et al. (2024) iteratively *search* for informative frames,
answering NExT-QA with ~8 frames on average, while Fan et al. (2024) equip the
LLM with a structured memory over events and object tracks, queried through
tools. VideoRAG (Ren et al., 2025) scales the RAG framing to extremely long
multi-video corpora with a graph-organised index. The present system sits in
this family but differs in emphasis in three ways. First, *temporal grounding
is the deliverable*: retrieval is evaluated by tIoU against NExT-GQA intervals
and answers must cite timestamps, whereas the systems above report QA accuracy
alone. Second, it operates at *corpus* scope — the question does not name its
video, and finding it is part of the task. Third, its contribution is
*measurement*: a component-wise ablation programme (modality, fusion,
decomposition, re-ranking, backbone, evidence channel) with documented negative
results, rather than a new model or a leaderboard entry.

## 2.6 Summary of the gap

Each ingredient of this project is established: contrastive encoders for
zero-shot visual matching, sliding-window moment retrieval, RAG's
retrieve-rewrite-rerank toolkit, ReAct-style tool agents, and LLM video
orchestration. What the literature lacks — and what NExT-GQA's grounding
finding (Xiao et al., 2024) shows is missing in practice — is an account of how
these ingredients compose into a *temporally grounded*, corpus-scope video RAG
system, with the contribution of each component measured under one protocol,
including where standard components fail to transfer. Providing that account,
on reproducible consumer-scale infrastructure, is the role of Chapters 3–6.
