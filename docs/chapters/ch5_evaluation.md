# Chapter 5 — Evaluation

This chapter evaluates the system against the three research questions. Section 5.1
describes the experimental setup shared by all experiments. Section 5.2 addresses
RQ1 by comparing multi-modal retrieval against a text-only baseline. Section 5.3
addresses RQ2 through a series of ablations over the embedding and retrieval
strategy — modality fusion, query decomposition, second-stage re-ranking, and the
embedding backbone — including two negative results that shaped the final design.
Section 5.4 addresses RQ3 by measuring end-to-end question-answering accuracy of
the agentic pipeline, culminating in a controlled experiment that isolates the
contribution of visual evidence at the answering stage. Section 5.5 reports
system-level costs. Every number in this chapter is reproducible from the committed
result files (`results/*.json`); the exact command that produced each result is
recorded in `docs/EXPERIMENTS.md`.

## 5.1 Experimental setup

**Dataset.** All experiments use the validation split of NExT-QA (Xiao et al.,
2021), a benchmark of causal and temporal multiple-choice questions over short
(~44 s) everyday videos, together with the temporal grounding labels contributed
by NExT-GQA (Xiao et al., 2024).
The split contains 3,358 questions over 567 videos, and — after intersecting with
the grounding annotations — every question carries at least one ground-truth
temporal interval locating the evidence for its answer. NExT-GQA was chosen over
larger retrieval benchmarks (e.g. MSRVTT, ActivityNet Captions) precisely because
these interval labels allow retrieval to be scored with *temporal* precision rather
than at whole-video granularity.

**Indexing.** The ingestion pipeline of Chapter 3 produced 5,725 overlapping
segments (8 s window, 4 s stride) from the 567 videos. Each segment is indexed in
two ChromaDB collections (HNSW, cosine): a visual collection holding the pooled
CLIP embedding of the segment's keyframes (5,725 vectors), and a text collection
holding the CLIP text embedding of its speech transcript (3,433 vectors — 60% of
segments contain speech). Unless stated otherwise the index backbone is CLIP
ViT-B-32 (laion2b); Section 5.3.4 varies the backbone.

**Retrieval metrics.** For each question, the question text is used as the query
and the top-K segments are retrieved. A retrieved segment counts as a *hit* if it
comes from the ground-truth video **and** its temporal interval overlaps a
ground-truth interval with tIoU ≥ τ; requiring both conditions ties the metric to
moment-level retrieval rather than video identification. We report Recall@{1,5,10}
and MRR at the primary threshold τ = 0.5, nDCG@10 with graded tIoU gains
(ideal-normalised against the ground-truth video's own segments), and tIoU@1 — the
mean temporal overlap of the top-ranked segment, the dissertation's "temporal
accuracy" measure. Section 5.3.5 repeats key configurations at the looser τ = 0.3.

**Scopes.** Two retrieval scopes are evaluated. In **corpus** scope the query
searches all 5,725 segments across 567 videos — the full RQ1 setting, where the
system must find both the right video and the right moment. In **video** scope
the search is restricted to the ground-truth video's own segments, isolating pure
temporal localisation. The gap between the two scopes attributes error to
cross-video confusion versus within-video imprecision.

**QA metric.** Section 5.4 evaluates answer generation directly on NExT-QA's
five-way multiple-choice format: the agent receives the question and its five
options, gathers evidence with its tools, and must emit a final option index.
Accuracy is reported overall and per question type (CW/CH: causal *why*/*how*;
TN/TC/TP: temporal *next*/*current*/*previous*). The random-guess floor is 0.20.

**Implementation constants.** Unless varied by the ablation, fusion weight
α = 0.5, retrieval depth K = 10, re-ranker candidate pool 30, and the
question encoder matches the index backbone. All retrieval experiments are
deterministic given the committed decomposition cache; agent experiments use
temperature-default API calls and are therefore subject to normal LLM variance.

## 5.2 RQ1 — Multi-modal retrieval against a text-only baseline

Table 5.1 reports the three single-strategy configurations on both scopes at
τ = 0.5. The text-only baseline embeds each segment's Whisper transcript and
searches it with the question text — the standard "search what was said"
approach that RQ1 asks us to beat. A reminder of scale before reading the
table (§1.2): a hit requires the correct video *and* tIoU ≥ 0.5 between an
8-second window and the annotated moment, searched over all 5,725 segments —
a deliberately strict criterion under which absolute scores are small by
construction, so the informative content of the table is in the *relative*
comparisons between rows, not in the magnitudes.

**Table 5.1 — Retrieval by modality (ViT-B-32 index, τ = 0.5).**

| Scope | Modality | R@1 | R@5 | R@10 | MRR | nDCG@10 | tIoU@1 |
|---|---|---|---|---|---|---|---|
| corpus | visual | .026 | .075 | .111 | .047 | .117 | .035 |
| corpus | text | .002 | .005 | .011 | .004 | .011 | .003 |
| corpus | fused (α=.5) | .002 | .010 | .013 | .005 | .018 | .007 |
| video | visual | .148 | .394 | .491 | .250 | .659 | .203 |
| video | text | .080 | .232 | .305 | .143 | .390 | .117 |
| video | fused (α=.5) | .129 | .384 | .491 | .235 | .639 | .186 |

Three observations answer RQ1. First, **visual retrieval dominates text retrieval
by an order of magnitude in corpus scope** (R@10 .111 vs .011): to find a moment
among 567 videos of everyday footage, what is *on screen* is far more
discriminative than what is *said*. Second, the failure of the text channel has an
identifiable cause rather than being an artefact of poor transcription: NExT-QA's
home videos contain sparse, conversational, and frequently non-English speech
(German, Mandarin and other languages appear in the transcripts), while the CLIP
text encoder that embeds both queries and transcripts is trained predominantly on
English alt-text. The information is often simply absent — a question about what a
child *does* is rarely answered by what anyone *says*. Third, the corpus/video gap
localises the difficulty: within the correct video the system already ranks a
correct moment in its top five for 39% of questions, but across the corpus R@10
falls to 11% — **cross-video confusion, not within-video imprecision, is the
dominant error mode**, which motivates the precision-oriented interventions of
Section 5.3.

The by-type breakdown (Appendix B) shows temporal-precise question types (TP,
n = 52) retrieving worst in corpus scope, previewing the temporal-reasoning theme
of Section 5.4.

## 5.3 RQ2 — Embedding and retrieval strategy ablations

### 5.3.1 Negative result: naive late fusion never beats visual-only

The natural response to a weak-but-nonzero text channel is weighted late fusion:
score = α·visual + (1−α)·text. Figure 5.1 (fig4_alpha_sweep) sweeps α over
{0.5, 0.7, 0.8, 0.9} in corpus scope. Every metric rises monotonically toward the
α = 1.0 endpoint — visual-only — and none crosses it; the same holds in video
scope, where the best fused configuration (α = 0.8) ties visual-only on R@10
(.494 vs .491) but loses on R@1 (.139 vs .148) and MRR (.244 vs .250). At α = 0.5
fusion is catastrophic in corpus scope (R@10 .013) because the text scores act as
high-variance noise added to a weak-but-real visual signal.

![Late-fusion weight sweep in corpus scope.](../figures/fig4_alpha_sweep.pdf){#fig:alpha-sweep width=82%}

The conclusion is sharper than "tune α": **with this text channel, there is no
fusion weight at which the transcript embeddings contribute positive evidence.**
Fusion is not a weighting problem but a channel-quality problem — the actionable
implication being to repair the text channel (e.g. translate transcripts to
English before embedding, or use a dedicated multilingual text-retrieval encoder)
before re-introducing it, rather than to search the weight space further. All
subsequent experiments therefore use the visual modality.

### 5.3.2 Query decomposition recalls more, but no more precisely

NExT-QA questions are causal and temporal ("why did the boy pick up one present
from the group of them and move to the sofa"), whereas CLIP matches literal scene
descriptions. The W5 decomposer prompts an LLM to rewrite each question into at
most four short scene captions ("a boy picks up a present from a group of
presents", "a boy carries a present and walks to a sofa", …), retrieves each
caption plus the original question independently, and fuses the rankings with
reciprocal rank fusion. The rewrite is computed once per question and cached
(3,358 questions ≈ US$0.15 of LLM usage).

Decomposition lifts the deep-rank metrics in corpus scope — R@10 .111 → .122
(+9.9% relative), R@5 +6.7%, MRR +6.4% — while leaving R@1 and tIoU@1 unchanged
(Table 5.2, rows 1–2; Figure 5.2). The pattern is informative: translating
abstract questions into CLIP-native captions surfaces *additional correct
candidates* into the top ten, but rank fusion of near-tied cosine scores cannot
push them to position one. Recall problems and precision problems are separable;
decomposition solves a recall problem, and the next section supplies the missing
precision.

### 5.3.3 Second-stage re-ranking: a negative and a positive result

**Text cross-encoder (negative).** The project plan anticipated a standard text
cross-encoder re-ranker (bge-reranker) scoring (question, transcript) pairs over
the candidate pool. On a 100-question development subset this *reduced* quality at
every fusion weight tested — at weight 0.5 it halved R@5 (.070 → .030), and even
at 0.8 it remained below the no-reranking baseline. The mechanism mirrors §5.3.1:
the transcripts are conversational chatter, largely uncorrelated with the visual
events the questions ask about, so every reordering the cross-encoder makes
injects noise. An implementation subtlety compounds the problem and is worth
recording: 40% of segments have no transcript at all, and if such candidates
simply receive no re-ranker score, any scored candidate the cross-encoder likes
outranks *every* unscored one — structurally demoting exactly the visual-only
segments most likely to be correct. (Our first implementation had this flaw and
drove R@1 to zero.) The repaired design imputes the re-ranker term of a
modality-missing candidate from its retrieval rank, keeping both candidate classes
on one score scale; the negative result above is reported *with* this repair, so
it reflects the channel, not the bug.

**Visual re-ranker (positive).** The replacement re-ranker re-scores the top-30
candidates with a stronger vision model (CLIP ViT-L-14) over pre-computed segment
embeddings, fusing the ViT-L ranking with the first-stage ranking by weighted RRF.
Every segment has keyframes, so the missing-modality problem vanishes. Table 5.2
and Figure 5.2 (fig5_ablation_ladder) show the ladder at τ = 0.5, corpus scope:

![Retrieval ablation ladder from the baseline to the final configuration.](../figures/fig5_ablation_ladder.pdf){#fig:ablation-ladder width=86%}

**Table 5.2 — Corpus-scope ablation ladder (visual modality, τ = 0.5).**

| Configuration | R@1 | R@5 | R@10 | MRR | tIoU@1 |
|---|---|---|---|---|---|
| ViT-B-32 baseline | .026 | .075 | .111 | .047 | .035 |
| + decomposition (W5) | .025 | .080 | .122 | .050 | .035 |
| + ViT-L re-ranking (W6) | .029 | .087 | .121 | .053 | .041 |
| + both | .030 | .087 | .128 | .055 | .040 |
| SigLIP index (W8) | .044 | .106 | .143 | .070 | .054 |
| SigLIP + decomposition | **.048** | **.110** | **.161** | **.076** | .055 |

The two mid-pipeline interventions compose additively: decomposition contributes
at depth, re-ranking at the top of the list, and their combination improves every
metric over the baseline (+15–17% relative on MRR and R@10). In video scope the
combined configuration reaches R@1 .154, R@10 .501, tIoU@1 .209 — each the best
of any ViT-B configuration.

### 5.3.4 Backbone: SigLIP dominates, and two-stage retrieval matches one-stage

The final RQ2 axis exchanges the embedding backbone itself, holding segmentation
and evaluation fixed. Three backbones were compared as *the index*: CLIP ViT-B-32
(baseline), CLIP ViT-L-14, and SigLIP SO400M (webli), each with its own question
encoder.

Two findings emerge from Table 5.2 (bottom rows) and Table 5.3. First, **SigLIP is
categorically stronger for this task**: as the index it lifts corpus R@1 by 69%
relative over ViT-B (.026 → .044) and tIoU@1 by 54%, and with decomposition on
top reaches R@1 .048 — **nearly double the original baseline** — and video-scope
tIoU@1 .230. This is consistent with SigLIP's sigmoid-loss pretraining, which has
been reported to favour retrieval tasks (Zhai et al., 2023).

Second, ViT-L used as the full index (corpus R@1 .029, R@5 .090, R@10 .122,
MRR .054) is statistically indistinguishable from ViT-B *plus* ViT-L re-ranking
(.029/.087/.121/.053) — every metric agrees within ±.003. **The two-stage
cheap-index/expensive-re-ranker design recovers essentially all of the larger
model's quality** while embedding the corpus only with the small model; at larger
corpus scales, where re-embedding everything with the expensive model is
prohibitive, the two-stage design is the scalable route to the same accuracy. A
corollary worth stating for practitioners: a re-ranker must *differ* from the
index model — re-scoring with the index's own embeddings is an identity operation.

**Table 5.3 — Video-scope backbone comparison (visual modality, τ = 0.5).**

| Index backbone | R@1 | R@5 | R@10 | MRR | tIoU@1 |
|---|---|---|---|---|---|
| ViT-B-32 | .148 | .394 | .491 | .250 | .203 |
| ViT-L-14 | .151 | .407 | .496 | .257 | .210 |
| SigLIP SO400M | .170 | .406 | .494 | .266 | .221 |
| SigLIP + decomposition | **.176** | **.419** | **.496** | **.275** | **.230** |

### 5.3.5 Threshold sensitivity

τ = 0.5 is a strict criterion for 8-second windows scored against ground-truth
intervals of arbitrary length. Table 5.6 repeats the ViT-B baseline and the best
ViT-B configuration at τ = 0.3; the binary metrics roughly double, while tIoU@1,
which is threshold-free, is unchanged by construction.

**Table 5.6 — Threshold sensitivity (visual modality, ViT-B index).**

| Scope | Configuration | τ | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|---|
| corpus | baseline | 0.5 | .026 | .075 | .111 | .047 |
| corpus | baseline | 0.3 | .054 | .130 | .186 | .089 |
| corpus | + decompose + rerank | 0.5 | .030 | .087 | .128 | .055 |
| corpus | + decompose + rerank | 0.3 | .062 | .151 | .208 | .101 |
| video | baseline | 0.5 | .148 | .394 | .491 | .250 |
| video | baseline | 0.3 | .311 | .660 | .774 | .456 |
| video | + decompose + rerank | 0.5 | .154 | .410 | .501 | .258 |
| video | + decompose + rerank | 0.3 | .323 | .677 | .780 | .469 |

Two conclusions follow: the system's practical localisation ability ("lands
within a loose overlap of the right moment") is considerably better than the
strict headline numbers suggest — within the correct video, a correct moment is
in the top five for two-thirds of questions — and the relative ordering of
configurations is stable across thresholds, so the ablation conclusions above do
not depend on the choice of τ.

## 5.4 RQ3 — Agentic question answering

### 5.4.1 Agent comparison

Table 5.4 and Figure 5.3 (fig7_qa_by_type) compare the two agent designs of
Chapter 3 on the first 150 grounded val questions under a text-only LLM
(deepseek-chat): the single-loop function-calling agent (W4) and the LangGraph
state machine (W7), which adds the temporal tool (`get_segments_around`) and a
bounded self-reflection pass that audits the draft answer's citations against the
gathered evidence before accepting it.

![Question-answering accuracy by question type and agent configuration.](../figures/fig7_qa_by_type.pdf){#fig:qa-by-type width=88%}

**Table 5.4 — Five-choice QA accuracy, 150 val questions.** Rows 1–2 use the
text-only LLM (deepseek-chat); the last two rows use claude-opus-4-8
(§5.4.3–§5.4.4).

| Agent | Evidence | Overall | CW (n=61) | TN (n=44) | TC (n=22) | CH (n=21) |
|---|---|---|---|---|---|---|
| Simple loop (W4) | text-only | .447 | .459 | .341 | .591 | .429 |
| LangGraph (W7) | text-only | .547 | .590 | .341 | .773 | .619 |
| Prior-only, no tools (§5.4.3) | none | .560 | .639 | .455 | .591 | .476 |
| LangGraph, best config (§5.4.4) | multimodal | **.727** | **.852** | **.568** | .682 | **.762** |

Both agents clear the .20 random floor by a wide margin, confirming that
retrieved transcript evidence carries usable signal. The graph agent improves
overall accuracy by 10 points (+22% relative), with the gains concentrated in
causal questions (CW +13.1, CH +19.0 points) and temporal-current questions
(TC +18.2). Because both agents answered the identical 150 questions, the
comparison is paired: 29 questions flipped to correct under the graph agent
against 14 that flipped to incorrect, giving an exact McNemar p = .031, with a
paired-bootstrap 95% CI on the difference of [+.013, +.187] (10,000 resamples)
— the improvement is unlikely to be run-to-run LLM variance. Inspection of transcripts attributes the gains to three behaviours:
systematic anchor-then-walk retrieval on temporal questions, reformulated
follow-up searches when initial evidence is thin, and the reflection pass —
which in observed runs rejected drafts whose citations did not appear in the
gathered evidence, forcing a grounded rewrite.

The striking exception is TN (*what happened next*), frozen at .341 for **both**
agents. Qualitative inspection shows the temporal machinery working exactly as
designed — the agent anchors the cue event and retrieves the segments that follow
it — and then failing at the last step: the answer to a TN question is almost
always a *visual action* (what a person or animal does next), the following
segments are typically silent, and a text-only LLM receives no usable evidence.
In one representative case ("what does the white dog do after going to the
cushion", ground truth *smells the black dog*), the text-only agent responded
that the post-anchor segments contain no speech and the action "cannot be
determined from the available evidence" — a correct description of its own
evidence starvation.

### 5.4.2 Controlled experiment: visual evidence at the answering stage

The TN result yields a precise hypothesis: the bottleneck is the *evidence
channel*, not the temporal mechanism. To test it, the same 44 TN questions were
re-run with a single change: the provider was switched to a multimodal LLM
(claude-opus-4-8), whose tool results embed each segment's keyframe image
alongside the transcript. Retrieval index, tools, prompts, and questions were
held fixed.

**Table 5.5 — Same 44 TN questions, evidence channel varied.**

| Configuration | TN accuracy |
|---|---|
| Text-only LLM, simple agent | .341 |
| Text-only LLM, graph agent (temporal tool) | .341 |
| **Multimodal LLM, keyframes in evidence** | **.636** |

Accuracy rises by 29.5 points (+87% relative, 44/44 questions completed). This
comparison is likewise paired on identical questions: against the graph agent,
16 questions flipped to correct under visual evidence and 3 flipped away (exact
McNemar p = .004; paired-bootstrap 95% CI [+.114, +.455]; against the simple
agent, p = .007). On the
representative case above, the multimodal agent answered "leans down and
sniffs/nuzzles the small black puppy [28–36 s]" — matching the ground truth
almost verbatim from the keyframes alone; Figure 5.4 (fig8_whitedog_case) shows
the keyframes in question beside the two models' verbatim answers. Because everything except the evidence
modality was held constant, the improvement is attributable to visual evidence at
the answering stage. Together with Section 5.2, this closes the loop on the
modality question at *both* ends of the pipeline: retrieval needs visual
embeddings to find the right moment, and generation needs visual evidence to
describe it.

![White-dog qualitative case: retrieved keyframes and answers under text-only and visual evidence.](../figures/fig8_whitedog_case.pdf){#fig:white-dog width=96%}

### 5.4.3 Interpreting the .341 floor

Two readings of the text-only TN floor deserve separation. Part of the .14 margin
above random reflects genuine transcript evidence; part likely reflects
NExT-QA answer priors — a language model can sometimes reject implausible
distractors without any evidence. The controlled experiment bounds the effect:
whatever fraction of .341 is prior-driven, the additional .295 from keyframes is
not, since priors were identical across conditions.

A **prior-only baseline** makes the prior component concrete: the multimodal
model (claude-opus-4-8) answering with *no tool access at all* — the question
and its five options, nothing else — scores .560 overall and .455 on TN
(Table 5.4, third row; 150/150 completed). Three consequences follow. First,
NExT-QA's multiple-choice format carries a substantial answer prior for a
strong LLM: .560 without any evidence, far above the .200 random floor, which
calibrates every absolute accuracy in this chapter. Second, evidence still
matters even for the strongest model tested: on the identical 150 questions its
full retrieval-and-evidence stack adds +.167 over its own prior (.560 → .727;
37 questions flip to correct against 12 that flip away; exact McNemar
p = .0005, paired-bootstrap 95% CI [+.080, +.253]), and keyframe evidence
lifts TN from a .455 prior to .636 in the controlled experiment. Third, the
baseline sharpens the resource caveat of §5.4.4: the stronger model's prior
*alone* (.560) already exceeds the text-only agent stack's .547, so
comparisons across Table 5.4's rows conflate base-model strength with evidence
delivery — the clean attributions are the within-model ones.

### 5.4.4 Best-configuration run and contextualisation with published systems

The strongest components identified across this chapter were finally combined
into one configuration: the SigLIP index (§5.3.4) with ViT-L visual re-ranking
inside the agent's search tool, the LangGraph agent (§5.4.1), and the
multimodal evidence channel (§5.4.2). On the same 150-question sample this
configuration reaches **.727** overall (Table 5.4, bottom row; 95% CI
[.655, .798]; zero failed runs; ≈US$25 of API usage). The gains concentrate
where each component predicts: causal questions reach .852 (CW) and .762 (CH)
— the multimodal evidence lets the model *see* the causes it previously had to
guess — and TN rises from .341 to .568, confirming §5.4.2's diagnosis while
remaining the hardest type.

To situate these numbers, Table 5.7 lists published zero-shot results on the
NExT-QA validation set alongside ours, and the closest published temporal
grounding figures on NExT-GQA.

**Table 5.7 — Contextualisation with published systems (protocols differ; see
caveats below).**

| System | Protocol | NExT-QA val acc. |
|---|---|---|
| Random | — | .200 |
| SeViLA zero-shot (Yu et al., 2023) | full val, fine-tuned localiser | ≈.636 |
| LLoVi (Zhang et al., 2024) | full val, GPT-based | ≈.677 |
| VideoAgent (Wang et al., 2024) | full val, GPT-4 + iterative frame selection | .713 |
| **This work, best config** | grounded 150-q sample, ≤6 segments/search | **.727** [.655, .798] |

| Grounding | Protocol | metric |
|---|---|---|
| FrozenBiLM (NG+) (Xiao et al., 2024) | NExT-GQA test, answer grounding | mIoU .096 |
| Temp[CLIP] (NG+) (Xiao et al., 2024) | NExT-GQA test, answer grounding | mIoU .121 |
| SeViLA (Xiao et al., 2024) | NExT-GQA test, answer grounding | mIoU .217 |
| **This work, SigLIP + decomposition** | NExT-GQA val, top-1 retrieved segment | tIoU@1 **.230** |

Three caveats govern the reading of Table 5.7, and are repeated wherever these
numbers are cited. First, the QA comparison is *indicative, not a leaderboard
claim*: published systems evaluate the full ≈5,000-question validation set
while ours is a 150-question sample of the grounded subset, so the honest
statement is that the best configuration performs **at, and in point estimate
above, the published zero-shot state of the art on our sample**, with a
confidence interval that overlaps the strongest published figure. Second, the
comparison is not equal-resource: our system answers from at most six
retrieved segments per search under ≈US$0.17 per question, but benefits from a
stronger underlying LLM (claude-opus-4-8) than the GPT-4-era published
systems — LLM generation and method both differ, and this chapter's internal
ablations (not Table 5.7) are what isolate the method's contribution. The
prior-only baseline (§5.4.3) quantifies the model-strength component directly:
claude-opus-4-8 reaches .560 on these questions with no tools at all, so the
honest within-model attribution for the method is .560 → .727 (+.167,
p = .0005). Third,
the grounding rows compare related but non-identical protocols (their mIoU
grounds a QA model's answer on the test split; our tIoU@1 scores the top
retrieved segment on val); we include them because both measure "where the
system points for its evidence", where our retrieval-first design proves
competitive with — and in point estimate above — published weakly-supervised
grounding, while additionally emitting the citation as a first-class output.

## 5.5 System-level measurements

The offline stage processed all 567 videos in ≈2 h on a single RTX 4050 laptop
GPU (6 GB): keyframe extraction at <1 s per video and Whisper large-v3 (int8,
CTranslate2) transcription at 2–20 s per video; CLIP ViT-B-32 embedding and
indexing of 5,725 segments adds ≈15 min, and the SigLIP index ≈35 min. The
resulting index is portable (17 MB of vectors per backbone in npz form plus the
ChromaDB store) and queries run interactively: question encoding is
sub-second on GPU, HNSW search over 5,725 vectors is milliseconds, and the visual
re-ranker adds one batched matrix product over 30 candidates. End-to-end agent
answers complete in 15–60 s depending on tool rounds, dominated by LLM latency.
Marginal API costs are small: query decomposition for the entire split cost
≈US$0.15 (cached thereafter), a 150-question text-only QA run ≈US$0.35, and the
44-question multimodal run ≈US$3. The complete evaluation suite of this chapter
is reproducible on one consumer laptop plus ≈US$5 of API usage.

## 5.6 Summary

Against RQ1, multi-modal (visual) retrieval outperforms the text-only baseline by
an order of magnitude in corpus scope, and the corpus/video gap identifies
cross-video confusion as the dominant error. Against RQ2, the ablations show that
(i) naive late fusion cannot rescue a weak text channel at any weight; (ii) query
decomposition buys recall at depth while visual re-ranking buys top-rank
precision, and they compose; (iii) SigLIP is the strongest index backbone, nearly
doubling baseline R@1; and (iv) a two-stage cheap-index/expensive-re-rank design
matches the expensive index outright. Against RQ3, a LangGraph agent with
temporal tools and self-reflection improves QA accuracy by 10 points over a
single-loop agent (McNemar p = .031), and a controlled evidence-channel
experiment shows visual evidence nearly doubling temporal-next accuracy
(p = .004) — establishing that modality matters at the answering stage
independently of retrieval. A prior-only baseline calibrates all QA numbers:
the strongest model scores .560 with no evidence at all, and the full stack's
+.167 above that prior (p = .0005) is the method's within-model contribution.
Combining the best-measured component in every slot yields .727 five-way
accuracy — at, and in point estimate above, the published zero-shot state of
the art on our sample — while temporal grounding proves competitive with
published weakly-supervised answer-grounding methods under a stricter,
citation-producing protocol. Chapter 6 discusses the limitations and
generality of these findings.
