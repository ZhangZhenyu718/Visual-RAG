# Chapter 6 — Discussion

## 6.1 Synthesis: three separable problems

The clearest lesson of Chapter 5 is that the end-to-end failures of a video RAG
system decompose into three problems that are *separable* — they have different
causes, respond to different interventions, and the interventions compose.

The first is a **recall problem**: the correct moment never enters the candidate
pool because the query is phrased in causal or temporal language that a
contrastive vision-language encoder cannot match against pixels. Query
decomposition addresses exactly this — rewriting the question into scene captions
lifted deep-rank recall (corpus R@10 +10% relative) while leaving the top rank
untouched (§5.3.2). The second is a **precision problem**: the correct moment is
in the pool but near-tied cosine scores cannot order it first. Second-stage
re-scoring with a stronger visual model addresses this, and only this (§5.3.3);
so does exchanging the index backbone outright (§5.3.4). The third is an
**evidence problem**: the correct moment is retrieved, correctly time-stamped,
and placed in front of the answerer — which then cannot answer because the
decisive evidence is visual and its input is text. No amount of retrieval
improvement fixes this; changing the evidence channel does (+87% on
temporal-next questions, §5.4.2).

Reading the results this way explains why the interventions compose additively
rather than cannibalising each other (Table 5.2): each removes a different
bottleneck. It also offers a diagnostic recipe for practitioners: compare corpus
against video scope to separate cross-video confusion from localisation error,
compare shallow against deep recall to separate recall from precision problems,
and compare text-only against multimodal answering to expose evidence starvation.

A second thread worth drawing out is that **the value of a modality is
role-dependent, not absolute**. The transcript channel failed twice as a
*retrieval key* — as embeddings (§5.2) and as a cross-encoder signal (§5.3.3) —
yet the same transcripts, delivered as *evidence* to the answering LLM, sustain
QA accuracy far above the random floor (.547 overall) and clearly carry the
causal questions, where much of the gain over chance cannot plausibly come from
priors alone. Speech in everyday video is a poor pointer to *where* something
happens, but once the right moment is found it still describes some of *what*
happens. Modality ablations that report a single "contribution" number conflate
these two roles; our results suggest they should be measured separately.

Finally, the two-stage equivalence result (§5.3.4) has an architectural
implication beyond this project. At 5,725 segments, indexing with the expensive
model is trivially affordable, and yet the cheap-index/expensive-re-rank design
already matched it to within measurement noise. Since re-ranking cost is fixed
(top-30 per query) while indexing cost grows linearly with the corpus, the
two-stage design is the only one of the two that survives scaling — and this
work provides evidence that the quality sacrifice can be nil.

## 6.2 Limitations

**Single benchmark, single domain.** All conclusions are drawn from NExT-QA/GQA:
short, everyday, speech-sparse home videos. The negative results about the text
channel — fusion never helping (§5.3.1), the text cross-encoder hurting
(§5.3.3) — are best read as *conditional on this domain*. In speech-dense,
single-language domains (lectures, meetings, news), transcripts are far more
informative and the same components may well invert sign. The positive results
(decomposition, visual re-ranking, backbone ordering, evidence-channel effect)
rest on properties of contrastive encoders and of the QA task rather than of
this corpus, and should transfer more readily; verifying this on a second
benchmark is the most direct extension of this work.

**A confound inside the text-channel result.** The weak text channel has two
entangled causes: the transcripts are often *uninformative* (chatter unrelated to
the questions), and they are often *unreadable* by the predominantly-English CLIP
text encoder (multilingual speech). The present experiments cannot apportion
blame between the two. A translate-then-embed variant — machine-translating all
transcripts to English before indexing — would separate them cheaply and was left
undone only for time.

**Segmentation granularity bounds temporal accuracy.** Segments are fixed 8-second
windows on a 4-second stride, while ground-truth intervals vary freely; even a
perfect ranker cannot exceed the tIoU that the best-aligned window achieves.
Part of the headline gap between τ = 0.5 and τ = 0.3 results (§5.3.5) is this
quantisation, not ranking error. tIoU@1 therefore carries a structural ceiling
that is a property of the segmenter; adaptive or shot-aligned segmentation would
raise the ceiling itself.

**Multiple-choice accuracy is a proxy.** NExT-QA's five-way format enables cheap,
objective scoring, but it measures answer *selection*, not answer *quality*: the
faithfulness of the generated prose and the correctness of its timestamp
citations are audited only by the agent's own reflection pass, not by human or
LLM-as-judge evaluation. The white-dog case study (§5.4.2) illustrates that the
generated answers can be precise and well-grounded, but no systematic
open-ended-answer evaluation was performed.

**Sample sizes and single runs.** Retrieval results cover all 3,358 questions and
are deterministic given the frozen decomposition cache. The QA experiments,
however, are budget-limited: 150 questions for the agent comparison, 44 for the
controlled vision experiment, and the text cross-encoder was rejected on a
100-question development subset. All agent numbers are single runs of
non-deterministic API models without confidence intervals; the reported effects
(+10 points overall, +29.5 points on TN) are large relative to plausible
variance, but small differences in Table 5.4's by-type columns (n ≤ 22) should
not be over-read.

**Asymmetries in the agent comparison.** The LangGraph agent currently targets
OpenAI-compatible providers only, so the controlled vision experiment ran under
the *simple* agent; the strongest configuration observed (multimodal evidence)
and the strongest agent (graph) have never been combined, and the .636 TN result
is therefore likely a floor, not a ceiling. Symmetrically, the graph agent's
reflection pass and the multimodal channel might interact (e.g. reflection could
audit visual claims), which remains untested.

**No user study.** The project plan scheduled a small user study (W10); it was
descoped in favour of the controlled vision experiment and the ablation matrix,
which we judged to produce more defensible evidence per unit time. The Streamlit
demo (§3.6) stands in for qualitative validation, but claims about end-user
utility remain unsupported and are deliberately absent from this dissertation.

## 6.3 Threats to validity

**Construct validity.** The hit criterion (correct video ∧ tIoU ≥ τ) is stricter
than video-level retrieval metrics common in prior work; this deflates absolute
numbers relative to that literature but is faithful to the moment-retrieval
construct. For QA, the multiple-choice format admits answer-prior effects: a
language model can reject implausible distractors without evidence. The
controlled experiment bounds this — priors were identical across both evidence
conditions, so the +29.5-point effect cannot be prior-driven — but the absolute
.341 text-only floor mixes prior with evidence in unknown proportion; a
no-retrieval prior-only baseline would complete the picture.

**Internal validity.** The decomposition cache is shared verbatim across all
configurations that use it, eliminating decomposition variance from those
comparisons. The re-ranker's fusion weight (0.5) and candidate pool (30) were
fixed a priori rather than tuned, and the development subset used to reject the
text cross-encoder (n = 100) is disjoint in purpose but not in items from the
full evaluation set; since the decision it informed was to *exclude* a component,
any bias is conservative. Agent experiments called the same LLM family for
decomposition and answering (DeepSeek), which could in principle correlate their
errors; retrieval experiments are unaffected, as decomposition quality is
identical across all retrieval configurations.

**External validity.** Beyond the domain limitation above, two scale caveats
apply. The corpus (567 videos, 5,725 segments) is small enough that exhaustive
HNSW search is effectively exact; ANN recall degradation at millions of segments
is not measured. And the two-stage equivalence claim is demonstrated at a scale
where *both* designs are feasible — its extrapolation to large corpora rests on
the argument of §6.1, not on direct measurement. Finally, all API-based results
are tied to unversioned commercial model endpoints (deepseek-chat,
claude-opus-4-8, mid-2026); the committed prompts, caches, and result files make
the analysis auditable, but bit-exact replication of agent runs is not
guaranteed by the providers.

## 6.4 Summary

The findings hold together as a single narrative: in everyday video, visual
signal dominates at every stage where a modality choice exists, and the
remaining errors are separable into recall, precision, and evidence problems
with independent, composable remedies. The principal limitations — one domain,
proxy QA metrics, budget-limited agent samples, and an untested
graph-plus-vision combination — are all addressable with more compute rather
than new machinery, and none threatens the direction of the reported effects,
whose magnitudes (+69% relative R@1 from the backbone, +87% from the evidence
channel) far exceed the plausible noise floor. Chapter 7 concludes and sets out
the corresponding future work.
