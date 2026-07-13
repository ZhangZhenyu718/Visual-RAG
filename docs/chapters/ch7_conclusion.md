# Chapter 7 — Conclusion and Future Work

## 7.1 Answers to the research questions

**RQ1 — retrieval effectiveness against text-only baselines.** A multi-modal
pipeline retrieving on visual embeddings outperforms transcript-based retrieval
by an order of magnitude at moment granularity on everyday video (corpus-scope
R@10 .111 vs .011 at tIoU ≥ 0.5), and the fully developed system — SigLIP index
with LLM query decomposition — nearly doubles the initial visual baseline itself
(R@1 .026 → .048; video-scope tIoU@1 .230). Scope analysis attributes the
remaining difficulty to cross-video confusion rather than within-video
localisation: given the right video, the system places a correct moment in its
top five for two-thirds of questions at the looser τ = 0.3 (R@5 .677). The
answer to RQ1 is therefore affirmative with a precise caveat: *visual* retrieval
is effective and improvable; retrieval on what is *said* is, on this domain, not
a viable baseline so much as a cautionary one.

**RQ2 — the multi-modal embedding strategy.** The experiments support a
strategy, not merely a ranking: index the visual channel with the strongest
affordable contrastive backbone (SigLIP SO400M dominated both CLIP variants,
+69% relative R@1 as the index); if the strongest backbone is unaffordable at
corpus scale, index cheaply and re-rank its top candidates with the stronger
model, which matched the expensive index to within measurement noise; translate
questions into the encoder's native register (scene captions) via LLM
decomposition for deep-rank recall; and do *not* fuse in a weak text channel at
any weight, nor re-rank with a text cross-encoder — both degrade quality for an
identified cause (sparse, multilingual, conversational speech). The optimal
strategy for audio and on-screen text on this benchmark is the null strategy;
Chapter 6 marks the domains where that conclusion should be re-tested.

**RQ3 — agentic decomposition and synthesis.** Query decomposition, temporal
tool use, and bounded self-reflection each demonstrably contribute. At the
retrieval layer, decomposition is safe-by-construction (the original query is
always retained) and lifts recall. At the answering layer, a LangGraph state
machine that anchors events, walks the timeline, and audits its own citations
raises five-way QA accuracy from .447 to .547 over a single-loop agent, with
the largest gains on causal and temporal-current questions. The controlled
evidence-channel experiment completes the answer: agentic machinery alone
cannot compensate for evidence the model cannot perceive — supplying keyframes
to the answerer nearly doubled temporal-next accuracy (.341 → .636) with
everything else held fixed. An effective video-QA agent is therefore *jointly*
a retrieval planner and a multimodal reader; either half alone leaves the
other's headroom unrealised.

## 7.2 Future work

Six directions follow directly from the limitations identified in Chapter 6,
ordered roughly by evidence-per-effort.

**Repair the text channel, then re-test fusion.** Machine-translate transcripts
to English (or adopt a multilingual text-retrieval encoder) before indexing,
disentangling the informativeness/readability confound of §6.2 — and only then
revisit late fusion and text re-ranking, whose negative results are conditional
on the broken channel.

**Combine the graph agent with multimodal evidence.** The strongest agent and
the strongest evidence channel were never run together; extending the LangGraph
harness to the Anthropic API (or a vision-capable open model) tests whether
reflection can audit *visual* claims, and how far beyond .636 the TN ceiling
moves.

**Complete the open-source model comparison.** The provider abstraction already
targets any OpenAI-compatible endpoint; running the QA suite against local
Qwen/Llama models (the plumbing and evaluation scripts exist) would quantify
the API-versus-open trade-off the project plan anticipated.

**A prior-only QA baseline.** Answering the multiple-choice questions with no
tool access would bound the answer-prior component of all QA numbers (§6.3),
sharpening every agent comparison at negligible cost.

**Adaptive segmentation.** Fixed 8-second windows impose a structural ceiling on
tIoU (§6.2); shot-aligned or query-adaptive windowing would raise the ceiling
itself, and the evaluation harness can measure the gain directly.

**A second domain and a user study.** Replicating the modality ablations on
speech-dense video (lectures, meetings) tests the conditionality claims of
Chapter 6, and the descoped user study remains the right instrument for claims
about end-user utility that no offline metric can support.

## 7.3 Closing remark

The project set out to make a moment of video as findable as a sentence of
text, under the discipline that every design choice be measured. Its most
useful legacy may be less the headline numbers than the shape of the evidence:
failures separated into recall, precision, and evidence problems; negative
results retained with their causes; and a pipeline in which each remedy could
be switched on, switched off, and priced. Video understanding systems will
continue to change quickly; an evaluation harness that can say *which part
helped, by how much, and why* will remain the transferable asset.
