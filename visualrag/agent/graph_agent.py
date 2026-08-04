"""W7 LangGraph agent — ReAct tool loop + self-reflection state machine.

Graph:            +----------- revise -----------+
                  v                              |
    agent --tool_calls--> tools --> agent ... --draft--> reflect --accept--> END

Two tools:
  - search_video_segments (W4/W5): retrieval, optionally restricted to a video.
  - get_segments_around   (W6 temporal): segments immediately before/after a
    timestamp. For "what happened after X": search anchors X, then walk `after`.

The reflect node re-reads the draft against the gathered evidence and either
accepts or sends one critique back (bounded self-reflection, plan §2 W7).

Providers: any OpenAI-compatible endpoint (DeepSeek / local, text-only tool
results) or Anthropic Claude (multimodal — keyframes embedded in tool results),
sharing one state machine. The provider therefore selects the *evidence
channel*, exactly as in the simple agent (§3.5).
"""

from __future__ import annotations

import json
import re
from typing import Optional, TypedDict

from langgraph.graph import StateGraph, END

from visualrag.agent.answerer import SEARCH_TOOL, SYSTEM_PROMPT, TEMPORAL_TOOL, VideoQA

GRAPH_SYSTEM = SYSTEM_PROMPT + """

Temporal questions: when the question asks what happened AFTER or BEFORE some event,
first find the event with search_video_segments, then use get_segments_around to read
the adjacent segments in time order. Base the answer on those, not on the anchor."""

REFLECT_PROMPT = """You are auditing a draft answer produced from video-segment evidence.

Question: {question}

Evidence gathered (all segments the tools returned; keyframe images may have been
shown to the answerer but are not reproduced here):
{evidence}

Draft answer:
{draft}

Check: (1) every [video @ start-end s] citation in the draft refers to a segment that
actually appears in the evidence; (2) the claims are supported by the cited segments
rather than invented; (3) if the question is temporal (after/before), the draft uses
segments in the right temporal relation to the anchor event.

Output JSON only: {{"verdict": "accept"}} if the draft passes, or
{{"verdict": "revise", "critique": "<one short paragraph: what is wrong and what to do>"}}."""


class AgentState(TypedDict, total=False):
    question: str
    video_id: Optional[str]
    messages: list
    searches: list
    rounds: int
    reflections: int
    draft: str
    answer: str
    usage: dict


class GraphVideoQA:
    """LangGraph wrapper. Public API mirrors VideoQA.answer()."""

    MAX_ROUNDS = 6
    MAX_REFLECTIONS = 1

    def __init__(self, cfg):
        self.qa = VideoQA(cfg)  # reuse retrieval, segment lookup, provider client
        self.claude = self.qa.provider == "claude"
        if not self.claude:
            self.oa_tools = [
                {"type": "function", "function": {
                    "name": t["name"], "description": t["description"],
                    "parameters": t["input_schema"]}}
                for t in (SEARCH_TOOL, TEMPORAL_TOOL)
            ]
        self.graph = self._build()

    # --- helpers ---------------------------------------------------------

    @staticmethod
    def _strip_tool_markup(text: str) -> str:
        """Forced-answer replies sometimes contain the model's tool-call markup
        as literal text (DeepSeek DSML tags) — cut everything from the first tag."""
        return re.split(r"<[^<>]{0,12}DSML", text)[0].strip()

    def _bump_usage(self, state: AgentState, in_tok: int, out_tok: int):
        u = state.setdefault("usage", {"input_tokens": 0, "output_tokens": 0})
        u["input_tokens"] += in_tok
        u["output_tokens"] += out_tok

    def _exec_tool(self, name: str, args: dict, video_id: Optional[str], as_blocks: bool):
        """Run a tool; return content (blocks for claude, text otherwise)."""
        result, kind = self.qa.run_tool(name, args, video_id)
        if kind == "search":
            self_searches_entry = {"query": args.get("query", ""),
                                   "modality": args.get("modality", self.qa.modality),
                                   "hits": result}
            content = self.qa.hits_to_blocks(result) if as_blocks else self.qa.hits_to_text(result)
        else:
            self_searches_entry = {"tool": name, "args": args}
            content = self.qa.rows_to_blocks(result) if as_blocks else self.qa.rows_to_text(result)
        return self_searches_entry, content

    # --- graph nodes: agent ----------------------------------------------

    def _agent_node(self, state: AgentState) -> AgentState:
        state["rounds"] = state.get("rounds", 0) + 1
        force = state["rounds"] >= self.MAX_ROUNDS
        if force:
            state["messages"].append({"role": "user", "content": (
                "Tool budget exhausted — do NOT request more tools. Give your final "
                "answer now from the evidence already gathered.")})
        if self.claude:
            resp = self.qa.client.messages.create(
                model=self.qa.model, max_tokens=self.qa.max_tokens,
                thinking={"type": "adaptive"},
                system=GRAPH_SYSTEM,
                tools=[SEARCH_TOOL, TEMPORAL_TOOL],
                **({"tool_choice": {"type": "none"}} if force else {}),
                messages=state["messages"],
            )
            self._bump_usage(state, resp.usage.input_tokens, resp.usage.output_tokens)
            state["messages"].append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                state["draft"] = "".join(
                    b.text for b in resp.content if b.type == "text").strip()
        else:
            resp = self.qa.client.chat.completions.create(
                model=self.qa.model, max_tokens=4096, tools=self.oa_tools,
                tool_choice="none" if force else "auto",
                messages=state["messages"],
            )
            self._bump_usage(state, resp.usage.prompt_tokens, resp.usage.completion_tokens)
            msg = resp.choices[0].message
            state["messages"].append(msg)
            if not msg.tool_calls:
                state["draft"] = self._strip_tool_markup(msg.content or "")
        return state

    # --- graph nodes: tools ------------------------------------------------

    def _tools_node(self, state: AgentState) -> AgentState:
        last = state["messages"][-1]
        vid = state.get("video_id")
        if self.claude:
            results = []
            for block in last["content"]:
                if block.type == "tool_use":
                    entry, content = self._exec_tool(block.name, dict(block.input), vid,
                                                     as_blocks=True)
                    state["searches"].append(entry)
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": content})
            state["messages"].append({"role": "user", "content": results})
        else:
            for call in last.tool_calls:
                args = json.loads(call.function.arguments)
                entry, content = self._exec_tool(call.function.name, args, vid,
                                                 as_blocks=False)
                state["searches"].append(entry)
                state["messages"].append({"role": "tool", "tool_call_id": call.id,
                                          "content": content})
        return state

    # --- graph nodes: reflect ----------------------------------------------

    def _evidence_text(self, state: AgentState) -> str:
        """Text view of all tool results gathered so far (for the reflect audit)."""
        parts = []
        for m in state["messages"]:
            if self.claude:
                if isinstance(m, dict) and m.get("role") == "user" \
                        and isinstance(m.get("content"), list):
                    for block in m["content"]:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            for c in block["content"]:
                                if isinstance(c, dict) and c.get("type") == "text":
                                    parts.append(c["text"])
            else:
                if isinstance(m, dict) and m.get("role") == "tool":
                    parts.append(m["content"])
        return "\n".join(parts)[-8000:]

    def _reflect_node(self, state: AgentState) -> AgentState:
        state["reflections"] = state.get("reflections", 0) + 1
        prompt = REFLECT_PROMPT.format(question=state["question"],
                                       evidence=self._evidence_text(state) or "(none)",
                                       draft=state["draft"])
        if self.claude:
            resp = self.qa.client.messages.create(
                model=self.qa.model, max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            self._bump_usage(state, resp.usage.input_tokens, resp.usage.output_tokens)
            text = "".join(b.text for b in resp.content if b.type == "text")
        else:
            resp = self.qa.client.chat.completions.create(
                model=self.qa.model, max_tokens=512,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            self._bump_usage(state, resp.usage.prompt_tokens, resp.usage.completion_tokens)
            text = resp.choices[0].message.content or ""
        try:
            verdict = json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))
        except Exception:
            verdict = {"verdict": "accept"}

        # Revising needs budget: at least one tool round plus the re-answer.
        can_revise = state["rounds"] <= self.MAX_ROUNDS - 2
        if verdict.get("verdict") == "revise" and can_revise \
                and state["reflections"] <= self.MAX_REFLECTIONS:
            state["messages"].append({"role": "user", "content": (
                "A reviewer found problems with your draft answer:\n"
                f"{verdict.get('critique', '')}\n"
                "Fix them — gather more evidence with the tools if needed, then answer again.")})
            state["draft"] = ""
        else:
            state["answer"] = state["draft"]
        return state

    # --- graph wiring ------------------------------------------------------

    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("agent", self._agent_node)
        g.add_node("tools", self._tools_node)
        g.add_node("reflect", self._reflect_node)
        g.set_entry_point("agent")
        g.add_conditional_edges(
            "agent",
            lambda s: "tools" if not s.get("draft") else "reflect",
            {"tools": "tools", "reflect": "reflect"})
        g.add_edge("tools", "agent")
        g.add_conditional_edges(
            "reflect",
            lambda s: "end" if s.get("answer") else "agent",
            {"end": END, "agent": "agent"})
        return g.compile()

    # --- public API ----------------------------------------------------------

    def answer(self, question: str, video_id: Optional[str] = None, **_) -> dict:
        user_text = question if not video_id else (
            f"{question}\n\n(Search is restricted to video {video_id}; "
            f"the question is about that video's timeline.)")
        messages = [{"role": "user", "content": user_text}]
        if not self.claude:
            messages.insert(0, {"role": "system", "content": GRAPH_SYSTEM})
        state: AgentState = {
            "question": question, "video_id": video_id,
            "messages": messages,
            "searches": [], "rounds": 0, "reflections": 0,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        final = self.graph.invoke(state, {"recursion_limit": 40})
        return {"answer": final.get("answer") or final.get("draft", ""),
                "searches": final["searches"], "usage": final["usage"],
                "rounds": final["rounds"], "reflections": final["reflections"]}
