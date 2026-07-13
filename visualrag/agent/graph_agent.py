"""W7: LangGraph agent — ReAct tool loop + self-reflection state machine.

Graph:            +----------- revise -----------+
                  v                              |
    agent --tool_calls--> tools --> agent ... --draft--> reflect --accept--> END

Two tools:
  - search_video_segments (W4/W5): retrieval, optionally restricted to a video.
  - get_segments_around   (W6 temporal): segments immediately before/after a
    timestamp. For "what happened after X": search anchors X, then walk `after`.

The reflect node re-reads the draft against the gathered evidence and either
accepts or sends one critique back (bounded self-reflection, plan §2 W7).

LLM calls go through the OpenAI-compatible API (DeepSeek by default) — same
provider plumbing as visualrag.agent.answerer.
"""

from __future__ import annotations

import json
import os
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

Evidence gathered (all segments the tools returned):
{evidence}

Draft answer:
{draft}

Check: (1) every [video @ start-end s] citation in the draft refers to a segment that
actually appears in the evidence; (2) the claims are supported by the cited segments'
transcripts/timing rather than invented; (3) if the question is temporal (after/before),
the draft uses segments in the right temporal relation to the anchor event.

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
        if self.qa.provider == "claude":
            raise NotImplementedError("graph agent currently targets OpenAI-compatible "
                                      "providers (deepseek/local); use the simple agent for claude")
        self.tools = [
            {"type": "function", "function": {
                "name": t["name"], "description": t["description"],
                "parameters": t["input_schema"]}}
            for t in (SEARCH_TOOL, TEMPORAL_TOOL)
        ]
        self.graph = self._build()

    # --- LLM + tool plumbing -------------------------------------------

    def _chat(self, state: AgentState, tool_choice="auto"):
        resp = self.qa.client.chat.completions.create(
            model=self.qa.model, max_tokens=4096, tools=self.tools,
            tool_choice=tool_choice, messages=state["messages"],
        )
        u = state.setdefault("usage", {"input_tokens": 0, "output_tokens": 0})
        u["input_tokens"] += resp.usage.prompt_tokens
        u["output_tokens"] += resp.usage.completion_tokens
        return resp.choices[0].message

    def _run_tool(self, name: str, args: dict, video_id: Optional[str]) -> str:
        result, kind = self.qa.run_tool(name, args, video_id)
        return self.qa.hits_to_text(result) if kind == "search" else self.qa.rows_to_text(result)

    # --- graph nodes -----------------------------------------------------

    @staticmethod
    def _strip_tool_markup(text: str) -> str:
        """Forced-answer replies sometimes contain the model's tool-call markup
        as literal text (DeepSeek DSML tags) — cut everything from the first tag."""
        return re.split(r"<[^<>]{0,12}DSML", text)[0].strip()

    def _agent_node(self, state: AgentState) -> AgentState:
        state["rounds"] = state.get("rounds", 0) + 1
        force_answer = state["rounds"] >= self.MAX_ROUNDS
        if force_answer:
            state["messages"].append({"role": "user", "content": (
                "Tool budget exhausted — do NOT request more tools. Give your final "
                "answer now from the evidence already gathered.")})
        msg = self._chat(state, tool_choice="none" if force_answer else "auto")
        state["messages"].append(msg)
        if not msg.tool_calls:
            state["draft"] = self._strip_tool_markup(msg.content or "")
        return state

    def _tools_node(self, state: AgentState) -> AgentState:
        msg = state["messages"][-1]
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            result = self._run_tool(call.function.name, args, state.get("video_id"))
            state["searches"].append({"tool": call.function.name, "args": args})
            state["messages"].append({"role": "tool", "tool_call_id": call.id,
                                      "content": result})
        return state

    def _reflect_node(self, state: AgentState) -> AgentState:
        state["reflections"] = state.get("reflections", 0) + 1
        evidence = "\n".join(m["content"] for m in state["messages"]
                             if isinstance(m, dict) and m.get("role") == "tool")[-8000:]
        prompt = REFLECT_PROMPT.format(question=state["question"],
                                       evidence=evidence or "(none)",
                                       draft=state["draft"])
        resp = self.qa.client.chat.completions.create(
            model=self.qa.model, max_tokens=512,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        u = state["usage"]
        u["input_tokens"] += resp.usage.prompt_tokens
        u["output_tokens"] += resp.usage.completion_tokens
        try:
            verdict = json.loads(re.search(r"\{.*\}", resp.choices[0].message.content,
                                           re.DOTALL).group(0))
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
        state: AgentState = {
            "question": question, "video_id": video_id,
            "messages": [{"role": "system", "content": GRAPH_SYSTEM},
                         {"role": "user", "content": user_text}],
            "searches": [], "rounds": 0, "reflections": 0,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        final = self.graph.invoke(state, {"recursion_limit": 40})
        return {"answer": final.get("answer") or final.get("draft", ""),
                "searches": final["searches"], "usage": final["usage"],
                "rounds": final["rounds"], "reflections": final["reflections"]}
