"""W4 MVP vertical slice: question -> retrieve -> LLM answer with timestamp citations.

Single-turn native function-calling (plan §2): Claude is given one tool,
`search_video_segments`, backed by the Retriever. Tool results carry each hit's
timestamps + transcript text + a keyframe image, so the model grounds its answer
in what the segment actually shows, and must cite [video_id start-end s].
"""

from __future__ import annotations

import base64
import os
from typing import Optional

from visualrag.schema import read_jsonl


SYSTEM_PROMPT = """You are a video question-answering assistant backed by a multi-modal \
video retrieval index. Every segment you receive from the search tool has a video id, \
a [start-end] time range in seconds, an optional speech transcript, and a keyframe image.

Rules:
- Ground every claim in the retrieved segments (image content and/or transcript).
- Cite evidence inline as [video_id @ start-end s] right after the claim it supports.
- The transcript may be empty or in another language; the keyframe shows what is on screen.
- If the retrieved segments do not contain enough evidence to answer, say so explicitly \
and describe the closest thing you did find. Never invent visual details.
- Temporal questions (what happened AFTER/BEFORE X): first locate the anchor moment X \
with search_video_segments, then use get_segments_around to read the adjacent segments \
in time order — base the answer on those, not on the anchor segment alone.
- Answer concisely: a direct answer first, then the supporting evidence."""

SEARCH_TOOL = {
    "name": "search_video_segments",
    "description": (
        "Search the video corpus for short temporal segments relevant to a natural-language "
        "query. Returns the top-k segments, each with video_id, [start-end] seconds, speech "
        "transcript, and a keyframe image. Call this before answering any question about the "
        "videos. Phrase the query as a visual/audio description of what to find (e.g. 'a boy "
        "unwrapping a present on a sofa'), not as a question."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to look for, phrased as a description of the scene or speech.",
            },
            "modality": {
                "type": "string",
                "enum": ["visual", "text", "fused"],
                "description": (
                    "Which index to search: 'visual' (what is on screen — the default and "
                    "strongest), 'text' (what is said in the transcript), 'fused' (both)."
                ),
            },
        },
        "required": ["query"],
    },
}


TEMPORAL_TOOL = {
    "name": "get_segments_around",
    "description": (
        "Get the video segments immediately BEFORE or AFTER a timestamp in a specific "
        "video. Essential for temporal questions ('what did X do AFTER/BEFORE Y?'): "
        "first locate the anchor moment Y with search_video_segments, then call this "
        "with the anchor's end time and direction='after' (or start time and 'before') "
        "to see what actually happens next — do not guess from the anchor segment alone."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "video_id": {"type": "string"},
            "timestamp": {"type": "number", "description": "anchor time in seconds"},
            "direction": {"type": "string", "enum": ["after", "before", "around"]},
            "n": {"type": "integer", "description": "number of segments (default 4)"},
        },
        "required": ["video_id", "timestamp", "direction"],
    },
}


class _SegmentLookup:
    """segment_id -> {keyframe_paths, transcript}, lazily loaded per video."""

    def __init__(self, cfg):
        self.seg_dir = os.path.join(cfg.get_path("paths.artifacts", "artifacts"), "segments")
        self._by_video: dict[str, dict[str, dict]] = {}

    def _video(self, video_id: str) -> dict[str, dict]:
        if video_id not in self._by_video:
            path = os.path.join(self.seg_dir, f"{video_id}.jsonl")
            rows = read_jsonl(path) if os.path.exists(path) else []
            self._by_video[video_id] = {r["segment_id"]: r for r in rows}
        return self._by_video[video_id]

    def get(self, segment_id: str) -> dict:
        return self._video(segment_id.split("::", 1)[0]).get(segment_id, {})

    def around(self, video_id: str, timestamp: float, direction: str = "after",
               n: int = 4) -> list[dict]:
        """Segments adjacent to a timestamp — the temporal-reasoning primitive
        ("what happened after X": anchor X via search, then walk `after`)."""
        rows = sorted(self._video(video_id).values(), key=lambda r: r["start"])
        if direction == "after":
            picked = [r for r in rows if r["start"] >= timestamp - 1e-6]
        elif direction == "before":
            picked = [r for r in rows if r["end"] <= timestamp + 1e-6][::-1]
        else:  # around
            picked = sorted(rows, key=lambda r: abs((r["start"] + r["end"]) / 2 - timestamp))
        return picked[:n]


def _image_block(path: str) -> Optional[dict]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


class VideoQA:
    """Provider-agnostic answerer. `agent.provider`:
    - "claude": Anthropic SDK, multimodal — keyframes go into the tool result.
    - "deepseek": OpenAI-compatible API, TEXT-ONLY — no vision support, so the
      model grounds on transcripts + timestamps only (visual evidence omitted).
    """

    def __init__(self, cfg, retriever=None):
        self.cfg = cfg
        if retriever is None:
            from visualrag.retrieve.retriever import Retriever
            retriever = Retriever(cfg)
        self.retriever = retriever
        self.segments = _SegmentLookup(cfg)

        self.provider = cfg.get_path("agent.provider", "claude")
        _defaults = {"claude": ("claude-opus-4-8", None, None),
                     "deepseek": ("deepseek-chat", "https://api.deepseek.com", "DEEPSEEK_API_KEY"),
                     "local": ("qwen2.5:7b-instruct", "http://localhost:11434/v1", None)}
        if self.provider not in _defaults:
            raise ValueError(f"unknown agent.provider {self.provider!r}")
        default_model, default_url, self._key_env = _defaults[self.provider]
        self.model = cfg.get_path("agent.model", default_model)
        self.base_url = cfg.get_path("agent.base_url", default_url)
        self.k = int(cfg.get_path("agent.k", 6))
        self.modality = cfg.get_path("agent.modality", "visual")  # W3: visual > fused at alpha 0.5
        self.alpha = float(cfg.get_path("agent.alpha", 0.8))
        self.max_images = int(cfg.get_path("agent.max_images", 6))
        self.max_tokens = int(cfg.get_path("agent.max_tokens", 16000))
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if self.provider == "claude":
                import anthropic
                self._client = anthropic.Anthropic()
            else:  # any OpenAI-compatible endpoint (DeepSeek, Ollama, vLLM, ...)
                from openai import OpenAI
                key = os.environ[self._key_env] if self._key_env else "not-needed"
                self._client = OpenAI(api_key=key, base_url=self.base_url)
        return self._client

    # --- Tool execution -------------------------------------------------

    def search(self, query: str, modality: Optional[str] = None,
               video_id: Optional[str] = None) -> list[dict]:
        where = {"video_id": video_id} if video_id else None
        return self.retriever.search(
            query, modality=modality or self.modality, k=self.k,
            alpha=self.alpha, where=where,
        )

    def hits_to_blocks(self, hits: list[dict]) -> list[dict]:
        """Render retrieval hits as tool_result content blocks (text + keyframes)."""
        blocks: list[dict] = []
        images_used = 0
        for rank, h in enumerate(hits, 1):
            m = h["metadata"]
            seg = self.segments.get(h["segment_id"])
            transcript = (seg.get("transcript", "") + " " + seg.get("ocr_text", "")).strip()
            blocks.append({"type": "text", "text": (
                f"[{rank}] video_id={m.get('video_id')} time={m.get('start')}-{m.get('end')}s "
                f"score={h['score']:.3f}\n"
                f"transcript: {transcript if transcript else '(no speech)'}\n"
                f"keyframe:"
            )})
            kfs = seg.get("keyframe_paths", [])
            img = _image_block(kfs[len(kfs) // 2]) if kfs else None
            if img and images_used < self.max_images:
                blocks.append(img)
                images_used += 1
            else:
                blocks.append({"type": "text", "text": "(no keyframe available)"})
        if not blocks:
            blocks.append({"type": "text", "text": "No segments found."})
        return blocks

    def rows_to_blocks(self, rows: list[dict]) -> list[dict]:
        """Render raw segment rows (temporal tool) as content blocks with keyframes."""
        blocks: list[dict] = []
        used = 0
        for i, r in enumerate(rows, 1):
            transcript = (r.get("transcript", "") + " " + r.get("ocr_text", "")).strip()
            blocks.append({"type": "text", "text": (
                f"[{i}] video_id={r['video_id']} time={r['start']}-{r['end']}s\n"
                f"transcript: {transcript if transcript else '(no speech)'}\nkeyframe:")})
            kfs = r.get("keyframe_paths", [])
            img = _image_block(kfs[len(kfs) // 2]) if kfs else None
            if img and used < self.max_images:
                blocks.append(img)
                used += 1
            else:
                blocks.append({"type": "text", "text": "(no keyframe available)"})
        return blocks or [{"type": "text", "text": "No segments in that direction."}]

    def rows_to_text(self, rows: list[dict]) -> str:
        if not rows:
            return "No segments in that direction."
        return "\n\n".join(
            f"[{i}] video_id={r['video_id']} time={r['start']}-{r['end']}s\n"
            f"transcript: {((r.get('transcript') or '') + ' ' + (r.get('ocr_text') or '')).strip() or '(no speech)'}"
            for i, r in enumerate(rows, 1))

    def run_tool(self, name: str, args: dict, video_id: Optional[str] = None):
        """Execute a tool call; returns (hits_or_rows, kind)."""
        if name == "get_segments_around":
            rows = self.segments.around(args["video_id"], float(args["timestamp"]),
                                        args.get("direction", "after"), int(args.get("n", 4)))
            return rows, "temporal"
        hits = self.search(args["query"], args.get("modality"), video_id=video_id)
        return hits, "search"

    def hits_to_text(self, hits: list[dict]) -> str:
        """Text-only rendering of hits (for providers without vision)."""
        lines = []
        for rank, h in enumerate(hits, 1):
            m = h["metadata"]
            seg = self.segments.get(h["segment_id"])
            transcript = (seg.get("transcript", "") + " " + seg.get("ocr_text", "")).strip()
            lines.append(
                f"[{rank}] video_id={m.get('video_id')} time={m.get('start')}-{m.get('end')}s "
                f"score={h['score']:.3f}\n"
                f"transcript: {transcript if transcript else '(no speech)'}\n"
                f"(keyframe image omitted — text-only model)"
            )
        return "\n\n".join(lines) if lines else "No segments found."

    # --- Answering ------------------------------------------------------

    def answer(self, question: str, video_id: Optional[str] = None,
               max_rounds: int = 3) -> dict:
        """Ask a question; returns {answer, searches: [{query, hits}], usage}."""
        user_text = question if not video_id else (
            f"{question}\n\n(Search is restricted to video {video_id}; "
            f"the question is about that video's timeline.)"
        )
        if self.provider == "claude":
            return self._answer_claude(user_text, video_id, max_rounds)
        return self._answer_openai_compat(user_text, video_id, max_rounds)

    def _answer_claude(self, user_text: str, video_id: Optional[str],
                       max_rounds: int) -> dict:
        messages = [{"role": "user", "content": user_text}]
        searches: list[dict] = []
        usage = {"input_tokens": 0, "output_tokens": 0}
        response = None

        for _ in range(max_rounds):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                thinking={"type": "adaptive"},
                system=SYSTEM_PROMPT,
                tools=[SEARCH_TOOL, TEMPORAL_TOOL],
                messages=messages,
            )
            usage["input_tokens"] += response.usage.input_tokens
            usage["output_tokens"] += response.usage.output_tokens
            if response.stop_reason != "tool_use":
                break

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result, kind = self.run_tool(block.name, dict(block.input), video_id)
                    if kind == "search":
                        searches.append({"query": block.input["query"],
                                         "modality": block.input.get("modality", self.modality),
                                         "hits": result})
                        content = self.hits_to_blocks(result)
                    else:
                        searches.append({"tool": block.name, "args": dict(block.input)})
                        content = self.rows_to_blocks(result)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": content,
                    })
            messages.append({"role": "user", "content": tool_results})

        # Rounds exhausted while still asking for tools -> force a final answer.
        if response is not None and response.stop_reason == "tool_use":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                thinking={"type": "adaptive"},
                system=SYSTEM_PROMPT,
                tools=[SEARCH_TOOL, TEMPORAL_TOOL],
                tool_choice={"type": "none"},
                messages=messages,
            )
            usage["input_tokens"] += response.usage.input_tokens
            usage["output_tokens"] += response.usage.output_tokens

        answer = "".join(b.text for b in response.content if b.type == "text") if response else ""
        return {"answer": answer.strip(), "searches": searches, "usage": usage}

    def _answer_openai_compat(self, user_text: str, video_id: Optional[str],
                              max_rounds: int) -> dict:
        """OpenAI-compatible tool-calling loop (DeepSeek). Text-only tool results."""
        import json

        tools = [{
            "type": "function",
            "function": {"name": t["name"], "description": t["description"],
                         "parameters": t["input_schema"]},
        } for t in (SEARCH_TOOL, TEMPORAL_TOOL)]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        searches: list[dict] = []
        usage = {"input_tokens": 0, "output_tokens": 0}
        msg = None

        for _ in range(max_rounds):
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=min(self.max_tokens, 8192),  # deepseek-chat output cap
                tools=tools,
                messages=messages,
            )
            usage["input_tokens"] += response.usage.prompt_tokens
            usage["output_tokens"] += response.usage.completion_tokens
            msg = response.choices[0].message
            if not msg.tool_calls:
                break

            messages.append(msg)
            for call in msg.tool_calls:
                args = json.loads(call.function.arguments)
                result, kind = self.run_tool(call.function.name, args, video_id)
                if kind == "search":
                    searches.append({"query": args["query"],
                                     "modality": args.get("modality", self.modality),
                                     "hits": result})
                    content = self.hits_to_text(result)
                else:
                    searches.append({"tool": call.function.name, "args": args})
                    content = self.rows_to_text(result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": content,
                })

        # Rounds exhausted while still asking for tools -> force a final answer.
        if msg is not None and msg.tool_calls:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=min(self.max_tokens, 8192),
                tools=tools,
                tool_choice="none",
                messages=messages,
            )
            usage["input_tokens"] += response.usage.prompt_tokens
            usage["output_tokens"] += response.usage.completion_tokens
            msg = response.choices[0].message

        answer = (msg.content or "") if msg else ""
        return {"answer": answer.strip(), "searches": searches, "usage": usage}
