"""W5: LLM query decomposition for CLIP-based moment retrieval.

NExT-QA questions are causal/temporal ("why did the boy ... and move to the
sofa"), but CLIP matches literal scene descriptions. The decomposer rewrites a
question into 2-4 short, concrete sub-queries describing moments that would
contain the answer evidence; retrieval runs each sub-query and rank-fuses.

Decompositions are cached to `artifacts/decompositions/<name>.json`
(question -> [sub-queries]) so the 3358-question eval calls the LLM once, ever.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

DECOMPOSE_PROMPT = """You convert a video question into search queries for a CLIP-based \
video moment retrieval system. CLIP matches short, literal descriptions of what is visible \
on screen (objects, people, actions, scenes) — it does not understand causal or abstract \
language.

Given the question, output JSON: {"queries": ["...", "..."]} with 2 to 4 short declarative \
scene descriptions (max ~12 words each) of moments that would contain evidence for the \
answer. Rules:
- Describe concrete visible content only (who/what/where/action), present tense.
- If the question implies multiple events (before/after/when), give one query per event.
- No question words, no "why/how", no explanations — just scene captions.

Question: {question}"""


def _extract_json(text: str) -> dict:
    """Parse a JSON object out of an LLM reply (tolerates code fences)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    return json.loads(m.group(0))


class QueryDecomposer:
    """LLM-backed question -> sub-queries, with a persistent JSON cache."""

    def __init__(self, cfg, cache_name: str = "default"):
        self.provider = cfg.get_path("decompose.provider", cfg.get_path("agent.provider", "deepseek"))
        default_model = "claude-opus-4-8" if self.provider == "claude" else "deepseek-chat"
        self.model = cfg.get_path("decompose.model", default_model)
        self.n_max = int(cfg.get_path("decompose.max_queries", 4))
        self.include_original = bool(cfg.get_path("decompose.include_original", True))
        self.workers = int(cfg.get_path("decompose.workers", 16))

        root = cfg.get_path("paths.artifacts", "artifacts")
        self.cache_path = os.path.join(root, "decompositions", f"{cache_name}.json")
        self._cache: Optional[dict] = None
        self._client = None

    # --- cache ----------------------------------------------------------

    @property
    def cache(self) -> dict:
        if self._cache is None:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, encoding="utf-8") as f:
                    self._cache = json.load(f)
            else:
                self._cache = {}
        return self._cache

    def _save_cache(self) -> None:
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=0)
        os.replace(tmp, self.cache_path)

    # --- LLM call ---------------------------------------------------------

    @property
    def client(self):
        if self._client is None:
            if self.provider == "claude":
                import anthropic
                self._client = anthropic.Anthropic()
            else:
                from openai import OpenAI
                self._client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                                      base_url="https://api.deepseek.com")
        return self._client

    def _call_llm(self, question: str) -> list[str]:
        prompt = DECOMPOSE_PROMPT.replace("{question}", question)
        if self.provider == "claude":
            resp = self.client.messages.create(
                model=self.model, max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in resp.content if b.type == "text")
        else:
            resp = self.client.chat.completions.create(
                model=self.model, max_tokens=512,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content or ""
        queries = _extract_json(text).get("queries", [])
        queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        return queries[: self.n_max]

    # --- public API -------------------------------------------------------

    def decompose(self, question: str) -> list[str]:
        """Sub-queries for one question (cached). Falls back to [question] on failure.
        The original question is appended when `include_original` (robustness: the
        decomposition should augment retrieval, never replace it)."""
        if question not in self.cache:
            try:
                self.cache[question] = self._call_llm(question)
            except Exception as e:
                print(f"[decompose] LLM failed for {question[:60]!r}: {e}")
                self.cache[question] = []
            self._save_cache()
        queries = list(self.cache[question])
        if self.include_original or not queries:
            queries.append(question)
        return queries

    def decompose_batch(self, questions: list[str], save_every: int = 200) -> dict[str, list[str]]:
        """Concurrently decompose all uncached questions; returns question -> sub-queries
        (cache content only, without the appended original)."""
        todo = [q for q in dict.fromkeys(questions) if q not in self.cache]
        if todo:
            print(f"[decompose] {len(todo)} uncached questions "
                  f"({len(questions) - len(todo)} cached) via {self.provider}:{self.model}, "
                  f"{self.workers} workers")
            done = 0
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = {pool.submit(self._call_llm, q): q for q in todo}
                for fut in as_completed(futures):
                    q = futures[fut]
                    try:
                        self.cache[q] = fut.result()
                    except Exception as e:
                        print(f"[decompose] failed {q[:60]!r}: {e}")
                        self.cache[q] = []
                    done += 1
                    if done % save_every == 0:
                        self._save_cache()
                        print(f"[decompose] {done}/{len(todo)}")
            self._save_cache()
        return {q: self.cache.get(q, []) for q in questions}
