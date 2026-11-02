"""RAG orchestration: retrieve -> build grounded prompt -> generate -> cite sources."""

from __future__ import annotations

import logging
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

from .retrieval import Hit, Retriever

logger = logging.getLogger("discord-rag")

SYSTEM_PROMPT = (
    "You are a helpful assistant for an AI internship Discord server.\n"
    "Answer using ONLY the provided CONTEXT.\n"
    "If the answer is not in the context, say you don't know.\n"
    "Cite the chunk IDs you used in square brackets, e.g. [intern_faq.txt_chunk_3]."
)

NO_HITS_ANSWER = "I couldn't find anything relevant in the knowledge base for that question."
NO_LLM_ANSWER = "LLM not configured. Showing the most relevant sources instead."
LLM_ERROR_ANSWER = "The language model call failed (credits or rate limits). Try again later."
INTERNAL_ERROR_ANSWER = "Something went wrong while processing your request."


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class OpenAICompatibleLLM:
    """Any OpenAI-compatible chat endpoint (OpenRouter/DeepSeek, OpenAI, vLLM, Ollama...)."""

    def __init__(self, api_key: str, base_url: str, model: str, max_tokens: int = 500, temperature: float = 0.2):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def complete(self, system: str, user: str) -> str:
        from openai import APIConnectionError, APIStatusError

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except (APIStatusError, APIConnectionError) as e:
            raise LLMError(str(e)[:200]) from e
        return resp.choices[0].message.content or ""


class LLMError(RuntimeError):
    """Raised by LLM adapters for provider-side failures (quota, rate limit, 5xx)."""


@dataclass
class RagAnswer:
    answer: str
    sources: list[Hit]
    latency_ms: int
    request_id: str
    status: str = "ok"  # ok | no_hits | no_llm | llm_error | error
    timings_ms: dict = field(default_factory=dict)


def build_context(hits: list[Hit], max_chars: int) -> str:
    """Concatenate chunks, tagged with their IDs, under a hard character budget."""
    parts: list[str] = []
    total = 0
    for h in hits:
        block = f"[{h.chunk_id}] {h.text}".strip()
        if total + len(block) + 2 > max_chars:
            break
        parts.append(block)
        total += len(block) + 2
    return "\n\n".join(parts)


class RagService:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLM | None,
        top_k: int = 5,
        min_score: float = 0.0,
        max_context_chars: int = 6000,
    ):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score
        self.max_context_chars = max_context_chars
        self.metrics: Counter = Counter()

    def answer(self, query: str) -> RagAnswer:
        rid = uuid.uuid4().hex[:8]
        start = time.perf_counter()
        timings: dict[str, int] = {}
        self.metrics["requests_total"] += 1
        query = query.strip()

        def done(answer: str, sources: list[Hit], status: str) -> RagAnswer:
            latency = int((time.perf_counter() - start) * 1000)
            self.metrics[f"requests_{status}_total"] += 1
            self.metrics["latency_ms_total"] += latency
            logger.info("[%s] status=%s hits=%d latencyMs=%d", rid, status, len(sources), latency)
            return RagAnswer(answer, sources, latency, rid, status, timings)

        try:
            t0 = time.perf_counter()
            raw = self.retriever.search(query, self.top_k)
            timings["retrieval"] = int((time.perf_counter() - t0) * 1000)
            hits = [h for h in raw if h.score >= self.min_score]
            self.metrics["retrieval_hits_total"] += len(hits)

            if not hits:
                return done(NO_HITS_ANSWER, [], "no_hits")
            if self.llm is None:
                return done(NO_LLM_ANSWER, hits, "no_llm")

            context = build_context(hits, self.max_context_chars)
            t1 = time.perf_counter()
            text = self.llm.complete(SYSTEM_PROMPT, f"QUESTION:\n{query}\n\nCONTEXT:\n{context}")
            timings["llm"] = int((time.perf_counter() - t1) * 1000)
            return done(text, hits, "ok")
        except LLMError:
            logger.exception("[%s] llm_error", rid)
            return done(LLM_ERROR_ANSWER, [], "llm_error")
        except Exception:
            logger.exception("[%s] unhandled_error", rid)
            return done(INTERNAL_ERROR_ANSWER, [], "error")
