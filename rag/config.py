"""Runtime configuration, read once from environment variables (and .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Settings:
    # Retrieval backend: "atlas" (MongoDB Atlas Vector Search) or "memory" (local, no DB needed)
    retriever_backend: str = "memory"
    # Retrieval strategy for the memory backend: "dense", "bm25" or "hybrid"
    retrieval_mode: str = "hybrid"
    docs_dir: str = "docs"

    # MongoDB
    mongodb_uri: str = ""
    mongodb_db: str = "discord_rag"
    mongodb_collection: str = "chunks"
    atlas_vector_index: str = "vector_index"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_cache_dir: str = "/tmp/hf_cache"

    # LLM (any OpenAI-compatible endpoint, e.g. DeepSeek via OpenRouter)
    llm_api_key: str = ""
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "deepseek/deepseek-chat"
    max_tokens: int = 500

    # RAG tuning
    top_k: int = 5
    num_candidates: int = 100
    min_score: float = 0.25
    snippet_chars: int = 180
    max_context_chars: int = 6000
    embed_cache_ttl_seconds: int = 600
    chunk_strategy: str = "lines"  # "lines" or "fixed"; chosen with `python -m eval.run_eval`
    chunk_size: int = 900
    chunk_overlap: int = 150

    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> Settings:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:  # pragma: no cover - dotenv is optional at runtime
            pass

        return cls(
            retriever_backend=_env("RETRIEVER_BACKEND", "memory").lower(),
            retrieval_mode=_env("RETRIEVAL_MODE", "hybrid").lower(),
            docs_dir=_env("DOCS_DIR", "docs"),
            mongodb_uri=_env("MONGODB_URI"),
            mongodb_db=_env("MONGODB_DB", "discord_rag"),
            mongodb_collection=_env("MONGODB_COLLECTION", "chunks"),
            atlas_vector_index=_env("ATLAS_VECTOR_INDEX", "vector_index"),
            embedding_model=_env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            hf_cache_dir=_env("SENTENCE_TRANSFORMERS_HOME") or _env("HF_HOME") or "/tmp/hf_cache",
            # DEEPSEEK_* names kept for backwards compatibility with existing .env files
            llm_api_key=_env("LLM_API_KEY") or _env("DEEPSEEK_API_KEY"),
            llm_base_url=_env("LLM_BASE_URL") or _env("DEEPSEEK_BASE_URL", "https://openrouter.ai/api/v1"),
            llm_model=_env("LLM_MODEL") or _env("DEEPSEEK_MODEL", "deepseek/deepseek-chat"),
            max_tokens=int(_env("MAX_TOKENS", "500")),
            top_k=int(_env("TOP_K", "5")),
            num_candidates=int(_env("NUM_CANDIDATES", "100")),
            min_score=float(_env("MIN_SCORE", "0.25")),
            snippet_chars=int(_env("SNIPPET_CHARS", "180")),
            max_context_chars=int(_env("MAX_CONTEXT_CHARS", "6000")),
            embed_cache_ttl_seconds=int(_env("EMBED_CACHE_TTL_SECONDS", "600")),
            chunk_strategy=_env("CHUNK_STRATEGY", "lines").lower(),
            chunk_size=int(_env("CHUNK_SIZE", "900")),
            chunk_overlap=int(_env("CHUNK_OVERLAP", "150")),
        )

    def public_dict(self) -> dict:
        """Configuration safe to expose over the API (no secrets)."""
        hidden = {"mongodb_uri", "llm_api_key", "extra"}
        out = {k: v for k, v in self.__dict__.items() if k not in hidden}
        out["llm_configured"] = bool(self.llm_api_key)
        return out
