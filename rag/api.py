"""FastAPI application factory.

`create_app()` takes its dependencies as arguments, so tests can inject a fake retriever
and LLM, while `build_service_from_settings()` wires the real ones from environment config.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .config import Settings
from .service import RagService

logger = logging.getLogger("discord-rag")


class RagQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    userId: str | None = None
    channelId: str | None = None
    messageId: str | None = None


class SourceItem(BaseModel):
    docName: str | None = None
    chunkId: str | None = None
    score: float | None = None
    snippet: str | None = None


class RagQueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    latencyMs: int
    numHits: int
    topK: int
    status: str
    requestId: str


class FeedbackRequest(BaseModel):
    query: str
    answer: str
    helpful: bool
    comment: str | None = None
    sources: list[SourceItem] = []
    userId: str | None = None
    channelId: str | None = None
    messageId: str | None = None


class InMemoryFeedbackStore:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def insert_one(self, doc: dict) -> None:
        self.items.append(doc)


def create_app(
    service: RagService,
    settings: Settings | None = None,
    feedback_store: Any | None = None,
    health_check: Any | None = None,
) -> FastAPI:
    settings = settings or Settings()
    feedback_store = feedback_store if feedback_store is not None else InMemoryFeedbackStore()
    app = FastAPI(title="Discord RAG Backend", version="0.2.0")

    @app.get("/healthz")
    def healthz():
        backend_ok = True
        if health_check is not None:
            try:
                health_check()
            except Exception:
                backend_ok = False
        return {
            "status": "ok" if backend_ok else "degraded",
            "backendOk": backend_ok,
            "retrieverBackend": settings.retriever_backend,
            "llmConfigured": service.llm is not None,
        }

    @app.get("/config")
    def config():
        return settings.public_dict()

    @app.get("/metrics")
    def metrics():
        return dict(service.metrics)

    @app.post("/api/rag-query", response_model=RagQueryResponse)
    def rag_query(req: RagQueryRequest):
        logger.info("query=%r userId=%s channelId=%s", req.query[:120], req.userId, req.channelId)
        res = service.answer(req.query)
        n = settings.snippet_chars
        return RagQueryResponse(
            answer=res.answer,
            sources=[
                SourceItem(
                    docName=h.doc_name,
                    chunkId=h.chunk_id,
                    score=round(h.score, 4),
                    snippet=h.text[:n] + ("…" if len(h.text) > n else ""),
                )
                for h in res.sources
            ],
            latencyMs=res.latency_ms,
            numHits=len(res.sources),
            topK=service.top_k,
            status=res.status,
            requestId=res.request_id,
        )

    @app.post("/api/feedback")
    def feedback(req: FeedbackRequest):
        service.metrics["feedback_total"] += 1
        doc = req.model_dump()
        doc["ts"] = time.time()
        try:
            feedback_store.insert_one(doc)
            return {"ok": True}
        except Exception:
            service.metrics["errors_total"] += 1
            logger.exception("feedback_insert_failed")
            return {"ok": False}

    return app


def build_service_from_settings(settings: Settings) -> tuple[RagService, Any, Any]:
    """Wire real dependencies. Returns (service, feedback_store, health_check)."""
    from .chunking import chunk_documents, load_documents
    from .embeddings import CachedEmbedder, SentenceTransformerEmbedder
    from .retrieval import AtlasVectorRetriever, build_memory_retriever
    from .service import OpenAICompatibleLLM

    llm = None
    if settings.llm_api_key:
        llm = OpenAICompatibleLLM(settings.llm_api_key, settings.llm_base_url, settings.llm_model, settings.max_tokens)

    needs_embedder = settings.retriever_backend == "atlas" or settings.retrieval_mode != "bm25"
    embedder = None
    if needs_embedder:
        embedder = CachedEmbedder(
            SentenceTransformerEmbedder(settings.embedding_model, settings.hf_cache_dir),
            ttl_seconds=settings.embed_cache_ttl_seconds,
        )

    if settings.retriever_backend == "atlas":
        if not settings.mongodb_uri:
            raise ValueError("RETRIEVER_BACKEND=atlas requires MONGODB_URI")
        from pymongo import MongoClient

        mongo = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=30000, maxPoolSize=50)
        db = mongo[settings.mongodb_db]
        retriever = AtlasVectorRetriever(
            db[settings.mongodb_collection], embedder, settings.atlas_vector_index, settings.num_candidates
        )
        service = RagService(retriever, llm, settings.top_k, settings.min_score, settings.max_context_chars)
        return service, db["feedback"], lambda: mongo.admin.command("ping")

    if settings.retriever_backend != "memory":
        raise ValueError(f"unknown RETRIEVER_BACKEND {settings.retriever_backend!r}")

    chunks = chunk_documents(
        load_documents(settings.docs_dir), settings.chunk_strategy, settings.chunk_size, settings.chunk_overlap
    )
    retriever = build_memory_retriever(chunks, settings.retrieval_mode, embedder)
    # Cosine threshold only makes sense for pure dense scores; BM25/RRF scores live on other scales.
    min_score = settings.min_score if settings.retrieval_mode == "dense" else 0.0
    logger.info("memory backend: %d chunks, mode=%s", len(chunks), settings.retrieval_mode)
    service = RagService(retriever, llm, settings.top_k, min_score, settings.max_context_chars)
    return service, InMemoryFeedbackStore(), None
