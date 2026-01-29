import os
import time
import uuid
import logging
from typing import List, Dict, Any
from collections import Counter

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from cachetools import TTLCache
from openai import APIStatusError

# Load environment variables 
load_dotenv()

# Logging helps trace requests during debugging and demos
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("discord-rag")

# MongoDB configuration
# Fail fast if the DB connection string is missing
MONGODB_URI = os.getenv("MONGODB_URI", "").strip()
if not MONGODB_URI:
    raise ValueError("Missing MONGODB_URI. Add it to your .env file.")

DB_NAME = os.getenv("MONGODB_DB", "discord_rag").strip()
COLL_NAME = os.getenv("MONGODB_COLLECTION", "chunks").strip()
VECTOR_INDEX = os.getenv("ATLAS_VECTOR_INDEX", "vector_index").strip()

# Use timeouts for reliability
mongo = MongoClient(
    MONGODB_URI,
    serverSelectionTimeoutMS=30000,
    connectTimeoutMS=30000,
    socketTimeoutMS=30000,
    maxPoolSize=50,
)
coll = mongo[DB_NAME][COLL_NAME]
feedback_coll = mongo[DB_NAME]["feedback"]

# DeepSeek accessed via OpenRouter-compatible API
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://openrouter.ai/api/v1").strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek/deepseek-chat").strip()
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "500"))

# RAG tuning parameters
# These control retrieval quality, cost, and latency
TOP_K = int(os.getenv("TOP_K", "5"))
NUM_CANDIDATES = int(os.getenv("NUM_CANDIDATES", "100"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.25"))
SNIPPET_CHARS = int(os.getenv("SNIPPET_CHARS", "180"))
EMBED_CACHE_TTL_SECONDS = int(os.getenv("EMBED_CACHE_TTL_SECONDS", "600"))

# Prevent oversized prompts being sent to the LLM
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))

# Ensures model files are cached in a writable directory
HF_CACHE_DIR = (
    os.getenv("SENTENCE_TRANSFORMERS_HOME")
    or os.getenv("HF_HOME")
    or "/tmp/hf_cache"
)
os.makedirs(HF_CACHE_DIR, exist_ok=True)

# Lightweight 384-dim embedding model matching Atlas vector index
embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=HF_CACHE_DIR,
)

# Cache query embeddings to reduce repeated compute
embed_cache = TTLCache(maxsize=2048, ttl=EMBED_CACHE_TTL_SECONDS)

# LLM client - Only initialize if a key is present 
llm_client = None
if DEEPSEEK_API_KEY:
    llm_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# App + simple metrics
app = FastAPI(title="Discord RAG Backend")
metrics = Counter()

# Request / response models
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
    sources: List[SourceItem]
    latencyMs: int
    numHits: int
    topK: int


class FeedbackRequest(BaseModel):
    query: str
    answer: str
    helpful: bool
    comment: str | None = None
    sources: List[SourceItem] = []
    userId: str | None = None
    channelId: str | None = None
    messageId: str | None = None

# Helper functions
def embed_query(text: str) -> List[float]:
    # Cache embeddings to avoid recomputation for repeat queries
    key = text.strip()
    if key in embed_cache:
        metrics["embed_cache_hits_total"] += 1
        return embed_cache[key]

    metrics["embed_cache_misses_total"] += 1
    vec = embedder.encode([key], normalize_embeddings=True)[0].tolist()
    embed_cache[key] = vec
    return vec


def build_context(hits: List[Dict[str, Any]], max_chars: int) -> str:
    # Build context while enforcing a hard character limit
    parts = []
    total = 0
    for h in hits:
        chunk = f"[{h.get('chunkId')}] {h.get('text','')}".strip()
        if not chunk:
            continue
        if total + len(chunk) + 2 > max_chars:
            break
        parts.append(chunk)
        total += len(chunk) + 2
    return "\n\n".join(parts)

# Endpoints
@app.get("/healthz")
def healthz():
    # Dependency check for monitoring/demo purposes
    mongo_ok = True
    try:
        mongo.admin.command("ping")
    except Exception:
        mongo_ok = False

    return {
        "status": "ok" if mongo_ok else "degraded",
        "mongoOk": mongo_ok,
        "llmConfigured": llm_client is not None,
    }


@app.get("/config")
def config():
    # Runtime configuration info for demo UIs
    return {
        "db": DB_NAME,
        "collection": COLL_NAME,
        "vectorIndex": VECTOR_INDEX,
        "topK": TOP_K,
        "numCandidates": NUM_CANDIDATES,
        "minScore": MIN_SCORE,
        "snippetChars": SNIPPET_CHARS,
        "embedCacheTtlSeconds": EMBED_CACHE_TTL_SECONDS,
        "maxContextChars": MAX_CONTEXT_CHARS,
        "model": DEEPSEEK_MODEL,
        "baseUrl": DEEPSEEK_BASE_URL,
        "maxTokens": MAX_TOKENS,
        "llmConfigured": llm_client is not None,
    }


@app.get("/metrics")
def get_metrics():
    # Simple counters for observability
    return dict(metrics)


@app.post("/api/rag-query", response_model=RagQueryResponse)
def rag_query(req: RagQueryRequest):
    request_id = str(uuid.uuid4())[:8]
    start = time.time()
    metrics["requests_total"] += 1

    query = req.query.strip()
    logger.info(
        f"[{request_id}] query='{query[:120]}' userId={req.userId} channelId={req.channelId} messageId={req.messageId}"
    )

    try:
        # Embed user query
        t0 = time.time()
        qvec = embed_query(query)
        metrics["embed_total"] += 1
        metrics["embed_latency_ms_total"] += int((time.time() - t0) * 1000)

        # Vector search in MongoDB Atlas
        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX,
                    "path": "embedding",
                    "queryVector": qvec,
                    "numCandidates": NUM_CANDIDATES,
                    "limit": TOP_K,
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "docName": 1,
                    "chunkId": 1,
                    "text": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        t1 = time.time()
        raw_hits: List[Dict[str, Any]] = list(coll.aggregate(pipeline, maxTimeMS=8000))
        metrics["retrieval_total"] += 1
        metrics["vector_search_latency_ms_total"] += int((time.time() - t1) * 1000)

        # Filter low-quality retrievals
        hits = [h for h in raw_hits if float(h.get("score", 0)) >= MIN_SCORE]
        metrics["retrieval_hits_total"] += len(hits)

        logger.info(f"[{request_id}] retrieved={len(raw_hits)} kept={len(hits)} minScore={MIN_SCORE}")

        if not hits:
            metrics["requests_no_hits_total"] += 1
            latency = int((time.time() - start) * 1000)
            return RagQueryResponse(
                answer="I couldn’t find anything relevant in the knowledge base for that question.",
                sources=[],
                latencyMs=latency,
                numHits=0,
                topK=TOP_K,
            )

        # Prepare source snippets 
        sources: List[SourceItem] = []
        for h in hits:
            txt = h.get("text") or ""
            snippet = txt[:SNIPPET_CHARS] + ("…" if len(txt) > SNIPPET_CHARS else "")
            sources.append(
                SourceItem(
                    docName=h.get("docName"),
                    chunkId=h.get("chunkId"),
                    score=round(float(h.get("score", 0.0)), 4),
                    snippet=snippet,
                )
            )

        # Limit context size to control cost and latency
        context = build_context(hits, MAX_CONTEXT_CHARS)

        system_prompt = (
            "You are a helpful assistant for an AI internship Discord server.\n"
            "Use ONLY the provided CONTEXT.\n"
            "If the answer is not present, say you don't know.\n"
            "Cite chunk IDs when relevant."
        )
        user_prompt = f"QUESTION:\n{query}\n\nCONTEXT:\n{context}"

        if llm_client is None:
            latency = int((time.time() - start) * 1000)
            return RagQueryResponse(
                answer="LLM not configured.",
                sources=sources,
                latencyMs=latency,
                numHits=len(hits),
                topK=TOP_K,
            )

        # Call LLM
        metrics["llm_calls_total"] += 1
        t2 = time.time()

        completion = llm_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=MAX_TOKENS,
        )

        metrics["llm_latency_ms_total"] += int((time.time() - t2) * 1000)

        answer = completion.choices[0].message.content or ""
        latency = int((time.time() - start) * 1000)
        metrics["latency_ms_total"] += latency
        metrics["requests_success_total"] += 1

        logger.info(f"[{request_id}] done latencyMs={latency}")
        return RagQueryResponse(
            answer=answer,
            sources=sources,
            latencyMs=latency,
            numHits=len(hits),
            topK=TOP_K,
        )

    except APIStatusError as e:
        # Handle LLM credit / rate-limit errors
        metrics["errors_total"] += 1
        latency = int((time.time() - start) * 1000)
        logger.error(
            f"[{request_id}] LLM/API error status={getattr(e, 'status_code', None)} msg={str(e)[:200]}"
        )
        return RagQueryResponse(
            answer="LLM call failed (credits/limits). Try again later or lower MAX_TOKENS.",
            sources=[],
            latencyMs=latency,
            numHits=0,
            topK=TOP_K,
        )

    except Exception:
        # Catch-all to prevent crashing the API
        metrics["errors_total"] += 1
        latency = int((time.time() - start) * 1000)
        logger.exception(f"[{request_id}] unhandled_error")
        return RagQueryResponse(
            answer="Something went wrong while processing your request.",
            sources=[],
            latencyMs=latency,
            numHits=0,
            topK=TOP_K,
        )


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    # Store user feedback for future evaluation/improvement
    metrics["feedback_total"] += 1
    doc = {
        "ts": time.time(),
        "query": req.query,
        "answer": req.answer,
        "helpful": req.helpful,
        "comment": req.comment,
        "sources": [s.model_dump() for s in req.sources],
        "userId": req.userId,
        "channelId": req.channelId,
        "messageId": req.messageId,
    }

    try:
        feedback_coll.insert_one(doc)
        return {"ok": True}
    except Exception:
        metrics["errors_total"] += 1
        logger.exception("feedback_insert_failed")
        return {"ok": False}
