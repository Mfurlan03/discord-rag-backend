import pytest
from fastapi.testclient import TestClient

from rag.api import InMemoryFeedbackStore, create_app
from rag.chunking import Chunk, chunk_documents, chunk_fixed, chunk_lines, load_documents
from rag.config import Settings
from rag.embeddings import CachedEmbedder, HashingEmbedder
from rag.formatting import format_reply
from rag.retrieval import BM25Retriever, DenseRetriever, Hit, HybridRetriever, build_memory_retriever
from rag.service import INTERNAL_ERROR_ANSWER, LLM_ERROR_ANSWER, NO_HITS_ANSWER, LLMError, RagService, build_context

DOCS = {
    "pay.txt": "Q: Is the internship paid?\nA: The internship is unpaid but offers real project experience.",
    "hours.txt": "Q: What is the time commitment?\nA: A minimum of 10 hours per week is required.",
    "visa.txt": "Post visa sponsorship requests in the #visa-sponsorship channel and tag Marla.",
}


@pytest.fixture
def chunks():
    return chunk_documents(DOCS, "lines", 200, 40)


class FakeLLM:
    def __init__(self, reply="stub answer [pay.txt_chunk_0]", exc=None):
        self.reply, self.exc, self.calls = reply, exc, []

    def complete(self, system, user):
        self.calls.append((system, user))
        if self.exc:
            raise self.exc
        return self.reply


# ---------- chunking ----------


def test_fixed_chunks_cover_text_with_overlap():
    text = "abcdefghij" * 30
    pieces = chunk_fixed(text, 100, 20)
    assert all(len(p) <= 100 for p in pieces)
    assert pieces[0][-20:] == pieces[1][:20]


def test_line_chunker_never_splits_short_lines():
    lines = [f"line {i} " + "x" * 40 for i in range(50)]
    pieces = chunk_lines("\n".join(lines), 300, 60)
    for p in pieces:
        assert len(p) <= 300
        for ln in p.split("\n"):
            assert ln in lines


def test_line_chunker_hard_wraps_long_lines():
    pieces = chunk_lines("y" * 1000, 300, 50)
    assert all(len(p) <= 300 for p in pieces)


@pytest.mark.parametrize("fn", [chunk_fixed, chunk_lines])
def test_overlap_must_be_smaller_than_chunk(fn):
    with pytest.raises(ValueError):
        fn("text", 100, 100)


def test_load_documents_strips_bom(tmp_path):
    (tmp_path / "a.txt").write_text("﻿hello", encoding="utf-8")
    assert load_documents(str(tmp_path)) == {"a.txt": "hello"}


def test_chunk_ids_are_unique(chunks):
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


# ---------- retrieval ----------


def test_bm25_ranks_keyword_match_first(chunks):
    hits = BM25Retriever(chunks).search("is the internship paid", 3)
    assert hits[0].doc_name == "pay.txt"


def test_bm25_returns_nothing_for_unknown_terms(chunks):
    assert BM25Retriever(chunks).search("zzzz qqqq", 3) == []


def test_dense_retriever_scores_are_cosine(chunks):
    hits = DenseRetriever(chunks, HashingEmbedder()).search("visa sponsorship channel", 3)
    assert hits[0].doc_name == "visa.txt"
    assert all(-1.0001 <= h.score <= 1.0001 for h in hits)


def test_hybrid_fuses_and_deduplicates(chunks):
    hybrid = HybridRetriever([BM25Retriever(chunks), DenseRetriever(chunks, HashingEmbedder())])
    hits = hybrid.search("hours per week", 3)
    assert hits[0].doc_name == "hours.txt"
    assert len({h.chunk_id for h in hits}) == len(hits)


def test_build_memory_retriever_validates_mode(chunks):
    with pytest.raises(ValueError):
        build_memory_retriever(chunks, "nope", HashingEmbedder())
    with pytest.raises(ValueError):
        build_memory_retriever(chunks, "dense", None)


def test_cached_embedder_hits_cache():
    emb = CachedEmbedder(HashingEmbedder())
    a = emb.encode(["hello world"])
    b = emb.encode(["  hello world "])
    assert (a == b).all() and emb.hits == 1 and emb.misses == 1


def test_retrieval_works_on_real_docs():
    chunks = chunk_documents(load_documents("docs"), "lines")
    hits = BM25Retriever(chunks).search("Is the internship paid or unpaid?", 5)
    assert any("unpaid" in h.text for h in hits)


# ---------- service ----------


def test_context_respects_budget():
    hits = [Hit("d", f"c{i}", "x" * 100, 1.0) for i in range(10)]
    ctx = build_context(hits, 350)
    assert len(ctx) <= 350 and ctx.startswith("[c0]")


def test_service_grounds_prompt_in_retrieved_context(chunks):
    llm = FakeLLM()
    res = RagService(BM25Retriever(chunks), llm, top_k=2).answer("is the internship paid?")
    assert res.status == "ok" and res.answer == llm.reply
    system, user = llm.calls[0]
    assert "ONLY the provided CONTEXT" in system
    assert "[pay.txt_chunk_0]" in user and "unpaid" in user


def test_service_no_hits_skips_llm(chunks):
    llm = FakeLLM()
    res = RagService(BM25Retriever(chunks), llm).answer("zzzz")
    assert res.status == "no_hits" and res.answer == NO_HITS_ANSWER and not llm.calls


def test_service_min_score_filters(chunks):
    res = RagService(BM25Retriever(chunks), FakeLLM(), min_score=1e9).answer("internship paid")
    assert res.status == "no_hits"


def test_service_without_llm_returns_sources(chunks):
    res = RagService(BM25Retriever(chunks), None).answer("internship paid")
    assert res.status == "no_llm" and res.sources


def test_service_handles_llm_error(chunks):
    res = RagService(BM25Retriever(chunks), FakeLLM(exc=LLMError("429"))).answer("internship paid")
    assert res.status == "llm_error" and res.answer == LLM_ERROR_ANSWER


def test_service_handles_unexpected_error():
    class Broken:
        def search(self, q, k):
            raise RuntimeError("db down")

    svc = RagService(Broken(), FakeLLM())
    res = svc.answer("anything")
    assert res.status == "error" and res.answer == INTERNAL_ERROR_ANSWER
    assert svc.metrics["requests_error_total"] == 1


# ---------- API ----------


@pytest.fixture
def client(chunks):
    store = InMemoryFeedbackStore()
    svc = RagService(BM25Retriever(chunks), FakeLLM(), top_k=2)
    c = TestClient(create_app(svc, Settings(snippet_chars=20), store))
    c.store = store
    return c


def test_api_rag_query(client):
    r = client.post("/api/rag-query", json={"query": "is the internship paid?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["numHits"] >= 1 and body["topK"] == 2
    src = body["sources"][0]
    assert src["docName"] == "pay.txt" and len(src["snippet"]) <= 21


@pytest.mark.parametrize("payload", [{}, {"query": ""}, {"query": "x" * 2001}])
def test_api_rejects_invalid_queries(client, payload):
    assert client.post("/api/rag-query", json=payload).status_code == 422


def test_api_feedback_and_metrics(client):
    client.post("/api/rag-query", json={"query": "hours per week"})
    r = client.post("/api/feedback", json={"query": "q", "answer": "a", "helpful": True})
    assert r.json() == {"ok": True} and client.store.items[0]["helpful"] is True
    m = client.get("/metrics").json()
    assert m["requests_total"] == 1 and m["feedback_total"] == 1


def test_api_config_hides_secrets(chunks):
    svc = RagService(BM25Retriever(chunks), None)
    c = TestClient(create_app(svc, Settings(llm_api_key="sk-secret", mongodb_uri="mongodb+srv://u:p@h")))
    body = c.get("/config").json()
    assert "sk-secret" not in str(body) and "mongodb+srv" not in str(body)
    assert body["llm_configured"] is True


def test_api_healthz_reports_degraded_backend(chunks):
    def failing():
        raise ConnectionError

    c = TestClient(create_app(RagService(BM25Retriever(chunks), None), health_check=failing))
    assert c.get("/healthz").json()["status"] == "degraded"


# ---------- discord formatting ----------


def test_format_reply_keeps_sources_when_truncating():
    data = {"answer": "a" * 5000, "sources": [{"docName": "d.txt", "chunkId": "d_0", "score": 0.91}]}
    msg = format_reply(data)
    assert len(msg) <= 1900 and msg.endswith("- d.txt (d_0, score 0.91)")


def test_format_reply_handles_missing_fields():
    assert format_reply({}) == "No answer returned."


def test_chunk_dataclass_is_hashable():
    assert hash(Chunk("d", "d_0", 0, "t"))
