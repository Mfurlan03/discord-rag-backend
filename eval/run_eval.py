"""Retrieval evaluation.

For every (chunking strategy x retrieval mode) configuration, run all labeled questions
and report:

* Hit@1 / Hit@3 / Hit@5 - share of questions whose gold passage is in the top-k chunks
* MRR@10               - mean reciprocal rank of the first chunk containing the gold passage

A chunk counts as relevant if it comes from the gold document and contains the gold
span (whitespace-normalized). Labeling by span keeps the labels valid across chunking
strategies.

Usage:
    python -m eval.run_eval                       # all modes (downloads MiniLM for dense/hybrid)
    python -m eval.run_eval --modes bm25          # offline
    python -m eval.run_eval --embedder hashing    # offline dense baseline
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from rag.chunking import Chunk, chunk_documents, load_documents
from rag.embeddings import build_embedder
from rag.retrieval import build_memory_retriever

ROOT = Path(__file__).resolve().parent.parent
_WS = re.compile(r"\s+")


def norm(s: str) -> str:
    return _WS.sub(" ", s).strip().lower()


@dataclass
class Question:
    id: str
    question: str
    doc: str
    span: str


def load_questions(path: Path) -> list[Question]:
    return [Question(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_relevant(chunk_doc: str, chunk_text: str, q: Question) -> bool:
    return chunk_doc == q.doc and norm(q.span) in norm(chunk_text)


def validate(questions: list[Question], chunks: list[Chunk]) -> list[str]:
    """Questions whose gold span is not fully contained in any chunk (unanswerable for this chunking)."""
    return [q.id for q in questions if not any(is_relevant(c.doc_name, c.text, q) for c in chunks)]


def evaluate(retriever, questions: list[Question], k_max: int = 10) -> dict:
    hits = {1: 0, 3: 0, 5: 0}
    rr_total = 0.0
    misses = []
    t0 = time.perf_counter()
    for q in questions:
        results = retriever.search(q.question, k_max)
        rank = next((i for i, h in enumerate(results, 1) if is_relevant(h.doc_name, h.text, q)), None)
        if rank is None:
            misses.append(q.id)
            continue
        rr_total += 1.0 / rank
        for k in hits:
            if rank <= k:
                hits[k] += 1
    n = len(questions)
    return {
        "hit@1": hits[1] / n,
        "hit@3": hits[3] / n,
        "hit@5": hits[5] / n,
        "mrr@10": rr_total / n,
        "ms_per_query": 1000 * (time.perf_counter() - t0) / n,
        "misses@10": misses,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--docs", default=str(ROOT / "docs"))
    p.add_argument("--questions", default=str(ROOT / "eval" / "questions.jsonl"))
    p.add_argument("--modes", nargs="+", default=["bm25", "dense", "hybrid"])
    p.add_argument("--chunking", nargs="+", default=["fixed", "lines"])
    p.add_argument("--embedder", choices=["sentence-transformers", "hashing"], default="sentence-transformers")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--chunk-size", type=int, default=900)
    p.add_argument("--overlap", type=int, default=150)
    p.add_argument("--out", default=None, help="Optional path for a markdown results table")
    args = p.parse_args()

    questions = load_questions(Path(args.questions))
    docs = load_documents(args.docs)
    embedder = None
    if any(m != "bm25" for m in args.modes):
        embedder = build_embedder(args.embedder, args.model)

    rows = []
    for strategy in args.chunking:
        chunks = chunk_documents(docs, strategy, args.chunk_size, args.overlap)
        broken = validate(questions, chunks)
        for mode in args.modes:
            r = evaluate(build_memory_retriever(chunks, mode, embedder), questions)
            label = mode if mode == "bm25" else f"{mode} ({args.embedder})"
            rows.append((strategy, len(chunks), label, r, broken))

    lines = [
        f"Questions: {len(questions)} | chunk_size={args.chunk_size} overlap={args.overlap}",
        "",
        "| Chunking | Chunks | Retriever | Hit@1 | Hit@3 | Hit@5 | MRR@10 | ms/query |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for strategy, n_chunks, label, r, _ in rows:
        lines.append(
            f"| {strategy} | {n_chunks} | {label} | {r['hit@1']:.3f} | {r['hit@3']:.3f} | "
            f"{r['hit@5']:.3f} | {r['mrr@10']:.3f} | {r['ms_per_query']:.1f} |"
        )
    lines.append("")
    for strategy, _, label, r, broken in rows:
        if broken:
            lines.append(f"- `{strategy}`: gold span split across chunks for {broken}")
        if r["misses@10"]:
            lines.append(f"- `{strategy}` / {label}: not retrieved in top 10: {r['misses@10']}")
    md = "\n".join(lines) + "\n"
    print(md)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
