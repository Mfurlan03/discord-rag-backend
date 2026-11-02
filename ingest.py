"""Ingest ./docs into MongoDB Atlas for RETRIEVER_BACKEND=atlas.

Chunks each document, embeds the chunks, and replaces that document's
previous chunks so re-running is idempotent. The memory backend needs no ingestion:
it chunks and indexes ./docs at startup.
"""

from __future__ import annotations

import argparse
import time

from rag.chunking import chunk_documents, load_documents
from rag.config import Settings
from rag.embeddings import SentenceTransformerEmbedder


def ingest(folder: str, settings: Settings, strategy: str | None = None) -> int:
    strategy = strategy or settings.chunk_strategy
    from pymongo import MongoClient

    if not settings.mongodb_uri:
        raise ValueError("Missing MONGODB_URI in environment/.env")

    coll = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=30000)[settings.mongodb_db][
        settings.mongodb_collection
    ]
    coll.create_index("chunkId", unique=True)
    embedder = SentenceTransformerEmbedder(settings.embedding_model, settings.hf_cache_dir)

    docs = load_documents(folder)
    if not docs:
        print(f"No .txt files found in {folder}/")
        return 0

    total = 0
    for name, text in docs.items():
        chunks = chunk_documents({name: text}, strategy, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            print(f"Skipped {name}: no valid chunks")
            continue
        embeddings = embedder.encode([c.text for c in chunks])
        now = time.time()
        records = [
            {
                "docName": c.doc_name,
                "chunkId": c.chunk_id,
                "chunkIndex": c.index,
                "text": c.text,
                "preview": c.text[:200],
                "embedding": emb.tolist(),
                "createdAt": now,
            }
            for c, emb in zip(chunks, embeddings, strict=True)
        ]
        deleted = coll.delete_many({"docName": name}).deleted_count
        coll.insert_many(records, ordered=False)
        total += len(records)
        print(f"{name}: deleted {deleted}, inserted {len(records)}")

    print(f"Done. Inserted {total} chunks into {settings.mongodb_db}.{settings.mongodb_collection}")
    return total


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--docs", default="docs")
    p.add_argument("--strategy", choices=["lines", "fixed"], default=None, help="defaults to CHUNK_STRATEGY")
    a = p.parse_args()
    ingest(a.docs, Settings.from_env(), a.strategy)
