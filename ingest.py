import os, glob, time
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

# MongoDB config 
# Fail fast if the database connection string is missing
MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    raise ValueError("Missing MONGODB_URI in environment/.env")

DB_NAME = os.getenv("MONGODB_DB", "discord_rag")
COLL_NAME = os.getenv("MONGODB_COLLECTION", "chunks")

# Use explicit timeouts to avoid hanging if Atlas is unreachable
client = MongoClient(
    MONGODB_URI,
    serverSelectionTimeoutMS=30000,
    connectTimeoutMS=30000,
    socketTimeoutMS=30000,
)
coll = client[DB_NAME][COLL_NAME]

# Ensure each chunk is uniquely identifiable
try:
    coll.create_index("chunkId", unique=True)
except Exception:
    pass

# Embedding model cache
# Use a writable cache directory 
HF_CACHE_DIR = (
    os.getenv("SENTENCE_TRANSFORMERS_HOME")
    or os.getenv("HF_HOME")
    or "/tmp/hf_cache"
)
os.makedirs(HF_CACHE_DIR, exist_ok=True)

# Embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=HF_CACHE_DIR
)

# Chunking 
# Split text into overlapping chunks to preserve context across boundaries
def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150):
    chunks = []
    i = 0
    step = max(1, chunk_size - overlap)

    while i < len(text):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        i += step

    return chunks

# Ingestion pipeline 
# Reads docs, chunks them, embeds them, and stores them in MongoDB
def ingest(folder: str = "docs"):
    files = glob.glob(os.path.join(folder, "*.txt"))
    if not files:
        print(f"No .txt files found in {folder}/")
        return

    total_inserted = 0

    for path in files:
        doc_name = os.path.basename(path)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        if not chunks:
            print(f"Skipped {doc_name}: no valid chunks")
            continue

        # Remove old chunks for this document before re-ingesting
        deleted = coll.delete_many({"docName": doc_name}).deleted_count

        embeddings = model.encode(
            chunks,
            normalize_embeddings=True
        ).tolist()

        now = time.time()
        records = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            records.append({
                "docName": doc_name,
                "chunkId": f"{doc_name}_chunk_{i}",
                "chunkIndex": i,
                "text": chunk,
                "preview": chunk[:200],
                "embedding": emb,
                "createdAt": now,
            })

        if records:
            coll.insert_many(records, ordered=False)
            total_inserted += len(records)
            print(f"{doc_name}: deleted {deleted}, inserted {len(records)}")

    print(f"Done. Inserted {total_inserted} chunks into {DB_NAME}.{COLL_NAME}")

if __name__ == "__main__":
    ingest()
