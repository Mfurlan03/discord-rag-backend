# Discord RAG Bot — Deployment & Run Guide

This project is a **Discord Retrieval-Augmented Generation (RAG) Bot** built with:
- FastAPI (backend API)
- MongoDB Atlas Vector Search (retrieval)
- Sentence Transformers (embeddings)
- DeepSeek (LLM via OpenRouter)
- Discord.py (bot interface)

The bot answers questions in Discord using **your own documents**, not just a generic LLM.

---

## Requirements
- Python **3.11**
- MongoDB Atlas cluster with a **Vector Search index**
- Discord bot token
- OpenRouter (DeepSeek) API key

---

## Environment Setup

Create a `.env` file in the project root (do **not** commit this file):

```env
# MongoDB
MONGODB_URI=your_mongodb_connection_string
MONGODB_DB=discord_rag
MONGODB_COLLECTION=chunks
ATLAS_VECTOR_INDEX=vector_index

# LLM (DeepSeek)
DEEPSEEK_API_KEY=your_openrouter_key
DEEPSEEK_BASE_URL=https://openrouter.ai/api/v1
DEEPSEEK_MODEL=deepseek/deepseek-chat

# RAG settings
MAX_TOKENS=500
MAX_CONTEXT_CHARS=6000

# Discord
DISCORD_BOT_TOKEN=your_discord_bot_token
RAG_API_URL=http://127.0.0.1:8000/api/rag-query
```

---

## Install Dependencies

Create and activate a virtual environment:

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Ingest Documents

1. Add `.txt` files to the `docs/` folder  
2. Run ingestion:

```bash
python ingest.py
```

This:
- Chunks documents
- Creates embeddings
- Stores them in MongoDB Atlas

Run this again anytime documents change.

---

## Run the Backend API

Start the FastAPI server:

```bash
python -m uvicorn app:app
```

Verify it works:
- http://127.0.0.1:8000/healthz
- http://127.0.0.1:8000/docs

---

## Run the Discord Bot

Open a **second terminal**, activate the venv again, then:

```bash
python bot.py
```

You should see:
```
Logged in as RAG Bot
```

---

## Test in Discord

In a server where the bot is added:

```text
!ask what is the AI bootcamp schedule?
```

The bot will:
1. Retrieve relevant document chunks
2. Send context to the LLM
3. Reply with a grounded answer + sources

---

## Optional: Docker (Backend)

Build image:
```bash
docker build -t discord-rag-backend .
```

Run container:
```bash
docker run --env-file .env -p 8000:8000 discord-rag-backend
```

---

## What to Show in a Demo Video

1. `python ingest.py`
2. Backend running (`/healthz`)
3. Discord bot running
4. Ask a real question in Discord
5. Show sources from documents

---

## Notes
- Keep `.env` private
- Allow your IP in MongoDB Atlas
- The bot must be running continuously to stay online

---

If the bot answers questions correctly in Discord, the project is complete.
