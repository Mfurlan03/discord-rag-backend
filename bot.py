import os
import discord
import httpx
from dotenv import load_dotenv

# Load secrets/config from .env
load_dotenv()

# Discord bot token + backend API endpoint
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
RAG_API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000/api/rag-query")

# Fail fast if the bot token is missing
if not TOKEN:
    raise ValueError("Missing DISCORD_BOT_TOKEN in .env")

# Enable message content intent 
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    # Confirms successful connection to Discord Gateway
    print(f"Logged in as {client.user}")

@client.event
async def on_message(message: discord.Message):
    # Ignore messages sent by the bot itself
    if message.author == client.user:
        return

    # Only handle messages starting with the !ask command
    if not message.content.startswith("!ask"):
        return

    # Extract user query after the command 
    query = message.content[len("!ask"):].strip()
    if not query:
        await message.channel.send("Usage: `!ask your question here`")
        return

    # Show typing indicator while processing
    await message.channel.typing()

    try:
        # Call the FastAPI RAG backend
        async with httpx.AsyncClient(timeout=30) as http:
            resp = await http.post(RAG_API_URL, json={"query": query})
            resp.raise_for_status()
            data = resp.json()

        answer = data.get("answer", "No answer returned.")
        sources = data.get("sources", [])

        # Build sources section for clarity
        source_lines = []
        for s in sources[:3]:
            doc = s.get("docName") or "unknown"
            chunk = s.get("chunkId") or "unknown"
            score = s.get("score")
            score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "?"
            source_lines.append(f"- {doc} ({chunk}, score {score_str})")

        msg = answer
        if source_lines:
            msg += "\n\n**Sources:**\n" + "\n".join(source_lines)

        # Enforce Discord’s 2000 character message limit
        if len(msg) > 1900:
            msg = msg[:1900] + "…"

        await message.channel.send(msg)

    except Exception as e:
        # Show backend or network errors
        await message.channel.send(f"Error calling RAG API: {str(e)[:200]}")

# Start the Discord bot
client.run(TOKEN)
