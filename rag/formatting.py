"""Formatting RAG API responses for Discord."""

from __future__ import annotations

DISCORD_LIMIT = 2000


def format_reply(data: dict, max_sources: int = 3, limit: int = DISCORD_LIMIT - 100) -> str:
    """Render an answer plus a short sources list, truncated to fit a Discord message.

    Sources are kept when truncating: the answer is shortened first, so users can always
    see where the information came from.
    """
    answer = (data.get("answer") or "No answer returned.").strip()
    lines = []
    for s in (data.get("sources") or [])[:max_sources]:
        doc = s.get("docName") or "unknown"
        chunk = s.get("chunkId") or "unknown"
        score = s.get("score")
        score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "?"
        lines.append(f"- {doc} ({chunk}, score {score_str})")
    footer = ("\n\n**Sources:**\n" + "\n".join(lines)) if lines else ""

    budget = limit - len(footer)
    if len(answer) > budget:
        answer = answer[: max(0, budget - 1)].rstrip() + "…"
    return (answer + footer)[:limit]
