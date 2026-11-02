"""Document loading and chunking."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    doc_name: str
    chunk_id: str
    index: int
    text: str


def chunk_fixed(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Original strategy: fixed-size character windows with overlap.

    Simple, but cuts through words and bullet points. Kept as a baseline for the eval.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    step = chunk_size - overlap
    out = []
    for i in range(0, len(text), step):
        piece = text[i : i + chunk_size]
        if piece.strip():
            out.append(piece)
    return out


def chunk_lines(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Structure-aware strategy: pack whole lines into chunks, never splitting a line
    unless it alone exceeds ``chunk_size``. Overlap is carried as whole trailing lines.

    FAQ / onboarding docs are mostly bullets and Q/A pairs, so keeping lines intact keeps
    a question next to its answer.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Hard-wrap pathological long lines so no chunk exceeds the budget.
        while len(line) > chunk_size:
            lines.append(line[:chunk_size])
            line = line[chunk_size - overlap :]
        lines.append(line)

    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in lines:
        if current and size + len(line) + 1 > chunk_size:
            chunks.append("\n".join(current))
            # carry trailing lines (up to `overlap` chars) into the next chunk
            carry: list[str] = []
            carried = 0
            for prev in reversed(current):
                if carried + len(prev) + 1 > overlap:
                    break
                carry.insert(0, prev)
                carried += len(prev) + 1
            current, size = carry, carried
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


CHUNKERS = {"fixed": chunk_fixed, "lines": chunk_lines}


def load_documents(folder: str) -> dict[str, str]:
    docs = {}
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        with open(path, encoding="utf-8-sig") as f:  # utf-8-sig strips the BOM some exports add
            docs[os.path.basename(path)] = f.read()
    return docs


def chunk_documents(
    docs: dict[str, str], strategy: str = "lines", chunk_size: int = 900, overlap: int = 150
) -> list[Chunk]:
    chunker = CHUNKERS[strategy]
    out = []
    for name, text in docs.items():
        for i, piece in enumerate(chunker(text, chunk_size, overlap)):
            out.append(Chunk(doc_name=name, chunk_id=f"{name}_chunk_{i}", index=i, text=piece))
    return out
