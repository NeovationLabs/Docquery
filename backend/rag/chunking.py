"""Text extraction from files + simple overlapping chunker (no LangChain needed)."""
import os
from pypdf import PdfReader
from .. import config


def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(file_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        return "\n".join(pages)
    elif ext in (".txt", ".md"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _split_into_paragraphs(text: str):
    # normalize whitespace, keep paragraph breaks
    lines = [l.strip() for l in text.splitlines()]
    text = "\n".join(lines)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []
    return paragraphs


def chunk_text(text: str, chunk_size: int = None, overlap: int = None):
    """
    Greedy paragraph-aware chunker with character overlap.
    Returns a list of chunk strings.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    paragraphs = _split_into_paragraphs(text)
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current = f"{current}\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            # start new chunk, carrying overlap from the tail of the previous chunk
            tail = current[-overlap:] if current else ""
            current = f"{tail}\n{para}".strip()
            # if a single paragraph is bigger than chunk_size, hard-split it
            while len(current) > chunk_size:
                chunks.append(current[:chunk_size])
                current = current[chunk_size - overlap :]

    if current:
        chunks.append(current)

    return [c for c in chunks if c.strip()]
