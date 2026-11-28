"""Text chunking with overlap and deterministic chunk IDs."""

import hashlib
from typing import Any

from app.contracts.ingestion import TextChunk


def _make_chunk_id(document_id: str, chunk_index: int) -> str:
    raw = f"{document_id}:{chunk_index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def split_text_into_chunks(
    text: str,
    document_id: str,
    source: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    extra_metadata: dict[str, Any] | None = None,
) -> list[TextChunk]:
    """Split text into overlapping chunks preserving boundary readability."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size.")

    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: list[TextChunk] = []
    start = 0
    index = 0
    total_len = len(cleaned)
    base_meta = dict(extra_metadata or {})

    while start < total_len:
        end = min(start + chunk_size, total_len)
        if end < total_len:
            # Prefer snapping to the nearest word boundary inside the trailing window.
            space_pos = cleaned.rfind(" ", start + chunk_size // 2, end)
            if space_pos > start:
                end = space_pos

        segment = cleaned[start:end].strip()
        if segment:
            meta = {
                **base_meta,
                "document_id": document_id,
                "source": source,
                "chunk_index": index,
                "text": segment,
            }
            chunks.append(
                TextChunk(
                    chunk_id=_make_chunk_id(document_id, index),
                    text=segment,
                    chunk_index=index,
                    metadata=meta,
                )
            )
            index += 1

        if end >= total_len:
            break
        start = max(end - chunk_overlap, start + 1)

    return chunks
