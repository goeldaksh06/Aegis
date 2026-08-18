from __future__ import annotations

from typing import List
from uuid import uuid4

from app.retrieval.schemas import Chunk, ChunkMetadata


class TextChunker:
    """
    Splits raw text into overlapping chunks suitable for embedding.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> None:

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(
        self,
        text: str,
        source: str,
        page: int | None = None,
    ) -> List[Chunk]:

        if not text.strip():
            return []

        chunks: List[Chunk] = []

        step = self.chunk_size - self.chunk_overlap

        chunk_index = 0

        for start in range(0, len(text), step):

            end = min(start + self.chunk_size, len(text))

            chunk_text = text[start:end]

            metadata = ChunkMetadata(
                source=source,
                page=page,
                chunk_index=chunk_index,
                start_offset=start,
                end_offset=end,
            )

            chunks.append(
                Chunk(
                    id=str(uuid4()),
                    text=chunk_text,
                    metadata=metadata,
                )
            )

            chunk_index += 1

            if end == len(text):
                break

        return chunks