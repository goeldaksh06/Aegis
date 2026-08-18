from __future__ import annotations

from dataclasses import dataclass

from app.retrieval.chunker import TextChunker
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.faiss_store import FAISSStore
from app.retrieval.schemas import Chunk


@dataclass(frozen=True)
class DocumentIndexer:
    """
    Orchestrates document ingestion into the vector store.

    This layer owns the write path for retrieval data:
    text -> chunks -> embeddings -> FAISS index.
    It intentionally does not know about agents, LLMs, or API transport.
    """

    chunker: TextChunker
    embedding_service: EmbeddingService
    store: FAISSStore

    def index_text(
        self,
        text: str,
        source: str,
        page: int | None = None,
    ) -> list[Chunk]:
        chunks = self.chunker.split_text(
            text=text,
            source=source,
            page=page,
        )

        if not chunks:
            return []

        vectors = self.embedding_service.embed_texts(
            [chunk.text for chunk in chunks]
        )

        if len(vectors) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")

        self.store.add(
            ids=[chunk.id for chunk in chunks],
            vectors=vectors,
        )

        return chunks