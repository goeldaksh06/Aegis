from __future__ import annotations

from collections.abc import Mapping

from app.retrieval.base import BaseRetriever
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.faiss_store import FAISSStore
from app.retrieval.schemas import (
    Chunk,
    RetrievedChunk,
    RetrievalQuery,
    RetrievalResult,
)


class Retriever(BaseRetriever):
    """
    Orchestrates embedding + vector search and returns typed retrieval results.

    This layer intentionally depends only on retrieval abstractions/components
    and has no knowledge of agents, LLM providers, or API transport.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        store: FAISSStore,
        chunks_by_id: Mapping[str, Chunk] | None = None,
    ) -> None:
        self._embedding_service = embedding_service
        self._store = store
        self._chunks_by_id: dict[str, Chunk] = dict(chunks_by_id or {})

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """
        Register chunks so retrieval results can resolve IDs to typed chunks.
        """
        for chunk in chunks:
            self._chunks_by_id[chunk.id] = chunk

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        query_vector = self._embedding_service.embed_text(query.query)

        ranked_ids = self._store.search(
            query_vector=query_vector,
            top_k=query.top_k,
        )

        retrieved: list[RetrievedChunk] = []

        for chunk_id, score in ranked_ids:
            chunk = self._chunks_by_id.get(chunk_id)

            if chunk is None:
                continue

            retrieved.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                )
            )

        return RetrievalResult(
            query=query.query,
            retrieved_chunks=retrieved,
        )
