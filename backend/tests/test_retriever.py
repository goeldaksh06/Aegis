from unittest.mock import Mock

import pytest

from app.retrieval.embeddings import EmbeddingService
from app.retrieval.faiss_store import FAISSStore
from app.retrieval.retriever import Retriever
from app.retrieval.schemas import Chunk, ChunkMetadata, RetrievalQuery


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            source="doc-1",
            page=1,
            chunk_index=0,
            start_offset=0,
            end_offset=len(text),
        ),
    )


@pytest.mark.asyncio
async def test_retrieve_successful_retrieval() -> None:
    embedding_service = Mock(spec=EmbeddingService)
    store = Mock(spec=FAISSStore)

    embedding_service.embed_text.return_value = [0.1, 0.2, 0.3]
    store.search.return_value = [("c1", 0.95), ("c2", 0.70)]

    chunks = {
        "c1": _chunk("c1", "Alpha"),
        "c2": _chunk("c2", "Beta"),
    }

    retriever = Retriever(
        embedding_service=embedding_service,
        store=store,
        chunks_by_id=chunks,
    )

    result = await retriever.retrieve(
        RetrievalQuery(query="find alpha", top_k=2)
    )

    assert result.query == "find alpha"
    assert len(result.retrieved_chunks) == 2
    assert result.retrieved_chunks[0].chunk.id == "c1"
    assert result.retrieved_chunks[0].score == 0.95
    assert result.retrieved_chunks[1].chunk.id == "c2"
    assert result.retrieved_chunks[1].score == 0.70

    embedding_service.embed_text.assert_called_once_with("find alpha")
    store.search.assert_called_once_with(
        query_vector=[0.1, 0.2, 0.3],
        top_k=2,
    )


@pytest.mark.asyncio
async def test_retrieve_empty_index_returns_no_chunks() -> None:
    embedding_service = Mock(spec=EmbeddingService)
    store = Mock(spec=FAISSStore)

    embedding_service.embed_text.return_value = [0.3, 0.2, 0.1]
    store.search.return_value = []

    retriever = Retriever(
        embedding_service=embedding_service,
        store=store,
        chunks_by_id={"c1": _chunk("c1", "Alpha")},
    )

    result = await retriever.retrieve(
        RetrievalQuery(query="nothing here", top_k=5)
    )

    assert result.query == "nothing here"
    assert result.retrieved_chunks == []


@pytest.mark.asyncio
async def test_retrieve_respects_top_k() -> None:
    embedding_service = Mock(spec=EmbeddingService)
    store = Mock(spec=FAISSStore)

    embedding_service.embed_text.return_value = [0.8, 0.1, 0.1]
    store.search.return_value = []

    retriever = Retriever(
        embedding_service=embedding_service,
        store=store,
        chunks_by_id={},
    )

    await retriever.retrieve(
        RetrievalQuery(query="top k test", top_k=7)
    )

    store.search.assert_called_once_with(
        query_vector=[0.8, 0.1, 0.1],
        top_k=7,
    )


@pytest.mark.asyncio
async def test_retrieve_skips_unknown_chunk_ids() -> None:
    embedding_service = Mock(spec=EmbeddingService)
    store = Mock(spec=FAISSStore)

    embedding_service.embed_text.return_value = [0.4, 0.4, 0.2]
    store.search.return_value = [("missing", 0.99)]

    retriever = Retriever(
        embedding_service=embedding_service,
        store=store,
        chunks_by_id={},
    )

    result = await retriever.retrieve(
        RetrievalQuery(query="unknown", top_k=1)
    )

    assert result.retrieved_chunks == []


@pytest.mark.asyncio
async def test_retrieve_no_results_when_store_returns_no_matches() -> None:
    embedding_service = Mock(spec=EmbeddingService)
    store = Mock(spec=FAISSStore)

    embedding_service.embed_text.return_value = [0.9, 0.05, 0.05]
    store.search.return_value = []

    retriever = Retriever(
        embedding_service=embedding_service,
        store=store,
        chunks_by_id={
            "c1": _chunk("c1", "Alpha"),
            "c2": _chunk("c2", "Beta"),
        },
    )

    result = await retriever.retrieve(
        RetrievalQuery(query="no hits", top_k=3)
    )

    assert result.query == "no hits"
    assert result.retrieved_chunks == []
