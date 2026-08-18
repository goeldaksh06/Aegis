from unittest.mock import AsyncMock, Mock

import pytest

from app.models.schemas import ToolRequest, ToolType
from app.retrieval.base import BaseRetriever
from app.retrieval.schemas import (
    Chunk,
    ChunkMetadata,
    RetrievedChunk,
    RetrievalResult,
)
from app.rag.rag_tool import RAGTool


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
async def test_rag_tool_successful_retrieval() -> None:
    retriever = Mock(spec=BaseRetriever)
    retriever.retrieve = AsyncMock(
        return_value=RetrievalResult(
            query="alpha query",
            retrieved_chunks=[
                RetrievedChunk(chunk=_chunk("c1", "Alpha context"), score=0.98),
                RetrievedChunk(chunk=_chunk("c2", "Beta context"), score=0.76),
            ],
        )
    )

    tool = RAGTool(retriever=retriever)

    result = await tool.run(
        ToolRequest(
            tool_type=ToolType.RAG,
            input="alpha query",
        )
    )

    assert result.tool_type == ToolType.RAG
    assert result.success is True
    assert "[1] score=0.9800 chunk_id=c1" in result.output
    assert "Alpha context" in result.output
    assert "[2] score=0.7600 chunk_id=c2" in result.output
    assert result.metadata["query"] == "alpha query"
    assert result.metadata["top_k"] == 5
    assert result.metadata["retrieved_count"] == 2

    retriever.retrieve.assert_awaited_once()
    called_query = retriever.retrieve.call_args.args[0]
    assert called_query.query == "alpha query"
    assert called_query.top_k == 5


@pytest.mark.asyncio
async def test_rag_tool_empty_index_returns_no_context() -> None:
    retriever = Mock(spec=BaseRetriever)
    retriever.retrieve = AsyncMock(
        return_value=RetrievalResult(query="missing", retrieved_chunks=[])
    )

    tool = RAGTool(retriever=retriever)

    result = await tool.run(
        ToolRequest(
            tool_type=ToolType.RAG,
            input="missing",
        )
    )

    assert result.success is True
    assert result.output == "No relevant context found for: missing"
    assert result.metadata["retrieved_count"] == 0
    assert result.metadata["chunks"] == []


@pytest.mark.asyncio
async def test_rag_tool_filters_out_low_relevance_chunks() -> None:
    retriever = Mock(spec=BaseRetriever)
    retriever.retrieve = AsyncMock(
        return_value=RetrievalResult(
            query="off-topic query",
            retrieved_chunks=[
                RetrievedChunk(chunk=_chunk("c1", "Barely related"), score=0.12),
                RetrievedChunk(chunk=_chunk("c2", "Also unrelated"), score=0.09),
            ],
        )
    )

    tool = RAGTool(retriever=retriever)

    result = await tool.run(
        ToolRequest(tool_type=ToolType.RAG, input="off-topic query")
    )

    assert result.output == "No relevant context found for: off-topic query"
    assert result.metadata["retrieved_count"] == 0
    assert result.metadata["chunks"] == []


@pytest.mark.asyncio
async def test_rag_tool_keeps_only_chunks_above_relevance_threshold() -> None:
    retriever = Mock(spec=BaseRetriever)
    retriever.retrieve = AsyncMock(
        return_value=RetrievalResult(
            query="mixed relevance",
            retrieved_chunks=[
                RetrievedChunk(chunk=_chunk("c1", "Strongly related"), score=0.6),
                RetrievedChunk(chunk=_chunk("c2", "Weakly related"), score=0.1),
            ],
        )
    )

    tool = RAGTool(retriever=retriever)

    result = await tool.run(
        ToolRequest(tool_type=ToolType.RAG, input="mixed relevance")
    )

    assert result.metadata["retrieved_count"] == 1
    assert "Strongly related" in result.output
    assert "Weakly related" not in result.output


@pytest.mark.asyncio
async def test_rag_tool_respects_top_k_from_metadata() -> None:
    retriever = Mock(spec=BaseRetriever)
    retriever.retrieve = AsyncMock(
        return_value=RetrievalResult(query="q", retrieved_chunks=[])
    )

    tool = RAGTool(retriever=retriever)

    await tool.run(
        ToolRequest(
            tool_type=ToolType.RAG,
            input="q",
            metadata={"top_k": 3},
        )
    )

    retriever.retrieve.assert_awaited_once()
    called_query = retriever.retrieve.call_args.args[0]
    assert called_query.top_k == 3


@pytest.mark.asyncio
async def test_rag_tool_defaults_top_k_when_metadata_is_invalid() -> None:
    retriever = Mock(spec=BaseRetriever)
    retriever.retrieve = AsyncMock(
        return_value=RetrievalResult(query="q", retrieved_chunks=[])
    )

    tool = RAGTool(retriever=retriever, default_top_k=7)

    await tool.run(
        ToolRequest(
            tool_type=ToolType.RAG,
            input="q",
            metadata={"top_k": "invalid"},
        )
    )

    retriever.retrieve.assert_awaited_once()
    called_query = retriever.retrieve.call_args.args[0]
    assert called_query.top_k == 7
