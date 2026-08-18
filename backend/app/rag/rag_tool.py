from __future__ import annotations

from dataclasses import dataclass

from app.models.schemas import ToolRequest, ToolResult, ToolType
from app.retrieval.base import BaseRetriever
from app.retrieval.schemas import RetrievalQuery
from app.tools.base import BaseTool


@dataclass(frozen=True)
class RAGTool(BaseTool):
    """
    Orchestrates retrieval for agent-facing tool calls.

    This layer converts a typed tool request into a retrieval query, delegates
    the actual vector search to the retriever abstraction, and returns a typed
    tool result without exposing FAISS, embeddings, or agent internals.
    """

    retriever: BaseRetriever
    default_top_k: int = 5
    min_relevance_score: float = 0.25
    tool_type: ToolType = ToolType.RAG

    async def run(self, request: ToolRequest) -> ToolResult:
        query = request.input.strip()
        top_k = self._resolve_top_k(request.metadata)

        retrieval = await self.retriever.retrieve(
            RetrievalQuery(query=query, top_k=top_k)
        )

        # FAISS always returns its top-k nearest neighbors even when nothing in the index
        # is actually relevant (e.g. an off-topic query against a small domain-specific
        # index) — without this cutoff, low-relevance chunks get injected into the prompt
        # as if they were real context, and the model either hallucinates a connection or
        # wrongly refuses to answer using its own general knowledge.
        relevant_chunks = [
            retrieved
            for retrieved in retrieval.retrieved_chunks
            if retrieved.score >= self.min_relevance_score
        ]

        chunks_payload: list[dict[str, object]] = []
        lines: list[str] = []

        if not relevant_chunks:
            output = f"No relevant context found for: {query}"
        else:
            for index, retrieved in enumerate(relevant_chunks, start=1):
                chunk = retrieved.chunk
                chunk_line = self._format_chunk_line(index, retrieved.score, chunk.id)
                lines.append(chunk_line)
                lines.append(chunk.text)

                chunks_payload.append(
                    {
                        "id": chunk.id,
                        "text": chunk.text,
                        "score": retrieved.score,
                        "metadata": chunk.metadata.model_dump(),
                    }
                )

            output = "\n\n".join(lines)

        return ToolResult(
            tool_type=self.tool_type,
            output=output,
            metadata={
                "query": query,
                "top_k": top_k,
                "retrieved_count": len(relevant_chunks),
                "chunks": chunks_payload,
            },
        )

    def _resolve_top_k(self, metadata: dict[str, object]) -> int:
        raw_top_k = metadata.get("top_k", self.default_top_k)

        try:
            resolved = int(raw_top_k)
        except (TypeError, ValueError):
            return self.default_top_k

        if resolved < 1:
            return self.default_top_k

        return resolved

    @staticmethod
    def _format_chunk_line(index: int, score: float, chunk_id: str) -> str:
        return f"[{index}] score={score:.4f} chunk_id={chunk_id}"