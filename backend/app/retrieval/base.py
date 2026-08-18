from __future__ import annotations

from abc import ABC, abstractmethod

from app.retrieval.schemas import RetrievalQuery, RetrievalResult


class BaseRetriever(ABC):
    """
    Abstract interface for all retrieval backends.

    Concrete implementations may use FAISS, Chroma, Pinecone,
    Weaviate, Elasticsearch, or any future retrieval engine.
    """

    @abstractmethod
    async def retrieve(
        self,
        query: RetrievalQuery,
    ) -> RetrievalResult:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query: Retrieval request.

        Returns:
            RetrievalResult containing ranked chunks.
        """
        raise NotImplementedError