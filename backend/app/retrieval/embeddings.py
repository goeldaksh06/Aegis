from __future__ import annotations

from typing import List

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """
    Service responsible for generating text embeddings.

    This class hides the underlying embedding model from the rest of
    the application. Consumers should never directly import or interact
    with sentence-transformers.

    The embedding model is loaded lazily and shared across all instances
    to avoid expensive repeated initialization.
    """

    _model: SentenceTransformer | None = None

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.model_name = model_name

    def _get_model(self) -> SentenceTransformer:
        """
        Lazily load the embedding model.

        Returns:
            Loaded SentenceTransformer model.
        """
        if EmbeddingService._model is None:
            EmbeddingService._model = SentenceTransformer(self.model_name)

        return EmbeddingService._model

    def embed_text(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text.

        Args:
            text: Input text.

        Returns:
            Embedding vector.
        """
        model = self._get_model()

        embedding = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    def embed_texts(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        model = self._get_model()

        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.tolist()