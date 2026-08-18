from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


class FAISSStore:
    """
    Wrapper around FAISS for vector indexing and similarity search.

    Responsibilities:
    - Maintain a FAISS index.
    - Maintain mapping between FAISS rows and chunk IDs.
    - Add vectors.
    - Search vectors.
    - Save/load index.

    This class intentionally knows nothing about:
    - LLMs
    - Embedding models
    - Chunking
    - Agents
    """

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero")

        self.dimension = dimension

        # Inner Product index
        # Since embeddings are normalized this becomes cosine similarity.
        self.index = faiss.IndexFlatIP(dimension)

        # Maps FAISS row -> chunk ID
        self._id_map: list[str] = []

    def add(
        self,
        ids: list[str],
        vectors: list[list[float]],
    ) -> None:
        """
        Add vectors to the index.

        Args:
            ids:
                Unique chunk IDs.

            vectors:
                Normalized embedding vectors.
        """

        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must have the same length")

        if not vectors:
            return

        array = np.asarray(vectors, dtype=np.float32)

        if array.shape[1] != self.dimension:
            raise ValueError(
                f"Expected vectors of dimension {self.dimension}"
            )

        faiss.normalize_L2(array)

        self.index.add(array)

        self._id_map.extend(ids)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Search for the nearest vectors.

        Returns:
            List[(chunk_id, similarity_score)]
        """

        if self.index.ntotal == 0:
            return []

        query = np.asarray([query_vector], dtype=np.float32)

        if query.shape[1] != self.dimension:
            raise ValueError(
                f"Expected vector dimension {self.dimension}"
            )

        faiss.normalize_L2(query)

        scores, indices = self.index.search(query, top_k)

        results: list[tuple[str, float]] = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            results.append(
                (
                    self._id_map[idx],
                    float(score),
                )
            )

        return results

    def save(self, directory: str | Path) -> None:
        """
        Save FAISS index and ID mapping.
        """

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(directory / "index.faiss"),
        )

        np.save(
            directory / "ids.npy",
            np.asarray(self._id_map),
        )

    def load(self, directory: str | Path) -> None:
        """
        Load FAISS index and ID mapping.
        """

        directory = Path(directory)

        self.index = faiss.read_index(
            str(directory / "index.faiss")
        )

        self._id_map = (
            np.load(
                directory / "ids.npy",
                allow_pickle=True,
            )
            .tolist()
        )

    @property
    def size(self) -> int:
        """
        Number of vectors stored.
        """
        return self.index.ntotal