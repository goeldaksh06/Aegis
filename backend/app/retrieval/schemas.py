from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChunkMetadata(BaseModel):
    """
    Metadata describing where a chunk originated.
    """

    model_config = ConfigDict(frozen=True)

    source: str = Field(..., description="Source document name or identifier.")
    page: int | None = Field(
        default=None,
        ge=1,
        description="Page number in the original document.",
    )
    chunk_index: int = Field(
        ...,
        ge=0,
        description="Zero-based index of the chunk in the document.",
    )
    start_offset: int = Field(
        ...,
        ge=0,
        description="Character offset where the chunk starts.",
    )
    end_offset: int = Field(
        ...,
        ge=0,
        description="Character offset where the chunk ends.",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (author, section, tags, etc.).",
    )


class Chunk(BaseModel):
    """
    Represents one chunk of text ready for embedding.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., description="Unique chunk identifier.")
    text: str = Field(..., min_length=1)
    metadata: ChunkMetadata


class RetrievalQuery(BaseModel):
    """
    Input to the retriever.
    """

    model_config = ConfigDict(frozen=True)

    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class RetrievedChunk(BaseModel):
    """
    One retrieved result with similarity score.
    """

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    # Cosine similarity from normalized embedding vectors ranges [-1.0, 1.0], not [0.0, 1.0] —
    # an unrelated query can legitimately produce a small negative score. The old ge=0.0 bound
    # crashed the whole /chat request with a 500 the first time a real off-topic query hit a
    # populated index (e.g. "earthqywke" scored -0.0115 against the seeded documents).
    score: float = Field(..., ge=-1.0, le=1.0)


class RetrievalResult(BaseModel):
    """
    Result returned by the retriever.
    """

    model_config = ConfigDict(frozen=True)

    query: str
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)