from unittest.mock import Mock

from app.retrieval.chunker import TextChunker
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.faiss_store import FAISSStore
from app.retrieval.indexer import DocumentIndexer
from app.retrieval.schemas import Chunk, ChunkMetadata


def _chunk(chunk_id: str, text: str, chunk_index: int) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            source="doc.txt",
            page=1,
            chunk_index=chunk_index,
            start_offset=chunk_index * 10,
            end_offset=chunk_index * 10 + len(text),
        ),
    )


def test_index_text_chunks_embeddings_and_store():
    chunker = Mock(spec=TextChunker)
    embedding_service = Mock(spec=EmbeddingService)
    store = Mock(spec=FAISSStore)

    chunks = [
        _chunk("c1", "Alpha context", 0),
        _chunk("c2", "Beta context", 1),
    ]

    chunker.split_text.return_value = chunks
    embedding_service.embed_texts.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    indexer = DocumentIndexer(
        chunker=chunker,
        embedding_service=embedding_service,
        store=store,
    )

    result = indexer.index_text(
        text="Alpha Beta",
        source="doc.txt",
        page=1,
    )

    assert result == chunks
    chunker.split_text.assert_called_once_with(
        text="Alpha Beta",
        source="doc.txt",
        page=1,
    )
    embedding_service.embed_texts.assert_called_once_with(
        ["Alpha context", "Beta context"]
    )
    store.add.assert_called_once_with(
        ids=["c1", "c2"],
        vectors=[
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
    )


def test_index_text_empty_input_skips_embedding_and_store():
    chunker = Mock(spec=TextChunker)
    embedding_service = Mock(spec=EmbeddingService)
    store = Mock(spec=FAISSStore)

    chunker.split_text.return_value = []

    indexer = DocumentIndexer(
        chunker=chunker,
        embedding_service=embedding_service,
        store=store,
    )

    result = indexer.index_text(
        text="   ",
        source="doc.txt",
    )

    assert result == []
    chunker.split_text.assert_called_once_with(
        text="   ",
        source="doc.txt",
        page=None,
    )
    embedding_service.embed_texts.assert_not_called()
    store.add.assert_not_called()


def test_index_text_raises_when_embedding_count_mismatches_chunks():
    chunker = Mock(spec=TextChunker)
    embedding_service = Mock(spec=EmbeddingService)
    store = Mock(spec=FAISSStore)

    chunker.split_text.return_value = [
        _chunk("c1", "Alpha context", 0),
        _chunk("c2", "Beta context", 1),
    ]
    embedding_service.embed_texts.return_value = [[0.1, 0.2, 0.3]]

    indexer = DocumentIndexer(
        chunker=chunker,
        embedding_service=embedding_service,
        store=store,
    )

    try:
        indexer.index_text(
            text="Alpha Beta",
            source="doc.txt",
        )
    except ValueError as exc:
        assert "Number of embeddings must match number of chunks" in str(exc)
    else:
        raise AssertionError("Expected ValueError to be raised")

    store.add.assert_not_called()