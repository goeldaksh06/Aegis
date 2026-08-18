import pytest

from app.retrieval.chunker import TextChunker


def test_small_document_creates_single_chunk():
    chunker = TextChunker()

    chunks = chunker.split_text(
        text="Hello World",
        source="test.txt",
    )

    assert len(chunks) == 1
    assert chunks[0].text == "Hello World"


def test_large_document_creates_multiple_chunks():
    chunker = TextChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    text = "A" * 500

    chunks = chunker.split_text(
        text=text,
        source="large.txt",
    )

    assert len(chunks) > 1


def test_chunk_metadata_is_preserved():
    chunker = TextChunker()

    chunks = chunker.split_text(
        text="Hello World",
        source="document.pdf",
        page=5,
    )

    metadata = chunks[0].metadata

    assert metadata.source == "document.pdf"
    assert metadata.page == 5
    assert metadata.chunk_index == 0


def test_overlap_exists():
    chunker = TextChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    text = "A" * 300

    chunks = chunker.split_text(
        text=text,
        source="overlap.txt",
    )

    assert chunks[0].text[-20:] == chunks[1].text[:20]