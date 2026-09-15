from unittest.mock import MagicMock, patch

from app.retrieval.embeddings import EmbeddingService


@patch("app.retrieval.embeddings.SentenceTransformer")
def test_embed_text_returns_vector(mock_model):
    fake_model = MagicMock()

    fake_model.encode.return_value.tolist.return_value = [
        0.1,
        0.2,
        0.3,
    ]

    mock_model.return_value = fake_model

    EmbeddingService._model = None

    service = EmbeddingService()

    vector = service.embed_text("Hello")

    assert vector == [0.1, 0.2, 0.3]


@patch("app.retrieval.embeddings.SentenceTransformer")
def test_embed_texts_returns_vectors(mock_model):
    fake_model = MagicMock()

    fake_model.encode.return_value.tolist.return_value = [
        [0.1, 0.2],
        [0.3, 0.4],
    ]

    mock_model.return_value = fake_model

    EmbeddingService._model = None

    service = EmbeddingService()

    vectors = service.embed_texts(
        [
            "hello",
            "world",
        ]
    )

    assert len(vectors) == 2


@patch("app.retrieval.embeddings.SentenceTransformer")
def test_model_loaded_only_once(mock_model):
    fake_model = MagicMock()

    fake_model.encode.return_value.tolist.return_value = [0.1]

    mock_model.return_value = fake_model

    EmbeddingService._model = None

    service = EmbeddingService()

    service.embed_text("first")
    service.embed_text("second")
    service.embed_text("third")

    assert mock_model.call_count == 1