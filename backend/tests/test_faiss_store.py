from pathlib import Path

from app.retrieval.faiss_store import FAISSStore


def test_add_vectors():
    store = FAISSStore(dimension=3)

    store.add(
        ids=["a", "b"],
        vectors=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )

    assert store.size == 2


def test_search_returns_best_match():
    store = FAISSStore(dimension=3)

    store.add(
        ids=["football", "weather"],
        vectors=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )

    results = store.search(
        [1.0, 0.0, 0.0],
        top_k=1,
    )

    assert len(results) == 1
    assert results[0][0] == "football"


def test_save_and_load(tmp_path: Path):
    store = FAISSStore(dimension=3)

    store.add(
        ids=["chunk1"],
        vectors=[
            [1.0, 0.0, 0.0],
        ],
    )

    store.save(tmp_path)

    loaded = FAISSStore(dimension=3)

    loaded.load(tmp_path)

    assert loaded.size == 1

    results = loaded.search(
        [1.0, 0.0, 0.0],
        top_k=1,
    )

    assert results[0][0] == "chunk1"