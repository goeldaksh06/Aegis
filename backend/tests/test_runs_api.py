import asyncio

from fastapi.testclient import TestClient

from app.database import db
from app.main import app


def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test_runs.db")
    monkeypatch.setattr(db, "DATABASE_URL", f"sqlite+aiosqlite:///{db.DB_PATH}")
    db.get_engine.cache_clear()
    db.get_sessionmaker.cache_clear()
    db._ready = False


def _register(client: TestClient, email: str) -> str:
    response = client.post("/auth/register", json={"email": email, "password": "correct-horse-1"})
    assert response.status_code == 201, response.text
    return response.json()["access_token"]


def test_runs_endpoint_requires_authentication(tmp_path, monkeypatch):
    _isolated_db(tmp_path, monkeypatch)
    client = TestClient(app)

    response = client.get("/runs")

    assert response.status_code == 401


def test_runs_endpoint_returns_only_the_authenticated_users_runs(tmp_path, monkeypatch):
    _isolated_db(tmp_path, monkeypatch)
    client = TestClient(app)

    token_a = _register(client, "alice@example.com")
    token_b = _register(client, "bob@example.com")

    user_a_id = client.get("/auth/me", headers={"Authorization": f"Bearer {token_a}"}).json()["id"]
    user_b_id = client.get("/auth/me", headers={"Authorization": f"Bearer {token_b}"}).json()["id"]

    asyncio.run(db.save_run(user_id=user_a_id, prompt="alice's prompt", status="success", agent="research"))
    asyncio.run(db.save_run(user_id=user_b_id, prompt="bob's prompt", status="success", agent="analyst"))

    response_a = client.get("/runs", headers={"Authorization": f"Bearer {token_a}"})
    assert response_a.status_code == 200
    body_a = response_a.json()
    assert len(body_a) == 1
    assert body_a[0]["prompt"] == "alice's prompt"

    response_b = client.get("/runs", headers={"Authorization": f"Bearer {token_b}"})
    body_b = response_b.json()
    assert len(body_b) == 1
    assert body_b[0]["prompt"] == "bob's prompt"


def test_run_detail_returns_404_for_another_users_run(tmp_path, monkeypatch):
    _isolated_db(tmp_path, monkeypatch)
    client = TestClient(app)

    token_a = _register(client, "alice2@example.com")
    token_b = _register(client, "bob2@example.com")
    user_a_id = client.get("/auth/me", headers={"Authorization": f"Bearer {token_a}"}).json()["id"]

    run_id = asyncio.run(
        db.save_run(user_id=user_a_id, prompt="alice's private mission", status="success", agent="research")
    )

    response = client.get(f"/runs/{run_id}", headers={"Authorization": f"Bearer {token_b}"})
    assert response.status_code == 404

    own_response = client.get(f"/runs/{run_id}", headers={"Authorization": f"Bearer {token_a}"})
    assert own_response.status_code == 200
    assert own_response.json()["prompt"] == "alice's private mission"
