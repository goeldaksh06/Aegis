import asyncio

from fastapi.testclient import TestClient

from app.database import db
from app.main import app


def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test_auth.db")
    monkeypatch.setattr(db, "DATABASE_URL", f"sqlite+aiosqlite:///{db.DB_PATH}")
    db.get_engine.cache_clear()
    db.get_sessionmaker.cache_clear()
    db._ready = False


def test_register_returns_token_and_creates_user(tmp_path, monkeypatch):
    _isolated_db(tmp_path, monkeypatch)
    client = TestClient(app)

    response = client.post("/auth/register", json={"email": "new@example.com", "password": "correct-horse-1"})

    assert response.status_code == 201
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["user"]["email"] == "new@example.com"
    assert "access_token" in body


def test_register_rejects_duplicate_email(tmp_path, monkeypatch):
    _isolated_db(tmp_path, monkeypatch)
    client = TestClient(app)

    client.post("/auth/register", json={"email": "dup@example.com", "password": "correct-horse-1"})
    response = client.post("/auth/register", json={"email": "dup@example.com", "password": "another-pass-1"})

    assert response.status_code == 409


def test_register_rejects_invalid_email(tmp_path, monkeypatch):
    _isolated_db(tmp_path, monkeypatch)
    client = TestClient(app)

    response = client.post("/auth/register", json={"email": "not-an-email", "password": "correct-horse-1"})

    assert response.status_code == 422


def test_login_succeeds_with_correct_password(tmp_path, monkeypatch):
    _isolated_db(tmp_path, monkeypatch)
    client = TestClient(app)

    client.post("/auth/register", json={"email": "login@example.com", "password": "correct-horse-1"})
    response = client.post("/auth/login", json={"email": "login@example.com", "password": "correct-horse-1"})

    assert response.status_code == 200
    assert "access_token" in response.json()


def test_login_rejects_wrong_password(tmp_path, monkeypatch):
    _isolated_db(tmp_path, monkeypatch)
    client = TestClient(app)

    client.post("/auth/register", json={"email": "wrongpw@example.com", "password": "correct-horse-1"})
    response = client.post("/auth/login", json={"email": "wrongpw@example.com", "password": "totally-wrong"})

    assert response.status_code == 401


def test_login_rejects_unknown_email(tmp_path, monkeypatch):
    _isolated_db(tmp_path, monkeypatch)
    client = TestClient(app)

    response = client.post("/auth/login", json={"email": "ghost@example.com", "password": "correct-horse-1"})

    assert response.status_code == 401


def test_me_requires_valid_token(tmp_path, monkeypatch):
    _isolated_db(tmp_path, monkeypatch)
    client = TestClient(app)

    response = client.get("/auth/me")
    assert response.status_code == 401

    bad_token_response = client.get("/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert bad_token_response.status_code == 401


def test_me_returns_current_user_with_valid_token(tmp_path, monkeypatch):
    _isolated_db(tmp_path, monkeypatch)
    client = TestClient(app)

    register_response = client.post(
        "/auth/register", json={"email": "me@example.com", "password": "correct-horse-1"}
    )
    token = register_response.json()["access_token"]

    response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    assert response.json()["email"] == "me@example.com"


def test_passwords_are_hashed_not_stored_in_plaintext(tmp_path, monkeypatch):
    _isolated_db(tmp_path, monkeypatch)
    client = TestClient(app)

    client.post("/auth/register", json={"email": "hash@example.com", "password": "correct-horse-1"})

    user = asyncio.run(db.get_user_by_email("hash@example.com"))
    assert user.hashed_password != "correct-horse-1"
    assert user.hashed_password.startswith("$2b$")
