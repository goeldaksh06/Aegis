from fastapi.testclient import TestClient

from app.llm.default_models import register_default_models
from app.llm.registry import MODEL_REGISTRY
from app.main import app


def test_app_startup_registers_default_models():
    MODEL_REGISTRY.clear()

    with TestClient(app):
        assert "gemini-2.5-flash-lite" in MODEL_REGISTRY
        assert "gemini-2.5-flash" in MODEL_REGISTRY

    MODEL_REGISTRY.clear()
    register_default_models()
