from app.llm.providers import openai as openai_provider_module


def test_openai_provider_uses_configured_base_url(monkeypatch):
    captured_kwargs = {}

    class DummyClient:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(openai_provider_module, "AsyncOpenAI", DummyClient)
    monkeypatch.setattr(openai_provider_module.settings, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(openai_provider_module.settings, "OPENAI_BASE_URL", "https://engine.misterpilot.online/v1")

    provider = openai_provider_module.OpenAIProvider()

    assert provider is not None
    assert captured_kwargs == {
        "api_key": "test-key",
        "base_url": "https://engine.misterpilot.online/v1",
    }