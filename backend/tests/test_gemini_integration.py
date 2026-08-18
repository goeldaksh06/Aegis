import pytest

from app.llm.factory import get_provider, register_provider


@pytest.mark.asyncio
async def test_gemini_generate_with_mock():
    # Use the mock provider to avoid paid API calls in CI/local runs
    provider = get_provider("mock")

    response = await provider.generate(
        prompt="Reply with exactly: AEGIS_OK",
        model="mock-model",
    )

    assert response.content.strip() == "AEGIS_OK"
    assert response.provider == "mock"
    assert response.model == "mock-model"
