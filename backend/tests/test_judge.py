import pytest

from app.eval.judge import judge_response
from app.models.schemas import ProviderCallResult, RoutingResult, AgentType


class StubLLMService:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = []

    async def generate(self, request, policy, *, system_prompt=None):
        self.calls.append((request, policy, system_prompt))
        return type(
            "Execution",
            (),
            {
                "response": ProviderCallResult(
                    content=self.reply,
                    model="stub-model",
                    provider="openai",
                ),
                "routing": RoutingResult(
                    agent=AgentType.RESEARCH,
                    model="stub-model",
                    provider="openai",
                    confidence=1.0,
                    reason="stub",
                ),
            },
        )()


@pytest.mark.asyncio
async def test_judge_response_skips_mock_provider():
    service = StubLLMService("8")
    score = await judge_response(
        service, task="do something", response_content="ok", provider="mock"
    )
    assert score is None
    assert service.calls == []


@pytest.mark.asyncio
async def test_judge_response_parses_and_normalizes_score():
    service = StubLLMService("8")
    score = await judge_response(
        service, task="do something", response_content="ok", provider="openai"
    )
    assert score == 0.8


@pytest.mark.asyncio
async def test_judge_response_returns_none_when_unparseable():
    service = StubLLMService("not a number")
    score = await judge_response(
        service, task="do something", response_content="ok", provider="openai"
    )
    assert score is None


@pytest.mark.asyncio
async def test_judge_response_returns_none_on_provider_error():
    class FailingLLMService:
        async def generate(self, request, policy, *, system_prompt=None):
            raise RuntimeError("provider is down")

    score = await judge_response(
        FailingLLMService(), task="do something", response_content="ok", provider="openai"
    )
    assert score is None
