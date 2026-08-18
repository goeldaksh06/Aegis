from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str

    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None


class BaseLLMProvider(ABC):

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: str,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate a response using the selected model."""
        raise NotImplementedError