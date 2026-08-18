import time

from openai import AsyncOpenAI

from app.config.settings import settings
from app.llm.base import BaseLLMProvider, LLMResponse


class OpenAIProvider(BaseLLMProvider):
    def __init__(self) -> None:
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not configured.")

        client_kwargs: dict[str, str] = {
            "api_key": settings.OPENAI_API_KEY,
        }

        if settings.OPENAI_BASE_URL.strip():
            client_kwargs["base_url"] = settings.OPENAI_BASE_URL.strip()

        self.client = AsyncOpenAI(**client_kwargs)

    async def generate(
        self,
        prompt: str,
        model: str,
        system_prompt: str | None = None,
    ) -> LLMResponse:

        messages = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        start_time = time.perf_counter()

        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        usage = response.usage

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=model,
            provider="openai",
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            latency_ms=latency_ms,
        )