import time

from google import genai
from google.genai import types

from app.config.settings import settings
from app.llm.base import BaseLLMProvider, LLMResponse


class GeminiProvider(BaseLLMProvider):
    def __init__(self) -> None:
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not configured.")

        self.client = genai.Client(
            api_key=settings.GEMINI_API_KEY
        )

    async def generate(
        self,
        prompt: str,
        model: str,
        system_prompt: str | None = None,
    ) -> LLMResponse:

        start = time.perf_counter()

        config = None

        if system_prompt:
            config = types.GenerateContentConfig(
                system_instruction=system_prompt
            )

        response = await self.client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

        latency_ms = (time.perf_counter() - start) * 1000

        usage = response.usage_metadata

        return LLMResponse(
            content=response.text or "",
            model=model,
            provider="gemini",
            input_tokens=(
                usage.prompt_token_count
                if usage else None
            ),
            output_tokens=(
                usage.candidates_token_count
                if usage else None
            ),
            latency_ms=latency_ms,
        )