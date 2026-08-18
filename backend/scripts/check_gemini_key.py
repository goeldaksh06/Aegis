from __future__ import annotations

import asyncio
import traceback

from app.config.settings import settings
from google import genai


async def main() -> None:
    print("Using GEMINI_API_KEY length:", len(settings.GEMINI_API_KEY or ""))
    try:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)

        print("Created genai.Client; attempting small generate_content call...")

        resp = await client.aio.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents="Ping",
        )

        print("Success:")
        print(resp)

    except Exception as exc:
        print("Exception raised during Gemini call:")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
