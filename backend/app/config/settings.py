from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    APP_NAME: str = "Aegis"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173"

    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = ""
    ANTHROPIC_API_KEY: str = ""
    GEMINI_API_KEY: str = ""

    MODEL_PROVIDER: str = "gemini"
    MODEL_NAME: str = "gemini-2.5-flash"

    LOG_LEVEL: str = "INFO"

    # Local embeddings (sentence-transformers + torch) need more RAM than a free-tier host
    # (e.g. Render's 512MB) can spare once actually loaded — set to false to skip constructing
    # the retrieval stack entirely so agents still answer, just without retrieved-document
    # grounding, instead of the process OOM-crashing on the first RAG-using request.
    RAG_ENABLED: bool = True

    # Dev-only default — MUST be overridden via JWT_SECRET_KEY in .env for any deployment
    # that isn't a single developer's own machine. A predictable secret lets anyone forge
    # valid auth tokens.
    JWT_SECRET_KEY: str = "dev-only-insecure-secret-change-me"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60 * 24

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_cors_origins() -> list[str]:
    origins = settings.CORS_ORIGINS.strip()
    if not origins:
        return []
    if origins == "*":
        return ["*"]
    return [origin.strip() for origin in origins.split(",") if origin.strip()]


settings = get_settings()