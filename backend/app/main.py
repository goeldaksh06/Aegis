from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.chat_stream import router as chat_stream_router
from app.api.health import router as health_router
from app.api.runs import router as runs_router
from app.config.settings import get_cors_origins, settings
from app.database.db import init_db
from app.llm.default_models import register_default_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    register_default_models()
    await init_db()
    yield

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
)

cors_origins = get_cors_origins()
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(chat_stream_router)
app.include_router(runs_router)


@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "running"
    }