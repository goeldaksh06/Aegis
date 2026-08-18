from __future__ import annotations

import uuid
from datetime import datetime, timezone
from functools import lru_cache

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config.settings import BACKEND_DIR

DB_PATH = BACKEND_DIR / "aegis.db"
DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class RunRecord(Base):
    """One row per request/response *event* (a "mission"), including failed/blocked ones.

    user_id is nullable so the existing zero-friction anonymous demo (auto-run on page load,
    scenario chips) keeps working without requiring login — /chat and /chat/stream accept an
    optional bearer token and attach user_id when present, but never require it. Personal
    mission history (GET /runs) does require auth and is always filtered server-side by the
    authenticated user's id, never a client-supplied one.
    """

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
    prompt: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String)
    agent: Mapped[str] = mapped_column(String, nullable=True)
    model: Mapped[str] = mapped_column(String, nullable=True)
    provider: Mapped[str] = mapped_column(String, nullable=True)
    risk_level: Mapped[str] = mapped_column(String, nullable=True)
    risk_score: Mapped[float] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=True)
    error: Mapped[str] = mapped_column(Text, nullable=True)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=True)
    moderation_blocked: Mapped[bool] = mapped_column(Boolean, nullable=True)
    conversation_id: Mapped[str] = mapped_column(String, nullable=True)


class AgentStep(Base):
    """One row per real LLM call ("generation" stage) within a mission.

    A mission with one agent has one step; an orchestrated mission (Planner dispatching to
    sub-agents) has one step per agent actually invoked. This is what powers the per-agent
    observability breakdown (real timing/tokens/cost per agent, not the request-level total
    RunRecord already tracked) — populated from the same on_stage checkpoints that already
    drive the live SSE trace, so there is exactly one source of this data, not two.
    """

    __tablename__ = "agent_steps"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id: Mapped[str] = mapped_column(String, ForeignKey("runs.id"), index=True)
    agent: Mapped[str] = mapped_column(String)
    step_index: Mapped[int] = mapped_column(Integer, default=0)
    model: Mapped[str] = mapped_column(String, nullable=True)
    provider: Mapped[str] = mapped_column(String, nullable=True)
    duration_ms: Mapped[float] = mapped_column(Float, nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=True)
    retrieved_count: Mapped[int] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String, default="ok")
    error: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class ConversationMessage(Base):
    """Persisted turn-by-turn history for multi-turn conversations (see app/conversation/).

    A separate table from RunRecord rather than reusing it: RunRecord is one row per
    request/response *event* (including errors, which have no message content worth
    replaying), while this is one row per actual conversation *message* in order, which is
    what needs to be reloaded and replayed as context on the next turn.
    """

    __tablename__ = "conversation_messages"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id: Mapped[str] = mapped_column(String, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    role: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(Text)


@lru_cache
def get_engine():
    return create_async_engine(DATABASE_URL, echo=False)


@lru_cache
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(), expire_on_commit=False)


_ready = False


_NEW_RUN_COLUMNS: dict[str, str] = {
    "cost_usd": "FLOAT",
    "moderation_blocked": "BOOLEAN",
    "conversation_id": "VARCHAR",
    "user_id": "VARCHAR",
}


async def _add_missing_columns(conn) -> None:
    """Lightweight ad-hoc migration for columns added to RunRecord after first deploy.

    `Base.metadata.create_all` only creates tables that don't exist yet — it never alters an
    existing table, so a `runs` table created by an earlier version of this schema would
    otherwise silently lack new columns forever. There's no Alembic in this project (SQLite +
    a handful of demo tables doesn't justify a migration framework), so this does the one
    thing that actually matters: add any missing column with a NULL default, which is exactly
    what every new nullable column here needs — existing rows just get NULL for it.
    """
    result = await conn.exec_driver_sql("PRAGMA table_info(runs)")
    existing_columns = {row[1] for row in result.fetchall()}

    for column_name, column_type in _NEW_RUN_COLUMNS.items():
        if column_name not in existing_columns:
            await conn.exec_driver_sql(f"ALTER TABLE runs ADD COLUMN {column_name} {column_type}")


async def init_db() -> None:
    global _ready
    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _add_missing_columns(conn)
    _ready = True


async def _ensure_ready() -> None:
    if not _ready:
        await init_db()


# --- Users -------------------------------------------------------------------------------

async def create_user(*, email: str, hashed_password: str) -> User:
    await _ensure_ready()
    async with get_sessionmaker()() as session:
        user = User(email=email, hashed_password=hashed_password)
        session.add(user)
        try:
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        await session.refresh(user)
        return user


async def get_user_by_email(email: str) -> User | None:
    await _ensure_ready()
    async with get_sessionmaker()() as session:
        result = await session.execute(select(User).where(User.email == email))
        return result.scalars().first()


async def get_user_by_id(user_id: str) -> User | None:
    await _ensure_ready()
    async with get_sessionmaker()() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        return result.scalars().first()


# --- Runs / missions -----------------------------------------------------------------------

async def save_run(
    *,
    id: str | None = None,
    user_id: str | None = None,
    prompt: str,
    status: str,
    agent: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    risk_level: str | None = None,
    risk_score: float | None = None,
    latency_ms: float | None = None,
    error: str | None = None,
    cost_usd: float | None = None,
    moderation_blocked: bool | None = None,
    conversation_id: str | None = None,
) -> str:
    await _ensure_ready()
    run_id = id or str(uuid.uuid4())
    async with get_sessionmaker()() as session:
        session.add(
            RunRecord(
                id=run_id,
                user_id=user_id,
                prompt=prompt,
                status=status,
                agent=agent,
                model=model,
                provider=provider,
                risk_level=risk_level,
                risk_score=risk_score,
                latency_ms=latency_ms,
                error=error,
                cost_usd=cost_usd,
                moderation_blocked=moderation_blocked,
                conversation_id=conversation_id,
            )
        )
        await session.commit()
    return run_id


async def list_runs(limit: int = 20, *, user_id: str | None = None) -> list[RunRecord]:
    await _ensure_ready()
    async with get_sessionmaker()() as session:
        query = select(RunRecord).order_by(RunRecord.created_at.desc()).limit(limit)
        if user_id is not None:
            query = query.where(RunRecord.user_id == user_id)
        result = await session.execute(query)
        return list(result.scalars().all())


async def get_run(run_id: str, *, user_id: str | None = None) -> RunRecord | None:
    """Fetch one run, optionally scoped to a specific owning user (never trust a client-
    supplied user id — the caller passes the id already resolved from a validated token)."""
    await _ensure_ready()
    async with get_sessionmaker()() as session:
        query = select(RunRecord).where(RunRecord.id == run_id)
        if user_id is not None:
            query = query.where(RunRecord.user_id == user_id)
        result = await session.execute(query)
        return result.scalars().first()


# --- Agent step observability ---------------------------------------------------------------

async def save_agent_steps(run_id: str, steps: list[dict]) -> None:
    if not steps:
        return
    await _ensure_ready()
    async with get_sessionmaker()() as session:
        for index, step in enumerate(steps):
            session.add(
                AgentStep(
                    run_id=run_id,
                    agent=step.get("agent") or "unknown",
                    step_index=index,
                    model=step.get("model"),
                    provider=step.get("provider"),
                    duration_ms=step.get("duration_ms"),
                    input_tokens=step.get("input_tokens"),
                    output_tokens=step.get("output_tokens"),
                    cost_usd=step.get("cost_usd"),
                    retrieved_count=step.get("retrieved_count"),
                    status=step.get("status", "ok"),
                    error=step.get("error"),
                )
            )
        await session.commit()


async def list_agent_steps(run_id: str) -> list[AgentStep]:
    await _ensure_ready()
    async with get_sessionmaker()() as session:
        result = await session.execute(
            select(AgentStep).where(AgentStep.run_id == run_id).order_by(AgentStep.step_index)
        )
        return list(result.scalars().all())


# --- Conversation memory ---------------------------------------------------------------------

async def append_conversation_message(conversation_id: str, role: str, content: str) -> None:
    await _ensure_ready()
    async with get_sessionmaker()() as session:
        session.add(
            ConversationMessage(conversation_id=conversation_id, role=role, content=content)
        )
        await session.commit()


async def load_conversation_history(conversation_id: str, limit_messages: int = 12) -> list[ConversationMessage]:
    """Return the most recent `limit_messages` turns for a conversation, oldest first.

    The cap is a deliberate context-window control, not an arbitrary number — without it, a
    long-running conversation would keep growing the prompt sent to the LLM on every turn
    (unbounded cost and eventually a context-length error). 12 messages (~6 exchanges) is a
    reasonable default for a crisis-copilot back-and-forth; a production system would likely
    replace this with token-aware truncation or summarization instead of a flat message count.
    """
    await _ensure_ready()
    async with get_sessionmaker()() as session:
        result = await session.execute(
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation_id)
            .order_by(ConversationMessage.created_at.desc())
            .limit(limit_messages)
        )
        return list(reversed(result.scalars().all()))
