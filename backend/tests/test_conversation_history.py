import pytest

from app.database import db
from app.conversation.history import load_prior_messages, persist_turn
from app.models.schemas import MessageRole


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test_conversations.db")
    monkeypatch.setattr(db, "DATABASE_URL", f"sqlite+aiosqlite:///{db.DB_PATH}")
    db.get_engine.cache_clear()
    db.get_sessionmaker.cache_clear()
    db._ready = False


@pytest.mark.asyncio
async def test_load_prior_messages_returns_empty_for_new_conversation():
    messages = await load_prior_messages("brand-new-conversation")
    assert messages == []


@pytest.mark.asyncio
async def test_persist_and_reload_conversation_turns_in_order():
    conversation_id = "conv-1"

    await persist_turn(conversation_id, "My name is Alex.", "Nice to meet you, Alex.")
    await persist_turn(conversation_id, "What is my name?", "Your name is Alex.")

    messages = await load_prior_messages(conversation_id)

    assert [m.content for m in messages] == [
        "My name is Alex.",
        "Nice to meet you, Alex.",
        "What is my name?",
        "Your name is Alex.",
    ]
    assert messages[0].role == MessageRole.USER
    assert messages[1].role == MessageRole.ASSISTANT


@pytest.mark.asyncio
async def test_conversations_are_isolated_by_id():
    await persist_turn("conv-a", "hello from a", "reply to a")
    await persist_turn("conv-b", "hello from b", "reply to b")

    messages_a = await load_prior_messages("conv-a")
    messages_b = await load_prior_messages("conv-b")

    assert [m.content for m in messages_a] == ["hello from a", "reply to a"]
    assert [m.content for m in messages_b] == ["hello from b", "reply to b"]
