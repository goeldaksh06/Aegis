from __future__ import annotations

from app.database.db import append_conversation_message, load_conversation_history
from app.models.schemas import ChatMessage, MessageRole


async def load_prior_messages(conversation_id: str) -> list[ChatMessage]:
    """Reload a conversation's recent history as ChatMessage objects, oldest first.

    Only ever called when a request explicitly supplies conversation_id — a request without
    one behaves exactly as before this feature existed (single-shot, no persistence, no
    behavior change for existing callers).
    """
    rows = await load_conversation_history(conversation_id)
    return [ChatMessage(role=MessageRole(row.role), content=row.content) for row in rows]


async def persist_turn(conversation_id: str, user_content: str, assistant_content: str) -> None:
    await append_conversation_message(conversation_id, MessageRole.USER.value, user_content)
    await append_conversation_message(conversation_id, MessageRole.ASSISTANT.value, assistant_content)
