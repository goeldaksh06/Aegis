from fastapi import APIRouter, Depends

from app.api.chat_service import ChatService
from app.app_container import get_chat_service
from app.auth.dependencies import get_current_user_optional
from app.database.db import User
from app.models.schemas import ChatRequest, ChatResponse


router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
	request: ChatRequest,
	chat_service: ChatService = Depends(get_chat_service),
	current_user: User | None = Depends(get_current_user_optional),
) -> ChatResponse:
	result = await chat_service.chat(request, user_id=current_user.id if current_user else None)
	return result.response
