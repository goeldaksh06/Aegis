from fastapi import APIRouter, Depends

from app.api.chat_service import ChatService
from app.app_container import get_chat_service
from app.models.schemas import ChatRequest, ChatResponse


router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
	request: ChatRequest,
	chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
	result = await chat_service.chat(request)
	return result.response

