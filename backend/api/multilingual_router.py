from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from ..services import llm_handler

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

@router.post("/chat", status_code=status.HTTP_200_OK)
async def handle_chat(request: ChatRequest):
    try:
        # Only process English messages and respond in English
        system_prompt = "You are a polite and helpful customer support agent for a global e-commerce company. Answer the user's query clearly and concisely."
        response = llm_handler.get_llm_response(system_prompt, "", request.message)
        return {"reply": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during chat processing: {e}") 