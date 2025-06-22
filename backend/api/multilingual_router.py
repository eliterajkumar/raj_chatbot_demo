from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from ..services import translation, llm_handler

router = APIRouter()

class ChatRequest(BaseModel):
    language: str
    message: str

@router.post("/chat", status_code=status.HTTP_200_OK)
async def handle_chat(request: ChatRequest):
    try:
        original_lang = request.language
        original_message = request.message
        
        # 1. Translate to English if necessary
        if original_lang != "en":
            message_to_process = translation.translate_text(original_message, dest_lang="en")
        else:
            message_to_process = original_message

        # 2. Get response from LLM
        system_prompt = "You are a polite and helpful customer support agent for a global e-commerce company. Answer the user's query clearly and concisely."
        english_response = llm_handler.get_llm_response(system_prompt, "", message_to_process)

        # 3. Translate response back to original language if necessary
        if original_lang != "en":
            final_response = translation.translate_text(english_response, dest_lang=original_lang)
        else:
            final_response = english_response
            
        return {"reply": final_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during chat processing: {e}") 