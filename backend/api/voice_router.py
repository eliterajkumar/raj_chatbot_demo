from fastapi import APIRouter, UploadFile, File, HTTPException, status
from pydantic import BaseModel

from ..services import transcription, llm_handler

router = APIRouter()

class RespondRequest(BaseModel):
    message: str

@router.post("/transcribe", status_code=status.HTTP_200_OK)
async def transcribe_audio(file: UploadFile = File(...)):
    if not (file.filename.endswith(".mp3") or file.filename.endswith(".wav")):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an MP3 or WAV file.")
    
    try:
        audio_bytes = await file.read()
        transcribed_text = transcription.transcribe_audio_with_whisper(audio_bytes, file.filename)
        return {"transcription": transcribed_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during transcription: {e}")

@router.post("/respond", status_code=status.HTTP_200_OK)
async def respond_to_message(request: RespondRequest):
    try:
        system_prompt = (
            "You are a friendly and professional real estate agent assistant. "
            "Your goal is to help clients with their inquiries about property bookings, availability, and pricing. "
            "Be concise, helpful, and polite. If you don't know the answer, say you will check with a senior agent."
        )
        response_text = llm_handler.get_llm_response(system_prompt, "", request.message)
        return {"reply": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {e}") 