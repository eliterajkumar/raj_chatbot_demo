import os
import requests
from io import BytesIO

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_URL = "https://api.openai.com/v1/audio/transcriptions"

def transcribe_audio_with_whisper(audio_bytes: bytes, original_filename: str) -> str:
    """Sends audio data to the OpenAI Whisper API for transcription."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    if not audio_bytes:
        raise ValueError("Audio data cannot be empty.")

    try:
        audio_file = BytesIO(audio_bytes)
        files = {'file': (original_filename, audio_file, 'audio/mpeg')}
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        data = {"model": "whisper-1"}

        response = requests.post(API_URL, headers=headers, files=files, data=data)
        response.raise_for_status()
        
        return response.json().get("text", "No text found in audio.")
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Whisper API request failed: {e}")
    except KeyError:
        raise ValueError("Failed to parse Whisper API response.") 