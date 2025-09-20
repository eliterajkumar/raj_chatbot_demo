from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from pathlib import Path

# Force load .env from project root (same dir as backend/)
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

# Debug: print masked API key (for testing only)
print("DEBUG: .env loaded from", env_path)
api_key = os.getenv("OPENROUTER_API_KEY")
if api_key:
    print("DEBUG: OPENROUTER_API_KEY loaded ✅ ->", api_key[:6] + "..." + api_key[-6:])
else:
    print("DEBUG: OPENROUTER_API_KEY ❌ NOT FOUND")

from .api import rag_router, voice_router, multilingual_router

app = FastAPI(
    title="Unified AI Services API",
    description="A single backend for RAG Chatbot, AI Voice Bot, and Multilingual Chatbot (English only).",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(rag_router.router, prefix="/rag", tags=["RAG Chatbot"])
app.include_router(voice_router.router, prefix="/voice", tags=["AI Voice Bot"])
app.include_router(multilingual_router.router, prefix="/multilingual", tags=["Multilingual Chatbot (English only)"])

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the Unified AI Services API. Visit /docs for more information."}
