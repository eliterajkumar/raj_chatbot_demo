from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .api import rag_router, voice_router, multilingual_router

# Load environment variables from a .env file
load_dotenv()

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

# Include the routers from each project
app.include_router(rag_router.router, prefix="/rag", tags=["RAG Chatbot"])
app.include_router(voice_router.router, prefix="/voice", tags=["AI Voice Bot"])
app.include_router(multilingual_router.router, prefix="/multilingual", tags=["Multilingual Chatbot (English only)"])

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the Unified AI Services API. Visit /docs for more information."} 