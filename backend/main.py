# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from pathlib import Path
import asyncio
import signal
import sys
import logging
from .api import admin_router

# Force load .env from project root (same dir as backend/)
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

# Debug: print masked API key (for testing only)
logging.basicConfig(level=logging.INFO)
logging.info("DEBUG: .env loaded from %s", env_path)
api_key = os.getenv("OPENROUTER_API_KEY")
if api_key:
    logging.info("DEBUG: OPENROUTER_API_KEY loaded ✅ -> %s", api_key[:6] + "..." + api_key[-6:])
else:
    logging.warning("DEBUG: OPENROUTER_API_KEY ❌ NOT FOUND")

from .api import rag_router
from .services import db as db_service  # used by cleanup loop

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
app.include_router(admin_router.router, prefix="/admin", tags=["Admin"])

# Background cleanup task handle
_cleanup_task: asyncio.Task | None = None

async def _cleanup_loop(poll_seconds: int = 60):
    """
    Periodically call db.cleanup_old_sessions to remove old conversations.
    """
    logging.info("Session cleanup loop started (poll %ds)", poll_seconds)
    try:
        while True:
            try:
                deleted = db_service.cleanup_old_sessions()
                if deleted:
                    logging.info("cleanup_old_sessions removed conversations: %s", deleted)
            except Exception as e:
                logging.exception("Error during cleanup_old_sessions: %s", e)
            await asyncio.sleep(poll_seconds)
    except asyncio.CancelledError:
        logging.info("Session cleanup loop cancelled. Exiting cleanup task.")
        raise

@app.on_event("startup")
async def startup_event():
    global _cleanup_task
    # start background cleanup loop
    loop = asyncio.get_event_loop()
    _cleanup_task = loop.create_task(_cleanup_loop(poll_seconds=60))
    logging.info("Application startup complete. Cleanup task started.")

@app.on_event("shutdown")
async def shutdown_event():
    global _cleanup_task
    if _cleanup_task:
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            logging.info("Cleanup task cancelled on shutdown.")
    logging.info("Application shutdown complete.")

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the Unified AI Services API. Visit /docs for more information."}
