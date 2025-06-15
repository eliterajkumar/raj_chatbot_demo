from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from rag_chain import qa_chain
import logging
import time
import asyncio

app = FastAPI(
    title="Fynorra AI Assistant API (No History)",
    description="An intelligent assistant powered by LLM & RAG to help with Fynorra-related queries. Chat history is disabled for simplified operation.",
    version="1.3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

class Query(BaseModel):
    question: str

INSTANT_QA = {
    "hi": "👋 Hello! I'm Fynorra's AI assistant. How can I help you today?",
    "hello": "👋 Hi there! Ask me anything about Fynorra.",
    "hey": "Hey! 👋 Need help with Fynorra?",
    "how are you": "I'm doing great! 😊 Ready to assist with anything related to Fynorra.",
    "what's up": "All good here! Let me know what you'd like to explore in Fynorra.",
    "what is fynorra": "Fynorra is an AI automation platform to build models, APIs, and automate workflows—no coding needed!",
    "what does fynorra do": "Fynorra helps businesses automate tasks, train AI models, and deploy smart APIs quickly.",
    "how can i use fynorra": "You can upload your data, fine-tune models, generate APIs, and automate workflows in minutes.",
    "is fynorra free": "Yes! Fynorra offers a free plan for basic features, plus pro & enterprise tiers.",
    "who built fynorra": "Fynorra is built by AI experts aiming to simplify intelligent automation for everyone.",
    "fynorra features": "Fynorra offers model training, API generation, custom chatbots, workflow automation, and more.",
    "fynorra contact": "Reach out to us at info@fynorra.com or use the contact form on our site.",
    "goodbye": "Goodbye! 😊 Feel free to come back anytime if you have more questions.",
    "bye": "Goodbye! 😊 Feel free to come back anytime if you have more questions."
}

@app.post("/ask")
async def ask_question(query: Query, request: Request):
    start_time = time.time()
    user_ip = request.client.host
    question = query.question.strip()
    question_lower = question.lower()

    logger.info(f"📥 From {user_ip} | Question: '{question}'")

    try:
        if question_lower in INSTANT_QA:
            logger.info("⚡ Instant answer triggered (exact match).")
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "question": question,
                    "answer": INSTANT_QA[question_lower],
                    "type": "instant",
                    "latency_ms": int((time.time() - start_time) * 1000)
                }
            )

        current_chat_history = []

        logger.info(f"🧠 Querying RAG chain.")
        answer = await asyncio.to_thread(qa_chain, question, current_chat_history)

        if "i don't have enough information" in answer.lower() or \
           "not directly available in the provided context" in answer.lower():
            logger.warning(f"🤷 No relevant info found in RAG for: '{question}'.")
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "question": question,
                    "answer": "I don't have enough information from Fynorra's knowledge base to answer that. Could you please rephrase, or ask about our core services like chatbots, automation, or software development?",
                    "type": "no_info",
                    "latency_ms": int((time.time() - start_time) * 1000)
                }
            )

        logger.info("✅ RAG answer returned successfully.")
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "question": question,
                "answer": answer,
                "type": "rag",
                "latency_ms": int((time.time() - start_time) * 1000)
            }
        )

    except Exception as e:
        logger.error(f"❌ Unhandled Error processing '{question}': {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "question": question,
                "answer": "Apologies! A critical error occurred. Please try again or contact support if the issue persists.",
                "error_detail": str(e),
                "type": "unhandled_error",
                "latency_ms": int((time.time() - start_time) * 1000)
            }
        )

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting Fynorra AI Assistant API in development mode on http://0.0.0.0:8001")
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
