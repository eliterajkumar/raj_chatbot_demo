# backend/api/rag_router.py
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
import os, re

from ..services import llm_handler, db

router = APIRouter()

@router.post("/chat")
async def chat_endpoint(request: Request):
    """
    Chat endpoint (site-only, text-only).
    Expects JSON body: {"message":"...", "session_id":"..."}
    Multipart/form uploads are rejected.
    """
    ct = request.headers.get("content-type", "")
    message_text, session_id = "", None

    # Reject multipart uploads on site version
    if "multipart/form-data" in ct:
        raise HTTPException(status_code=400, detail="File uploads are disabled on this deployment. Use demo project for file uploads.")

    # Parse JSON payload
    body = await request.json()
    message_text = body.get("message") or body.get("text") or ""
    session_id = body.get("session_id") or None

    # Ensure conversation (db.upsert_conversation should create or return conv)
    conv = db.upsert_conversation(session_id)
    conversation_id, session_id = conv["id"], conv["session_id"]

    # Save user message
    db.save_message(conversation_id, role="user", text=message_text, file_url=None)

    # No file parsing / RAG in site version
    sources_text = ""

    # Persona / system prompt
    persona = (
    "You are Fynorra’s friendly Sales Assistant. "
    "Always start the first reply in clear English. "
    "Greet the user once at the start of a session with 'Namaste 🙏' and a one-line intro. "
    "After that, avoid repeating the greeting in subsequent replies. "
    "If the user explicitly asks to talk in Hinglish (Hindi + English), then politely switch to Hinglish for the rest of the conversation. "
    "Never refuse English — always support English. "
    "Explain Fynorra services (AI Chatbots, Automation, IT Consulting) concisely, be persuasive but never pushy. "
    "When interest is detected, ask one qualifying question and propose a next step (like demo or contact)."
)
    # Conversation history
    history = db.get_last_messages(conversation_id, limit=6)
    history_text = "\n".join([f"{m['role']}: {m['text']}" for m in history])

    # Build context for LLM (history only, no RAG)
    context = sources_text + "\n" + history_text if history_text else sources_text

    # LLM call
    try:
        reply = llm_handler.get_llm_response(
            system_prompt=persona,
            context=context,
            user_question=message_text,
            model=os.environ.get("LLM_MODEL", None),  # optional override; llm_handler has its own default
            request_type="chat",
        )
    except Exception as e:
        print("LLM error (full):", repr(e))
        msg = str(e)
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {msg[:500]}")

    # Save assistant reply
    db.save_message(conversation_id, role="assistant", text=reply, file_url=None)

    # ---------------------------
    # Safer Lead detection (weighted + contact detection)
    # ---------------------------

    def extract_contact(text: str):
        email_re = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
        phone_re = r"(\+?\d[\d\s\-\(\)]{6,}\d)"
        return bool(re.search(email_re, text)), bool(re.search(phone_re, text))

    # higher weight = stronger buying intent
    KEYWORD_WEIGHTS = {
        r"\b(demo|schedule demo|book demo)\b": 0.95,
        r"\b(price|pricing|cost|quote)\b": 0.85,
        r"\b(interested|want to buy|purchase|signup|sign up|get started)\b": 0.75,
        r"\b(contact|call me|reach out|connect)\b": 0.65,
        r"\b(schedule|meeting|request demo)\b": 0.8,
    }

    def compute_interest_score(text: str):
        score = 0.0
        txt = (text or "").lower()
        for kw_re, w in KEYWORD_WEIGHTS.items():
            if re.search(kw_re, txt):
                score = max(score, w)
        return score

    combined_text = (message_text or "") + " " + (reply or "")
    contact_email_present, contact_phone_present = extract_contact(combined_text)
    score_message = compute_interest_score(message_text)
    score_combined = compute_interest_score(combined_text)

    # boost if contact info present
    if contact_email_present or contact_phone_present:
        score_combined = max(score_combined, 0.98)

    LEAD_THRESHOLD = float(os.environ.get("LEAD_THRESHOLD", 0.75))

    is_lead = score_combined >= LEAD_THRESHOLD

    # extra guard: don't treat short generic info questions as leads
    if not is_lead:
        words = (message_text or "").strip().split()
        if len(words) < 6 and any(qw in (message_text or "").lower() for qw in ["what", "who", "how", "tell", "batao", "kya"]):
            is_lead = False

    lead = None
    if is_lead:
        lead = db.create_lead(
            conversation_id,
            snippet=(message_text or "")[:500],
            score=float(score_combined),
            metadata={"detected_contact": contact_email_present or contact_phone_present}
        )
        try:
            db.notify_sales(lead)
        except Exception as e:
            print("Notify sales failed:", e)

    return JSONResponse({
        "reply": reply,
        "session_id": session_id,
        "is_lead": is_lead,
        "lead": lead
    })
