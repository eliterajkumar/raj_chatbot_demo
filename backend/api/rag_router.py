# backend/api/rag_router.py
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
import os
import re
import json
from pathlib import Path
from typing import List

from ..services import llm_handler, db

router = APIRouter()

# ---- simple local retrieval helpers ----
CONTEXT_PATH = Path("context/fynorra_master.json")

def load_master_context() -> dict:
    if not CONTEXT_PATH.exists():
        return {}
    with open(CONTEXT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def text_tokens_preview(text: str, n: int = 50) -> str:
    return (" ".join((text or "").split()[:n])).strip()

def find_relevant_chunks(text: str, max_chunks: int = 3) -> List[str]:
    """
    Heuristic retriever:
    - Searches core_services (by name/description), faqs, sales_material, products
    - Returns up to max_chunks text snippets joined as sources_text
    """
    data = load_master_context()
    if not data:
        return []

    text_l = (text or "").lower()
    if not text_l:
        # fallback: return company short bio / about
        company = data.get("company", {})
        summary = company.get("short_bio") or company.get("about") or ""
        return [f"Company summary: {summary}"] if summary else []

    tokens = text_l.split()
    tokens_sample = set(tokens[:8])

    snippets = []

    # search core_services
    for svc_group in data.get("core_services", []):
        for svc in svc_group.get("services", []):
            name = svc.get("name", "")
            desc = svc.get("description", "")
            combined = f"{name} {desc}".lower()
            # match if any of top tokens appear in combined text OR name tokens appear
            if tokens_sample & set(combined.split()) or any(t in combined for t in tokens_sample):
                short = f"Service: {name} — {desc}"
                snippets.append(short)

    # search faqs
    for f in data.get("faqs", []):
        q_a = f.get("q", "") + " " + f.get("a", "")
        if tokens_sample & set(q_a.lower().split()):
            snippets.append(f"FAQ: Q: {f.get('q')} A: {f.get('a')}")

    # search products
    for p in data.get("products", []):
        name = p.get("name", "")
        desc = p.get("description", "")
        combined = f"{name} {desc}".lower()
        if tokens_sample & set(combined.split()):
            snippets.append(f"Product: {p.get('name')} — {p.get('description')}")

    # include sales_material / elevator pitch as fallback context
    sm = data.get("sales_material", {})
    if sm:
        snippets.append(f"Sales: {sm.get('hero_headline','')} — {sm.get('elevator_pitch','')}")

    # dedupe & clamp
    seen = set()
    out = []
    for s in snippets:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= max_chunks:
            break

    # if nothing found, return company summary fallback
    if not out:
        company = data.get("company", {})
        summary = company.get("short_bio") or company.get("about") or ""
        if summary:
            out.append(f"Company summary: {summary}")

    return out

# -------------------------
# Router endpoints
# -------------------------
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

    # ---------------------------
    # Local RAG retrieval: build sources_text from context/fynorra_master.json
    # ---------------------------
    try:
        # Prefer to search against the immediate message; fallback to recent history
        history_preview = ""
        recent_history = db.get_last_messages(conversation_id, limit=6)
        if recent_history:
            history_preview = " ".join([m.get("text","") for m in recent_history])
        search_input = message_text or history_preview or ""
        chunks = find_relevant_chunks(search_input)
        if chunks:
            sources_text = "\n--- SOURCE DOC ---\n" + "\n\n".join(chunks)
        else:
            # fallback: short company summary
            master = load_master_context()
            company_summary = master.get("company", {}).get("short_bio") or master.get("company", {}).get("about") or ""
            sources_text = f"Company summary: {text_tokens_preview(company_summary, 80)}" if company_summary else ""
    except Exception as e:
        print("Context load error:", e)
        sources_text = ""

    # Persona / system prompt (unchanged from your provided persona)
    persona = (
        "You are Fynorra’s AI Assistant, acting as a friendly consultant and sales guide. "
        "Always start the first reply in clear English, with a one-time greeting: 'Namaste 🙏, I’m Fynorra Assistant.' "
        "After greeting, introduce yourself in one line: 'I can help you explore our AI chatbots, automation, cloud, and IT solutions.' "
        "Do not repeat this greeting in later replies. "
        "If the user asks to talk in Hinglish (Hindi + English), politely switch to Hinglish for the rest of the session. "
        "English must always be supported, never refuse it. "
        "Speak in a professional yet friendly tone — like a consultant who is approachable, not robotic. "
        "Explain Fynorra’s services concisely: AI Chatbots, Automation & Integrations, Custom Software, Cloud & DevOps, and IT Consulting. "
        "Use simple language for non-technical users, but provide technical clarity if asked. "
        "Never invent details — only answer from the provided Fynorra context data. "
        "If information is missing, reply: 'I don’t have that information right now.' "
        "When the user shows interest (e.g., asks about pricing, demo, or implementation), "
        "ask one polite qualifying question (like their business size, industry, or use case), "
        "then propose a clear next step: booking a demo, requesting a quote, or sharing contact details. "
        "Always highlight Fynorra’s brand values: 'Elevate Your Digital Vision' and 'Smart. Scalable. Reliable.' "
        "End conversations with a helpful offer, such as: 'Would you like me to share a quick demo link or connect you with our team?' "
    )

    # Conversation history (last few messages for context)
    history = recent_history or db.get_last_messages(conversation_id, limit=6)
    history_text = "\n".join([f"{m['role']}: {m['text']}" for m in history]) if history else ""

    # Build final context: include source doc snippets + recent history
    context = (sources_text + "\n" + history_text).strip() if (sources_text or history_text) else sources_text

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
