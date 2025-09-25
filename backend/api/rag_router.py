# backend/api/rag_router.py
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
import os
import re
import json
import logging
from pathlib import Path
from typing import List

from ..services import llm_handler, db

router = APIRouter()
logger = logging.getLogger("rag_router")

# Local RAG master context path (fallback)
CONTEXT_PATH = Path("context/fynorra_master_with_faqs")

def load_master_context() -> dict:
    if not CONTEXT_PATH.exists():
        return {}
    try:
        with open(CONTEXT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.exception("Failed to load master context: %s", e)
        return {}

def text_tokens_preview(text: str, n: int = 50) -> str:
    return (" ".join((text or "").split()[:n])).strip()

def find_relevant_chunks(text: str, max_chunks: int = 3) -> List[str]:
    data = load_master_context()
    if not data:
        return []

    text_l = (text or "").lower()
    if not text_l:
        company = data.get("company", {})
        summary = company.get("short_bio") or company.get("about") or ""
        return [f"Company summary: {summary}"] if summary else []

    tokens = text_l.split()
    tokens_sample = set(tokens[:8])
    snippets = []

    # core_services search
    for svc_group in data.get("core_services", []):
        for svc in svc_group.get("services", []):
            name = svc.get("name", "")
            desc = svc.get("description", "")
            combined = f"{name} {desc}".lower()
            if tokens_sample & set(combined.split()) or any(t in combined for t in tokens_sample):
                snippets.append(f"Service: {name} — {desc}")

    # faqs
    for f in data.get("faqs", []):
        q_a = (f.get("q", "") + " " + f.get("a", "")).lower()
        if tokens_sample & set(q_a.split()):
            snippets.append(f"FAQ: Q: {f.get('q')} A: {f.get('a')}")

    # products
    for p in data.get("products", []):
        name = p.get("name", "")
        desc = p.get("description", "")
        combined = f"{name} {desc}".lower()
        if tokens_sample & set(combined.split()):
            snippets.append(f"Product: {name} — {desc}")

    # sales material fallback
    sm = data.get("sales_material", {})
    hero = sm.get("hero_headline","")
    pitch = sm.get("elevator_pitch","")
    if hero or pitch:
        snippets.append(f"Sales: {hero} — {pitch}")

    # dedupe and clamp
    seen = set()
    out = []
    for s in snippets:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= max_chunks:
            break

    if not out:
        company = data.get("company", {})
        summary = company.get("short_bio") or company.get("about") or ""
        if summary:
            out.append(f"Company summary: {summary}")

    return out

# company-info intent detection (Hindi/English keywords)
COMPANY_INFO_RE = re.compile(
    r"\b(kab|kab shuru|founder|owner|who founded|owner kaun|establish|incorporat|cin|employees|kitne employees|headquarter|hq|where located|incorporation|founded)\b",
    re.I
)

# base system persona (English-first; Hinglish optional)
BASE_PERSONA = (
    "You are Fynorra AI Assistant — the official AI representative of Fynorra AI Solutions. "
    "Tone: professional, concise, helpful, sales-aware but not pushy. "
    "Rules:\n"
    "1) Default language: English. Only switch to Hinglish if the user explicitly requests Hindi/Hinglish. "
    "2) Greet once per session. Greeting (first reply only) should be:\n"
    "   'Namaste 🙏, I’m Fynorra AI — your AI automation partner. How can I help you today?'\n"
    "   Do not repeat this greeting in subsequent replies.\n"
    "3) Always ask 1-3 discovery questions before providing final quotes or detailed timelines.\n"
    "4) When answering factual queries, prefer retrieved documents (service docs or company_profile). If company_profile is unverified, prefix with 'According to public records...'.\n"
    "5) When recommending a service, include: service name, one-line why, estimated dev cost range, monthly OPEX range, timeline estimate, and a clear CTA.\n"
    "6) Never invent facts; if data is missing, say 'I don’t have that information right now' and offer human handoff.\n"
    "7) Emphasize Fynorra's automation-first model (AI agents handle most workflows) unless asked otherwise.\n"
    "8) End replies with a Direct Action Step (e.g., schedule a discovery call, request docs, or ask for confirmation)."
)

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

    # Ensure conversation
    conv = db.upsert_conversation(session_id)
    conversation_id, session_id = conv["id"], conv["session_id"]

    # Save user message
    db.save_message(conversation_id, role="user", text=message_text, file_url=None)

    # check recent history and whether assistant has greeted
    recent_history = db.get_last_messages(conversation_id, limit=20) or []
    assistant_has_greeted = any(m.get("role") == "assistant" and "Namaste" in (m.get("text") or "") for m in recent_history)

    # LANGUAGE: detect explicit Hindi/Hinglish request
    use_hinglish = False
    if re.search(r"\b(hindi|hinglish|हिंदी|हिंग्लिश|bol in hindi|bol hindi|हेलो हिंदी)\b", message_text, re.I):
        use_hinglish = True

    # RETRIEVAL: primary = vector DB via db.search_documents (if available), fallback = local master context
    sources_parts: List[str] = []
    try:
        hits = []
        if hasattr(db, "search_documents"):
            hits = db.search_documents(message_text, top_k=4) or []
        elif hasattr(db, "search_services"):
            hits = db.search_services(message_text, top_k=4) or []
        else:
            hits = []

        for h in hits:
            meta_name = (h.get("metadata") or {}).get("service_name") or h.get("id")
            text = h.get("text") or h.get("snippet") or ""
            sources_parts.append(f"[VECTOR-HIT] {meta_name}\n{text[:1500]}")
    except Exception as e:
        logger.debug("Vector search failed: %s", e)
        hits = []

    # Fallback to local context if no vector hits or for company-info
    if not sources_parts or COMPANY_INFO_RE.search(message_text):
        try:
            # If user asks company-info, prefer company_profile from DB (fetch_by_id) else local
            cp = None
            if hasattr(db, "fetch_by_id"):
                try:
                    cp = db.fetch_by_id("company_profile")
                except Exception:
                    cp = None
            if cp and isinstance(cp, dict):
                cp_text = (cp.get("metadata") or {}).get("text") or cp.get("text") or json.dumps(cp)
                verified = (cp.get("metadata") or {}).get("verified", False)
                prefix = "" if verified else "(According to public records) "
                sources_parts.insert(0, f"[COMPANY_PROFILE] {prefix}{cp_text[:1200]}")
            else:
                # local master fallback
                chunks = find_relevant_chunks(message_text)
                for c in chunks:
                    sources_parts.append(f"[LOCAL] {c}")
                # If nothing at all, include short company bio
                if not sources_parts:
                    master = load_master_context()
                    company_summary = master.get("company", {}).get("short_bio") or master.get("company", {}).get("about") or ""
                    if company_summary:
                        sources_parts.append(f"[LOCAL] Company summary: {text_tokens_preview(company_summary, 120)}")
        except Exception as e:
            logger.exception("Fallback retrieval error: %s", e)

    sources_text = ("\n\n--- SOURCE ---\n\n".join(sources_parts)).strip() if sources_parts else ""

    # build history context (short)
    history = recent_history[:6] if recent_history else []
    history_text = "\n".join([f"{m['role']}: {m['text']}" for m in history]) if history else ""

    # Compose system prompt with language preference
    lang_pref = "Prefer English in responses. Switch to Hinglish only if user asks." if not use_hinglish else "Use Hinglish for responses."
    system_prompt = BASE_PERSONA + "\n" + lang_pref

    # Build final context: sources first (priority), then history
    final_context = "\n".join([sources_text, history_text]).strip() if (sources_text or history_text) else sources_text or history_text

    # LLM call
    try:
        reply = llm_handler.get_llm_response(
            system_prompt=system_prompt,
            context=final_context,
            user_question=message_text,
            model=os.environ.get("LLM_MODEL", None),  # optional override
            request_type="chat",
        )
    except Exception as e:
        logger.exception("LLM error (full): %s", e)
        msg = str(e)
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {msg[:500]}")

    # Prepend canonical greeting once if not present and assistant hasn't greeted
    if not assistant_has_greeted and "Namaste" not in (reply or ""):
        # add English-first greeting (per new rule)
        greeting = "Namaste 🙏, I’m Fynorra AI — your AI automation partner. How can I help you today?\n\n"
        reply = greeting + reply

    # Save assistant reply
    db.save_message(conversation_id, role="assistant", text=reply, file_url=None)

    # ---------------------------
    # Safer Lead detection (weighted + contact detection)
    # ---------------------------

    def extract_contact(text: str):
        email_re = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
        phone_re = r"(\+?\d[\d\s\-\(\)]{6,}\d)"
        return bool(re.search(email_re, text)), bool(re.search(phone_re, text))

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
            logger.exception("Notify sales failed: %s", e)

    return JSONResponse({
        "reply": reply,
        "session_id": session_id,
        "is_lead": is_lead,
        "lead": lead
    })
