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

# Local RAG master context path (fallback) — ensure .json file
CONTEXT_PATH = Path("context/fynorra_master_with_faqs.json")

def load_master_context() -> dict:
    candidates = [CONTEXT_PATH, Path("context/fynorra_master_with_faqs"), Path("context/fynorra_master_combined.json"),
                  Path("/mnt/data/fynorra_master_with_faqs.json"), Path("/mnt/data/fynorra_master_combined.json")]
    for p in candidates:
        try:
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "company_profile" in data and "company" not in data:
                        data["company"] = data.get("company_profile")
                    if "core_services" in data and "services" not in data:
                        svc_list = []
                        for grp in data.get("core_services", []):
                            for s in grp.get("services", []):
                                svc_list.append(s)
                        data["services"] = svc_list
                    return data
        except Exception as e:
            logger.exception("Failed to load master context from %s: %s", p, e)
    return {}

def text_tokens_preview(text: str, n: int = 50) -> str:
    return (" ".join((text or "").split()[:n])).strip()

def _extract_text_from_rag_entry(entry) -> str:
    if not entry:
        return ""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        if "text" in entry:
            return entry.get("text", "")
        if "messages" in entry:
            msgs = entry.get("messages", [])
            return " ".join([m.get("content", "") for m in msgs if isinstance(m, dict)])
        return json.dumps(entry, ensure_ascii=False)
    return str(entry)

def find_relevant_chunks(text: str, max_chunks: int = 3) -> List[str]:
    data = load_master_context()
    if not data:
        return []

    text_l = (text or "").lower()
    if not text_l:
        company = data.get("company", {}) or {}
        summary = company.get("short_bio") or company.get("about") or company.get("business_activity") or ""
        return [f"Company summary: {summary}"] if summary else []

    tokens = text_l.split()
    tokens_sample = set(tokens[:8])
    snippets = []

    for svc in data.get("services", []):
        name = svc.get("service_name") or svc.get("name") or svc.get("service_id") or ""
        desc = svc.get("short_description") or svc.get("description") or svc.get("sales_pitch") or ""
        combined = f"{name} {desc}".lower()
        if tokens_sample & set(combined.split()) or any(t in combined for t in tokens_sample):
            snippets.append(f"Service: {name} — {desc}")

    for grp in data.get("core_services", []):
        for svc in grp.get("services", []):
            name = svc.get("name", "")
            desc = svc.get("description", "")
            combined = f"{name} {desc}".lower()
            if tokens_sample & set(combined.split()) or any(t in combined for t in tokens_sample):
                snippets.append(f"Service: {name} — {desc}")

    faqs_root = data.get("faqs", {}) or {}
    for entry in (faqs_root.get("rag_formatted") or []):
        entry_text = _extract_text_from_rag_entry(entry).lower()
        if tokens_sample & set(entry_text.split()) or any(t in entry_text for t in tokens_sample):
            preview = entry_text.replace("\n", " ").strip()[:600]
            snippets.append(f"FAQ: {preview}")

    for entry in (faqs_root.get("plain") or faqs_root.get("qa") or []):
        q = entry.get("q") or entry.get("question") or ""
        a = entry.get("a") or entry.get("answer") or ""
        combined = f"{q} {a}".lower()
        if tokens_sample & set(combined.split()) or any(t in combined for t in tokens_sample):
            snippets.append(f"FAQ: Q: {q} A: {a}")

    sm = data.get("sales_material", {}) or {}
    hero = sm.get("hero_headline","")
    pitch = sm.get("elevator_pitch","")
    if hero or pitch:
        combined = f"{hero} {pitch}".lower()
        if tokens_sample & set(combined.split()) or any(t in combined for t in tokens_sample):
            snippets.append(f"Sales: {hero} — {pitch}")

    # dedupe & clamp
    seen = set()
    out = []
    for s in snippets:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= max_chunks:
            break

    if not out:
        company = data.get("company", {}) or {}
        summary = company.get("short_bio") or company.get("about") or company.get("business_activity") or ""
        if summary:
            out.append(f"Company summary: {text_tokens_preview(summary, 60)}")

    return out

COMPANY_INFO_RE = re.compile(
    r"\b(kab|kab shuru|founder|owner|who founded|owner kaun|establish|incorporat|cin|employees|kitne employees|headquarter|hq|where located|incorporation|founded)\b",
    re.I
)

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
    ct = request.headers.get("content-type", "")
    message_text, session_id = "", None

    if "multipart/form-data" in ct:
        raise HTTPException(status_code=400, detail="File uploads are disabled on this deployment. Use demo project for file uploads.")

    body = await request.json()
    message_text = (body.get("message") or body.get("text") or "").strip()
    session_id = body.get("session_id") or None

    conv = db.upsert_conversation(session_id)
    conversation_id, session_id = conv["id"], conv["session_id"]

    db.save_message(conversation_id, role="user", text=message_text, file_url=None)

    recent_history = db.get_last_messages(conversation_id, limit=20) or []
    assistant_has_greeted = any(m.get("role") == "assistant" and "Namaste" in (m.get("text") or "") for m in recent_history)

    use_hinglish = False
    if re.search(r"\b(hindi|hinglish|हिंदी|हिंग्लिश|bol in hindi|bol hindi|हेलो हिंदी)\b", message_text, re.I):
        use_hinglish = True

    # ---------------------------
    # Improved greeting / branching logic
    # ---------------------------
    def is_short_greeting(txt: str) -> bool:
        if not txt:
            return False
        t = txt.lower().strip()
        greetings = ["hi", "hello", "hey", "hiya", "yo", "good morning", "good evening", "hello ji", "hello ji!"]
        if any(t == g or t.startswith(g + " ") or t.endswith(" " + g) for g in greetings):
            if len(t.split()) <= 4:
                return True
        return False

    def assistant_asked_choice(history) -> bool:
        for m in reversed(history):
            if m.get("role") == "assistant":
                txt = (m.get("text") or "").lower()
                if "would you like" in txt and ("overview" in txt or "talk about your business needs" in txt or "reply with 'overview' or 'needs'" in txt):
                    return True
        return False

    def interpret_choice_reply(txt: str) -> str:
        t = (txt or "").lower()
        if any(w in t for w in ["service", "services", "overview", "explain", "what do you offer", "show me"]):
            return "SERVICES"
        if any(w in t for w in ["need", "automate", "problem", "project", "help", "use case", "business", "we want"]):
            return "NEEDS"
        if any(w in t for w in ["demo", "call", "schedule", "meeting"]):
            return "DEMO"
        return "UNKNOWN"

    # 1) FIRST short greeting (assistant not greeted) -> short natural reply only
    if is_short_greeting(message_text) and not assistant_has_greeted and not assistant_asked_choice(recent_history):
        short_greeting = "Hey 👋 — I’m Fynorra AI. How can I assist you today?"
        db.save_message(conversation_id, role="assistant", text=short_greeting, file_url=None)
        return JSONResponse({"reply": short_greeting, "session_id": session_id, "is_lead": False, "lead": None})

    # 2) If user greets again after assistant greeted -> show choice prompt
    if is_short_greeting(message_text) and assistant_has_greeted and not assistant_asked_choice(recent_history):
        choice_prompt = (
            "Hey again 👋 — would you like a quick **overview of our services** or should I ask a couple of questions about your business needs? "
            "Reply with 'Overview' or 'Needs' (or say 'Demo' to schedule a call)."
        )
        db.save_message(conversation_id, role="assistant", text=choice_prompt, file_url=None)
        return JSONResponse({"reply": choice_prompt, "session_id": session_id, "is_lead": False, "lead": None})

    # 3) If assistant previously asked the choice, interpret reply and branch
    if assistant_asked_choice(recent_history):
        intent = interpret_choice_reply(message_text)
        if intent == "SERVICES":
            services_overview = (
                "We provide: AI Chatbots (website & WhatsApp), RAG-based assistants, Document OCR & Automation, "
                "CRM integrations, AI Content Pipelines, Voice/IVR, Dashboards & Predictive Analytics, and Custom Copilots. "
                "Which of these interests you most — or shall I suggest based on your industry?"
            )
            db.save_message(conversation_id, role="assistant", text=services_overview, file_url=None)
            return JSONResponse({"reply": services_overview, "session_id": session_id, "is_lead": False, "lead": None})
        if intent == "NEEDS":
            discovery = (
                "Great — to recommend the right solution I need a couple quick details:\n"
                "1) What primary business process or challenge are you looking to automate? \n"
                "2) Which industry are you in? \n"
                "3) Do you have a target timeline or budget range?\n\n"
                "Reply with short answers and I’ll suggest the best option and next step."
            )
            db.save_message(conversation_id, role="assistant", text=discovery, file_url=None)
            return JSONResponse({"reply": discovery, "session_id": session_id, "is_lead": False, "lead": None})
        if intent == "DEMO":
            demo_msg = "Sure — I can schedule a 20-min demo. Please share your preferred day/time and contact email/phone, or reply 'Call me' and our team will reach out."
            db.save_message(conversation_id, role="assistant", text=demo_msg, file_url=None)
            return JSONResponse({"reply": demo_msg, "session_id": session_id, "is_lead": True, "lead": None})
        reprompt = "Do you want a quick overview of our services, or shall I ask about your business needs? Reply 'Overview' or 'Needs'."
        db.save_message(conversation_id, role="assistant", text=reprompt, file_url=None)
        return JSONResponse({"reply": reprompt, "session_id": session_id, "is_lead": False, "lead": None})

    # -----------------------
    # Normal flow (retrieval + LLM)
    # -----------------------
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

    if not sources_parts or COMPANY_INFO_RE.search(message_text):
        try:
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
                chunks = find_relevant_chunks(message_text)
                for c in chunks:
                    sources_parts.append(f"[LOCAL] {c}")
                if not sources_parts:
                    master = load_master_context()
                    company_summary = master.get("company", {}).get("short_bio") or master.get("company", {}).get("about") or ""
                    if company_summary:
                        sources_parts.append(f"[LOCAL] Company summary: {text_tokens_preview(company_summary, 120)}")
        except Exception as e:
            logger.exception("Fallback retrieval error: %s", e)

    sources_text = ("\n\n--- SOURCE ---\n\n".join(sources_parts)).strip() if sources_parts else ""
    history = recent_history[:6] if recent_history else []
    history_text = "\n".join([f"{m['role']}: {m['text']}" for m in history]) if history else ""
    lang_pref = "Prefer English in responses. Switch to Hinglish only if user asks." if not use_hinglish else "Use Hinglish for responses."
    system_prompt = BASE_PERSONA + "\n" + lang_pref
    final_context = "\n".join([sources_text, history_text]).strip() if (sources_text or history_text) else sources_text or history_text

    try:
        reply = llm_handler.get_llm_response(
            system_prompt=system_prompt,
            context=final_context,
            user_question=message_text,
            model=os.environ.get("LLM_MODEL", None),
            request_type="chat",
        )
    except Exception as e:
        logger.exception("LLM error (full): %s", e)
        msg = str(e)
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {msg[:500]}")

    # Prepend canonical greeting once if not present and assistant hasn't greeted
    greeting = "Namaste 🙏, I’m Fynorra AI — your AI automation partner. How can I help you today?\n\n"
    if not assistant_has_greeted and not re.search(r"\b(namaste|nice to connect|would you like|overview|how can i help|reply with|'overview'|'needs')\b", (reply or "").lower()):
        reply = greeting + reply

    db.save_message(conversation_id, role="assistant", text=reply, file_url=None)

    # Lead detection (unchanged)
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

    if contact_email_present or contact_phone_present:
        score_combined = max(score_combined, 0.98)

    LEAD_THRESHOLD = float(os.environ.get("LEAD_THRESHOLD", 0.75))
    is_lead = score_combined >= LEAD_THRESHOLD

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
