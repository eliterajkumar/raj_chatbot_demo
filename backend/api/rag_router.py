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
                  Path("/context/fynorra_master_with_faqs.json"), Path("/context/fynorra_master_combined.json")]
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
    hero = sm.get("hero_headline", "")
    pitch = sm.get("elevator_pitch", "")
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


# COMPANY_INFO_RE updated to English (broader)
COMPANY_INFO_RE = re.compile(
    r"\b(?:"
    r"founder|founders|who founded|who is the founder|owner|owners|"
    r"when (?:was|were) (?:the )?(?:company|you) (?:founded|established)|founded in|established in|"
    r"incorporat(?:ed|ion)?|inc\b|ltd\b|llc\b|pvt(?:\.?\s*ltd)?|registered (?:in|under)|registration number|"
    r"company (?:number|no\.?)|CIN|company identification number|address|location|"
    r"headquarter(?:s)?|headquartered|hq\b|where (?:is|are) (?:the )?(?:company|you) (?:located|based)|"
    r"employees|staff|team size|number of employees"
    r")\b",
    re.I,
)


# Persona: tighten instruction to LLM to answer directly without prefaces.
BASE_PERSONA = (
    "You are Fynorra AI Assistant — the official AI representative of Fynorra AI Solutions. "
    "Tone: professional, concise, factual, and helpful. Answer directly to the user's question and do NOT add additional greetings, lead-ins or marketing 'feed' text. "
    "If the user message is a greeting (hi/hello), reply with a one-line greeting only. For all other queries, produce a direct, short answer (1-3 sentences) strictly based on provided company context. "
    "Rules:\n"
    "1) Default language: English. Only switch to Hinglish if the user explicitly requests it.\n"
    "2) Do not invent facts. If info is missing, say: 'I don't have that information right now; would you like me to connect you with our team?'.\n"
    "3) When asked about services, list matching services only (1-line each), no extra marketing blurbs.\n"
    "4) End with a clear Direct Action Step (single sentence) only when it is helpful (e.g., 'Reply \"Demo\" to schedule a 20-min demo').\n"
)


def _is_short_greeting(txt: str) -> bool:
    if not txt:
        return False
    t = txt.lower().strip()
    greetings = ["hi", "hello", "hey", "hiya", "yo", "good morning", "good evening", "hello ji", "namaste", "namaste ji"]
    if any(t == g or t.startswith(g + " ") or t.endswith(" " + g) for g in greetings):
        if len(t.split()) <= 4:
            return True
    return False


def _clean_reply(reply: str, user_message: str, allow_greeting: bool) -> str:
    """
    Post-process LLM reply:
     - If user didn't send a greeting, remove leading common greetings from reply.
     - Strip repeated assistant feed-like sentences such as 'How can I help you today?' at top.
     - Ensure reply is concise (trim long repetitive leading paragraphs).
    """
    if not reply:
        return reply or ""

    r = reply.strip()

    # remove leading greetings unless we explicitly allow greeting
    if not allow_greeting:
        # common greeting starters to strip
        r = re.sub(r'^\s*(namaste[^\n]*[\n]*)', '', r, flags=re.I)
        r = re.sub(r'^\s*(hi[^\n]*[\n]*)', '', r, flags=re.I)
        r = re.sub(r'^\s*(hello[^\n]*[\n]*)', '', r, flags=re.I)
        r = re.sub(r'^\s*(hey[^\n]*[\n]*)', '', r, flags=re.I)

        # remove stock lead-ins like "How can I help you today?" if at start
        r = re.sub(r'^\s*(how can i (help|assist) you (today)?[.?!]*[\n]*)', '', r, flags=re.I)
        r = re.sub(r'^\s*(how may I help you[.?!]*[\n]*)', '', r, flags=re.I)

    # trim excessive whitespace and repeated newlines
    r = re.sub(r'\n{3,}', '\n\n', r).strip()

    return r


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
    assistant_has_greeted = any(m.get("role") == "assistant" and re.search(r"\b(namaste|hi|hello)\b", (m.get("text") or ""), re.I) for m in recent_history)

    use_hinglish = False
    if re.search(r"\b(hindi|hinglish|हिंदी|हिंग्लिश|bol in hindi|bol hindi|हेलो हिंदी)\b", message_text, re.I):
        use_hinglish = True

    # ---------------------------
    # Greeting / choice logic (kept, but now explicit and minimal)
    # ---------------------------
    # 1) If user sends a short greeting and assistant hasn't greeted -> reply with one-line greeting only
    if _is_short_greeting(message_text) and not assistant_has_greeted:
        short_greeting = "Namaste 🙏 — I’m Fynorra AI — your AI automation partner."
        db.save_message(conversation_id, role="assistant", text=short_greeting, file_url=None)
        return JSONResponse({"reply": short_greeting, "session_id": session_id, "is_lead": False, "lead": None})

    # 2) If assistant earlier asked a choice, branch (keeps behavior)
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

    if assistant_asked_choice(recent_history):
        intent = interpret_choice_reply(message_text)
        if intent == "SERVICES":
            services_overview = (
                "AI Chatbots (Website & WhatsApp): conversational assistants for FAQs & lead capture.\n"
                "RAG Assistants: document-backed Q&A for product & policy docs.\n"
                "Document OCR & Automation: extract data and automate workflows.\n"
                "CRM & Integrations: sync leads, tickets, and workflows.\n"
                "Dashboards & Analytics: insights and predictive metrics.\n"
                "Reply with the service name you'd like details on, or 'Demo' to schedule a call."
            )
            db.save_message(conversation_id, role="assistant", text=services_overview, file_url=None)
            return JSONResponse({"reply": services_overview, "session_id": session_id, "is_lead": False, "lead": None})
        if intent == "NEEDS":
            discovery = (
                "To suggest the right solution, please share: (1) the process you want to automate, (2) your industry, and (3) rough timeline/budget."
            )
            db.save_message(conversation_id, role="assistant", text=discovery, file_url=None)
            return JSONResponse({"reply": discovery, "session_id": session_id, "is_lead": False, "lead": None})
        if intent == "DEMO":
            demo_msg = "Sure — to schedule a 20-min demo, please share a preferred date/time and contact email/phone."
            db.save_message(conversation_id, role="assistant", text=demo_msg, file_url=None)
            return JSONResponse({"reply": demo_msg, "session_id": session_id, "is_lead": True, "lead": None})
        reprompt = "Reply 'Overview' or 'Needs' — I can give a short overview of services or ask a few questions about your needs."
        db.save_message(conversation_id, role="assistant", text=reprompt, file_url=None)
        return JSONResponse({"reply": reprompt, "session_id": session_id, "is_lead": False, "lead": None})

    # -----------------------
    # Normal flow (retrieval + LLM) — build concise context
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
            meta_name = (h.get("metadata") or {}).get("service_name") or h.get("id") or h.get("metadata", {}).get("title")
            text = h.get("text") or h.get("snippet") or ""
            sources_parts.append(f"[VECTOR-HIT] {meta_name}\n{text[:1500]}")
    except Exception as e:
        logger.debug("Vector search failed: %s", e)
        hits = []

    # If no vector hits or the question is about company identity, use local master context
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
    history = recent_history[-6:] if recent_history else []
    # include only minimal recent history (role + text) to preserve context but avoid feed repetition
    history_text = "\n".join([f"{m['role']}: {m['text']}" for m in history]) if history else ""
    lang_pref = "Prefer English in responses. Switch to Hinglish only if user asks." if not use_hinglish else "Use Hinglish for responses."
    system_prompt = BASE_PERSONA + "\n" + lang_pref

    # final_context: keep short, only sou_
