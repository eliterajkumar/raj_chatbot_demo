# backend/api/rag_router.py
from fastapi import APIRouter, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse
import uuid, os, shutil, re

from ..services import pdf_processor, vector_store, llm_handler, db

router = APIRouter()

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/tmp/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_upload_file_tmp(upload_file: UploadFile) -> str:
    ext = os.path.splitext(upload_file.filename)[1]
    tmp_name = f"{uuid.uuid4().hex}{ext}"
    tmp_path = os.path.join(UPLOAD_DIR, tmp_name)
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return tmp_path


@router.post("/chat")
async def chat_endpoint(request: Request):
    """
    Chat endpoint for RAG assistant:
    - JSON body: {"message":"...", "session_id":"..."}
    - Multipart: message + optional file
    """
    ct = request.headers.get("content-type", "")
    message_text, session_id, file_path, file_url, file_kind = "", None, None, None, None

    # Handle input
    if "multipart/form-data" in ct:
        form = await request.form()
        message_text = form.get("message") or ""
        session_id = form.get("session_id") or None
        upload = form.get("file")
        if upload:
            file_path = save_upload_file_tmp(upload)
            file_kind = upload.content_type
    else:
        body = await request.json()
        message_text = body.get("message") or body.get("text") or ""
        session_id = body.get("session_id") or None

    # Ensure conversation
    conv = db.upsert_conversation(session_id)
    conversation_id, session_id = conv["id"], conv["session_id"]

    # Save user message
    db.save_message(conversation_id, role="user", text=message_text, file_url=None)

    # File parsing (PDF/Image OCR)
    extracted_text = ""
    if file_path:
        try:
            if file_kind == "application/pdf" or file_path.lower().endswith(".pdf"):
                extracted_text = pdf_processor.parse_pdf(file_path)
            else:
                extracted_text = pdf_processor.ocr_image(file_path)

            file_url = db.save_file_to_storage(file_path, conversation_id=conversation_id)
            chunks = pdf_processor.chunk_text(extracted_text)
            vector_store.upsert_chunks(chunks, metadata={"source": os.path.basename(file_path), "conversation_id": conversation_id})
        except Exception as e:
            print("File parse error:", e)

    # Retrieve knowledge (RAG)
    retrieved = vector_store.search(message_text, top_k=4) if message_text else []

    # Persona / system prompt (polished)
    persona = (
        "You are Fynorra’s friendly Sales Assistant. "
        "Greet the user once at the start of a session with 'Namaste 🙏' and a one-line intro, "
        "then avoid repeating the greeting in subsequent replies. "
        "Speak politely in Hinglish (mix Hindi + English). "
        "Explain Fynorra services (AI Chatbots, Automation, IT Consulting) concisely, be persuasive but never pushy, "
        "ask one qualifying question when interest is detected, and propose next step (demo/contact). "
        "When a file has been uploaded, mention that you have received it and wait for the user to request analysis before summarizing in detail."
    )

    # Conversation history
    history = db.get_last_messages(conversation_id, limit=6)
    history_text = "\n".join([f"{m['role']}: {m['text']}" for m in history])

    # RAG context
    sources_text = ""
    if retrieved:
        for i, r in enumerate(retrieved):
            sources_text += f"\n--- Source {i+1}: ({r.get('source')})\n{r.get('text')[:1000]}"
    if extracted_text:
        sources_text += f"\n--- Uploaded file content:\n{extracted_text[:2000]}"

    # LLM call
    try:
        reply = llm_handler.get_llm_response(
            system_prompt=persona,
            context=sources_text + "\n" + history_text,
            user_question=message_text,
            model=os.environ.get("LLM_MODEL", "openai/gpt-4o"),
        )
    except Exception as e:
        print("LLM error (full):", repr(e))
        msg = str(e)
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {msg[:500]}")

    # Save assistant reply
    db.save_message(conversation_id, role="assistant", text=reply, file_url=file_url)

    # Lead detection
    interest_regex = r"(interested|demo|price|cost|quote|proposal|contact|signup|buy|purchase|meeting|schedule)"
    is_lead = bool(re.search(interest_regex, message_text + " " + reply, flags=re.IGNORECASE))
    lead = None
    if is_lead:
        lead = db.create_lead(conversation_id, snippet=message_text[:500], score=0.7, metadata={"file_url": file_url})
        try:
            db.notify_sales(lead)
        except Exception as e:
            print("Notify sales failed:", e)

    # Clean up temp file
    if file_path and os.path.exists(file_path):
        try: os.remove(file_path)
        except: pass

    return JSONResponse({
        "reply": reply,
        "session_id": session_id,
        "is_lead": is_lead,
        "lead": lead
    })
