# backend/services/pdf_processor.py
"""
PDF processor: extract text, chunk, and index into vector store.

Provides:
- ingest_pdf_bytes(data: bytes, meta: dict) -> dict
- ingest_pdf_file(path: str, meta: dict) -> dict
- chunk_text(text, chunk_size=800, overlap=100)
- FastAPI file upload example endpoint (commented at bottom)

Design choices:
- Default chunk_size=800 words, overlap=100 words -> balances context vs vector granularity.
- Tokenization is simple whitespace-based (fast & no extra deps). If you later use OpenAI,
  you can replace with tiktoken-based token splits for more accurate chunking.
"""
from typing import List, Dict, Optional
import os
import io
import math
import uuid
import logging

# local vector store (expects replaced, HF+Chroma version)
from .vector_store import parse_pdf, ocr_image, upsert_chunks

logger = logging.getLogger("pdf_processor")
logger.setLevel(logging.INFO)

# ---------- chunking utilities ----------
def chunk_text_by_words(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    Splits text into chunks by words.
    - chunk_size: approx number of words per chunk
    - overlap: words overlapped between consecutive chunks
    Returns list of chunk strings.
    """
    if not text:
        return []

    words = text.split()
    if chunk_size <= 0:
        return [" ".join(words)]

    chunks = []
    i = 0
    n = len(words)
    while i < n:
        end = min(i + chunk_size, n)
        chunk = " ".join(words[i:end])
        chunks.append(chunk)
        if end == n:
            break
        i = end - overlap  # move by chunk_size - overlap
        if i < 0:
            i = 0
    return chunks

def sanitize_meta(meta: Optional[Dict]) -> Dict:
    """Ensure meta is a dict with expected keys."""
    if not meta:
        return {}
    return dict(meta)

# ---------- ingestion pipeline ----------
def ingest_pdf_bytes(data: bytes, metadata: Optional[Dict] = None,
                     chunk_size: int = 800, overlap: int = 100) -> Dict:
    """
    Ingest raw PDF bytes:
    1. Extract text using parse_pdf (PyMuPDF) or raise if missing.
    2. If extraction empty and OCR available, run OCR on PDF pages converted to images (best-effort).
    3. Chunk text and upsert to vector store.
    Returns: {"inserted": N, "ids": [...], "chunks": count}
    """
    metadata = sanitize_meta(metadata)
    text = ""
    try:
        text = parse_pdf(data)
    except Exception as e:
        # parse_pdf expected to raise if parsing failed
        logger.warning("parse_pdf failed: %s", e)
        # fallback: try OCR (if available)
        try:
            ocr_text = ocr_image(data)
            if ocr_text:
                text = ocr_text
        except Exception as e2:
            logger.warning("ocr_image failed: %s", e2)

    if not text or not text.strip():
        return {"inserted": 0, "ids": [], "chunks": 0, "error": "no text extracted"}

    chunks = chunk_text_by_words(text, chunk_size=chunk_size, overlap=overlap)
    # attach doc-level metadata (e.g., source, original filename, etc.)
    doc_meta = {
        "source": metadata.get("source", "uploaded_pdf"),
        "conversation_id": metadata.get("conversation_id"),
        "meta": metadata.get("meta", {}),
    }

    ids = upsert_chunks(chunks, doc_meta)
    return {"inserted": len(ids), "ids": ids, "chunks": len(chunks)}

def ingest_pdf_file(path: str, metadata: Optional[Dict] = None,
                    chunk_size: int = 800, overlap: int = 100) -> Dict:
    """
    Ingest PDF from disk path.
    """
    if not os.path.exists(path):
        return {"inserted": 0, "ids": [], "chunks": 0, "error": "file not found"}
    with open(path, "rb") as f:
        data = f.read()
    return ingest_pdf_bytes(data, metadata=metadata, chunk_size=chunk_size, overlap=overlap)

# ---------- convenience helper: process and return top-k retrieval ----------
def ingest_and_query_pdf_bytes(data: bytes, query: str,
                               metadata: Optional[Dict] = None,
                               chunk_size: int = 800, overlap: int = 100,
                               top_k: int = 4) -> Dict:
    """
    Ingest PDF bytes and run a similarity query immediately.
    Useful for quick feedback/testing.
    Returns: {"inserted": N, "top_results": [..]}
    """
    ingest_res = ingest_pdf_bytes(data, metadata=metadata, chunk_size=chunk_size, overlap=overlap)
    if ingest_res.get("inserted", 0) == 0:
        return {"inserted": 0, "top_results": [], "error": ingest_res.get("error")}
    # query using vector_store.search (search is exported from vector_store)
    # Import locally to avoid circular imports at module load
    from .vector_store import search
    top = search(query, top_k=top_k)
    return {"inserted": ingest_res["inserted"], "top_results": top}

# ---------- FastAPI endpoint example (copy into your router) ----------
FAKE_ENDPOINT_SNIPPET = """
# Example FastAPI POST endpoint (in your rag_router or relevant router)
from fastapi import APIRouter, UploadFile, File, Form
from .services.pdf_processor import ingest_pdf_bytes

router = APIRouter()

@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), conversation_id: str = Form(None)):
    data = await file.read()
    meta = {"source": file.filename, "conversation_id": conversation_id}
    result = ingest_pdf_bytes(data, metadata=meta)
    return result
"""

# ---------- module test helper (local) ----------
if __name__ == "__main__":  # quick local test
    logging.basicConfig(level=logging.INFO)
    sample_path = "test.pdf"
    if os.path.exists(sample_path):
        res = ingest_pdf_file(sample_path, metadata={"source": sample_path})
        print("Ingest result:", res)
    else:
        print("Place a 'test.pdf' in cwd to test this script.")
