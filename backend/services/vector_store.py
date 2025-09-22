# backend/services/vector_store.py
import os
import json
import uuid
import threading
import io
from typing import List, Dict, Optional

import requests
import numpy as np

# chroma (lightweight local vector DB)
import chromadb
from chromadb.config import Settings

# PDF / OCR helpers (optional)
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from PIL import Image
    import pytesseract
except Exception:
    Image = None
    pytesseract = None

# --- Config ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VECTOR_DIR = os.environ.get("VECTOR_DIR", os.path.join(BASE_DIR, "..", "vector_store"))
os.makedirs(VECTOR_DIR, exist_ok=True)

CHROMA_PERSIST_DIR = os.path.join(VECTOR_DIR, "chroma_db")
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "fynorra_docs")

HF_API_TOKEN = os.environ.get("HF_API_TOKEN")  # required for HF Inference API
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_EMBED_URL = os.environ.get("HF_EMBED_URL", f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBED_MODEL}")

# --- Chroma client (local, duckdb+parquet) ---
client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_PERSIST_DIR))
# ensure collection
try:
    collection = client.get_collection(CHROMA_COLLECTION)
except Exception:
    collection = client.create_collection(CHROMA_COLLECTION)

_lock = threading.Lock()

# ---------------- Embedding utilities ----------------
def _hf_embed_texts(texts: List[str], batch_size: int = 8) -> List[List[float]]:
    """
    Use Hugging Face Inference API (feature-extraction pipeline) to get embeddings.
    Returns list of vectors (floats). No local torch required.
    """
    if not HF_API_TOKEN:
        raise ValueError("HF_API_TOKEN environment variable not set (needed for HF inference).")

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": "application/json"}
    embeddings: List[List[float]] = []

    # HF pipeline can accept a single string or a list; to be safe we'll batch requests
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # HF inference pipeline expects either a single string or a list for some endpoints
        # We'll POST the batch as JSON array
        resp = requests.post(HF_EMBED_URL, headers=headers, json=batch, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # data could be a list of vectors, or list of list-of-token-vectors (rare)
        for item in data:
            if isinstance(item, list) and len(item) > 0 and isinstance(item[0], list):
                # token-level vectors -> average them
                vec = np.mean(np.array(item, dtype=np.float32), axis=0).astype(float).tolist()
            else:
                vec = [float(x) for x in item]
            embeddings.append(vec)

    return embeddings

def _normalize_vectors(vs: List[List[float]]) -> List[List[float]]:
    arr = np.array(vs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms
    return arr.astype(float).tolist()

# ---------------- Chroma operations ----------------
def upsert_chunks(chunks: List[str], metadata: Dict):
    """
    Insert chunks list into Chroma with metadata.
    metadata is attached per-chunk (source, conversation_id, meta dict)
    """
    if not chunks:
        return []

    ids = [str(uuid.uuid4()) for _ in chunks]
    metas = []
    for _ in chunks:
        metas.append({
            "source": metadata.get("source"),
            "conversation_id": metadata.get("conversation_id"),
            "meta": metadata.get("meta", {})
        })

    with _lock:
        # get embeddings from HF
        embs = _hf_embed_texts(chunks)
        embs = _normalize_vectors(embs)
        # add to chroma
        collection.add(ids=ids, documents=chunks, metadatas=metas, embeddings=embs)
        try:
            client.persist()
        except Exception:
            # persist may fail in some deployments; ignore but log if needed
            pass

    return ids

def search(query: str, top_k: int = 4) -> List[Dict]:
    """
    Query Chroma with an embedding of the query.
    Returns list of {source, text, meta, score}
    """
    if not query:
        return []

    with _lock:
        q_emb = _hf_embed_texts([query])[0]
        q_emb = _normalize_vectors([q_emb])[0]
        # query chroma
        results = collection.query(query_embeddings=[q_emb], n_results=top_k, include=["metadatas", "documents", "distances"])
        # results format: dict with 'ids', 'documents', 'metadatas', 'distances'
        out = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            out.append({
                "source": meta.get("source"),
                "text": doc,
                "meta": meta.get("meta"),
                "score": float(dist)
            })
        return out

# ---------------- PDF & OCR helpers (optional, keep as before) ----------------
def extract_text_from_pdf_bytes(data: bytes) -> str:
    if not fitz:
        raise RuntimeError("PyMuPDF (fitz) not installed.")
    doc = fitz.open(stream=data, filetype="pdf")
    parts = []
    for page in doc:
        parts.append(page.get_text())
    doc.close()
    return "\n".join(parts)

def parse_pdf(path_or_bytes):
    """
    Accept either file path or raw bytes. Returns extracted text.
    """
    try:
        # If given bytes
        if isinstance(path_or_bytes, (bytes, bytearray)):
            return extract_text_from_pdf_bytes(path_or_bytes)
        # If given path string
        if isinstance(path_or_bytes, str) and os.path.exists(path_or_bytes):
            with open(path_or_bytes, "rb") as f:
                data = f.read()
            return extract_text_from_pdf_bytes(data)
        raise ValueError("parse_pdf expects file path or bytes")
    except Exception as e:
        # bubble up to caller
        raise

def ocr_image(path_or_bytes):
    """
    Basic OCR wrapper: if pytesseract available, run OCR.
    Returns extracted text (string).
    """
    if not Image or not pytesseract:
        return ""  # optional: OCR not available
    if isinstance(path_or_bytes, (bytes, bytearray)):
        img = Image.open(io.BytesIO(path_or_bytes))
    elif isinstance(path_or_bytes, str) and os.path.exists(path_or_bytes):
        img = Image.open(path_or_bytes)
    else:
        raise ValueError("ocr_image expects file path or bytes")
    text = pytesseract.image_to_string(img)
    return text
