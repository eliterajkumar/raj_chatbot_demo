# backend/services/vector_store.py
import os
import json
import uuid
import threading
import io
import logging
from typing import List, Dict, Optional

import requests
import numpy as np

# Attempt chromadb import lazily inside get_chroma_client to avoid import-time crashes
_chromadb_imported = True
try:
    import chromadb  # type: ignore
    from chromadb.config import Settings  # type: ignore
except Exception:
    chromadb = None  # type: ignore
    Settings = None  # type: ignore
    _chromadb_imported = False

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

logger = logging.getLogger("vector_store")
logger.setLevel(logging.INFO)

# --- Config ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VECTOR_DIR = os.environ.get("VECTOR_DIR", os.path.join(BASE_DIR, "..", "vector_store"))
os.makedirs(VECTOR_DIR, exist_ok=True)

CHROMA_PERSIST_DIR = os.path.join(VECTOR_DIR, "chroma_db")
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "fynorra_docs")

HF_API_TOKEN = os.environ.get("HF_API_TOKEN")  # required for HF Inference API
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_EMBED_URL = os.environ.get(
    "HF_EMBED_URL",
    f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBED_MODEL}",
)

_lock = threading.Lock()

# --- in-memory fallback store shape (used when Chroma unavailable) ---
# {"_in_memory": True, "ids": [...], "documents": [...], "metadatas": [...], "embeddings": [..]}
_in_memory_store: Optional[Dict] = None

# ---------------- Chroma client lazy init ----------------
_chroma_client = None
_collection = None


def get_chroma_client():
    """
    Lazily create and return (client, collection).
    If chromadb is missing or config incompatible, fallback to in-memory store.
    """
    global _chroma_client, _collection, _in_memory_store

    if _chroma_client is not None or _collection is not None or _in_memory_store is not None:
        # if _in_memory_store exists, return that as collection
        if _in_memory_store is not None:
            return None, _in_memory_store
        return _chroma_client, _collection

    # Try to initialize chromadb client if import succeeded
    if _chromadb_imported and chromadb is not None:
        try:
            # primary attempt: explicit Settings (duckdb+parquet)
            client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_PERSIST_DIR))
            logger.info("Chroma client initialized with duckdb+parquet persist.")
        except Exception as exc1:
            logger.warning("Primary Chroma init failed: %s. Trying fallback client()", exc1)
            try:
                client = chromadb.Client()  # let chromadb decide defaults
                logger.info("Chroma client initialized with chromadb.Client() fallback.")
            except Exception as exc2:
                logger.error("Chroma fallback client() failed: %s. Will use in-memory store.", exc2)
                client = None

        if client:
            try:
                # ensure collection exists
                try:
                    collection = client.get_collection(CHROMA_COLLECTION)
                except Exception:
                    collection = client.create_collection(CHROMA_COLLECTION)
                _chroma_client = client
                _collection = collection
                return _chroma_client, _collection
            except Exception as exc3:
                logger.error("Failed ensuring chroma collection: %s. Falling back to in-memory.", exc3)
                client = None

    # Final fallback -> in-memory store
    _in_memory_store = {"_in_memory": True, "ids": [], "documents": [], "metadatas": [], "embeddings": []}
    logger.info("Using in-memory vector store fallback (non-persistent).")
    return None, _in_memory_store


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

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = requests.post(HF_EMBED_URL, headers=headers, json=batch, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        for item in data:
            if isinstance(item, list) and len(item) > 0 and isinstance(item[0], list):
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


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: (d,) vector
    b: (n, d) matrix
    returns: (n,) similarity scores
    """
    if a.ndim == 1:
        a = a / (np.linalg.norm(a) if np.linalg.norm(a) != 0 else 1.0)
    b_norms = np.linalg.norm(b, axis=1)
    b_norms[b_norms == 0] = 1.0
    b = b / b_norms[:, None]
    sims = np.dot(b, a)
    return sims


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
        metas.append(
            {
                "source": metadata.get("source"),
                "conversation_id": metadata.get("conversation_id"),
                "meta": metadata.get("meta", {}),
            }
        )

    with _lock:
        # get embeddings from HF
        embs = _hf_embed_texts(chunks)
        embs = _normalize_vectors(embs)

        client, collection = get_chroma_client()

        # If using in-memory fallback:
        if collection and isinstance(collection, dict) and collection.get("_in_memory"):
            for i, txt in enumerate(chunks):
                collection["ids"].append(ids[i])
                collection["documents"].append(txt)
                collection["metadatas"].append(metas[i])
                collection["embeddings"].append(embs[i])
            return ids

        # Else operate with Chroma collection
        try:
            collection.add(ids=ids, documents=chunks, metadatas=metas, embeddings=embs)
            try:
                # persist if possible
                if client:
                    client.persist()
            except Exception:
                # persist failing shouldn't block
                logger.debug("client.persist() raised; ignoring for now.")
        except Exception as e:
            logger.exception("vector_store.upsert_chunks error: %s", e)
            raise

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

        client, collection = get_chroma_client()

        # in-memory fallback search (brute-force cosine)
        if collection and isinstance(collection, dict) and collection.get("_in_memory"):
            docs = collection.get("documents", [])
            metas = collection.get("metadatas", [])
            emb_matrix = np.array(collection.get("embeddings", []), dtype=np.float32)
            if emb_matrix.size == 0:
                return []
            sims = _cosine_sim(np.array(q_emb, dtype=np.float32), emb_matrix)
            idx_sorted = np.argsort(-sims)[:top_k]
            out = []
            for idx in idx_sorted:
                out.append(
                    {
                        "source": metas[idx].get("source"),
                        "text": docs[idx],
                        "meta": metas[idx].get("meta"),
                        "score": float(sims[idx]),
                    }
                )
            return out

        # else query chroma normally
        try:
            results = collection.query(query_embeddings=[q_emb], n_results=top_k, include=["metadatas", "documents", "distances"])
            out = []
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]
            for doc, meta, dist in zip(docs, metas, dists):
                out.append(
                    {
                        "source": meta.get("source"),
                        "text": doc,
                        "meta": meta.get("meta"),
                        "score": float(dist),
                    }
                )
            return out
        except Exception as e:
            logger.exception("vector_store.search error when querying Chroma: %s", e)
            return []


# ---------------- PDF & OCR helpers (optional) ----------------
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
    except Exception:
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
