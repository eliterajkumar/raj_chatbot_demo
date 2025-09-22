# backend/services/vector_store.py
import os
import json
import uuid
import threading
from typing import List, Dict, Optional

# embeddings
from sentence_transformers import SentenceTransformer
import numpy as np

# faiss
import faiss

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VECTOR_DIR = os.environ.get("VECTOR_DIR", os.path.join(BASE_DIR, "..", "vector_store"))
os.makedirs(VECTOR_DIR, exist_ok=True)
VECTOR_INDEX_PATH = os.path.join(VECTOR_DIR, "faiss.index")
VECTOR_META_PATH = os.path.join(VECTOR_DIR, "metadata.json")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

_lock = threading.Lock()

# Load model once
_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
_dim = _model.get_sentence_embedding_dimension()

# In-memory metadata list (parallel to FAISS IDs)
if os.path.exists(VECTOR_META_PATH):
    with open(VECTOR_META_PATH, "r", encoding="utf-8") as f:
        _metadata = json.load(f)
else:
    _metadata = []  # list of dicts containing {id, text, source, conversation_id, other_meta}

# Load or init FAISS index
if os.path.exists(VECTOR_INDEX_PATH):
    try:
        _index = faiss.read_index(VECTOR_INDEX_PATH)
        # Ensure dimension matches
        if _index.d != _dim:
            # rebuild index if mismatch
            _index = faiss.IndexFlatIP(_dim)
            # if there are existing metadata, re-embed? safer to start fresh
            _metadata = []
    except Exception as e:
        print("faiss load error:", e)
        _index = faiss.IndexFlatIP(_dim)
else:
    _index = faiss.IndexFlatIP(_dim)  # inner product (use normalized vectors)

# helper: normalize vectors for IP similarity
def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms

def upsert_chunks(chunks: List[str], metadata: Dict):
    """
    chunks: list of text chunks
    metadata: dict to attach to each chunk (e.g., {"source": "file.pdf", "conversation_id": "..."})
    """
    global _index, _metadata
    if not chunks:
        return

    with _lock:
        embeddings = _model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        embeddings = _normalize(embeddings.astype("float32"))
        # Add to index
        try:
            _index.add(embeddings)
            # append metadata entries for each embedding
            for i, chunk in enumerate(chunks):
                entry = {
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "source": metadata.get("source"),
                    "conversation_id": metadata.get("conversation_id"),
                    "meta": metadata.get("meta", {})
                }
                _metadata.append(entry)
            # persist
            faiss.write_index(_index, VECTOR_INDEX_PATH)
            with open(VECTOR_META_PATH, "w", encoding="utf-8") as f:
                json.dump(_metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("vector_store.upsert_chunks error:", e)
            raise

def search(query: str, top_k: int = 4) -> List[Dict]:
    """
    Returns list of {source, text, score, meta}
    """
    global _index, _metadata
    if not query or _index.ntotal == 0:
        return []

    with _lock:
        q_emb = _model.encode([query], convert_to_numpy=True)
        q_emb = _normalize(q_emb.astype("float32"))
        try:
            D, I = _index.search(q_emb, top_k)
            results = []
            for score, idx in zip(D[0].tolist(), I[0].tolist()):
                if idx < 0 or idx >= len(_metadata):
                    continue
                meta = _metadata[idx]
                results.append({
                    "source": meta.get("source"),
                    "text": meta.get("text"),
                    "meta": meta.get("meta"),
                    "score": float(score)
                })
            return results
        except Exception as e:
            print("vector_store.search error:", e)
            return []
