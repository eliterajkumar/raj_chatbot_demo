# backend/lib/math_utils.py
from typing import List
import numpy as np

def normalize_vectors(vs: List[List[float]]) -> np.ndarray:
    """
    vs: list of vectors (list of floats)
    returns: 2D numpy array normalized (L2)
    """
    arr = np.array(vs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32)

def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: shape (m, d)
    b: shape (n, d)
    returns: (m, n) cosine similarity matrix
    """
    # assume rows are normalized already for max efficiency
    return np.dot(a, b.T)

def cosine_similarity(a_vec: List[float], b_vecs: List[List[float]]) -> List[float]:
    """
    a_vec: single vector
    b_vecs: list of vectors
    returns: list of cosine similarities
    """
    a = np.array(a_vec, dtype=np.float32)
    b = np.array(b_vecs, dtype=np.float32)
    a_norm = a / (np.linalg.norm(a) if np.linalg.norm(a) != 0 else 1.0)
    b_norms = np.linalg.norm(b, axis=1)
    b_norms[b_norms == 0] = 1.0
    b = b / b_norms[:, None]
    sims = np.dot(b, a_norm)
    return sims.tolist()
