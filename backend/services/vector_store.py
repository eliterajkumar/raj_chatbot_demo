from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

def create_vector_store(chunks: List[str]) -> Dict:
    """
    Creates an in-memory vector store from text chunks using TF-IDF.
    Returns a dictionary containing the vectorizer, documents, and the index.
    """
    if not chunks:
        raise ValueError("Input chunks cannot be empty.")
    
    vectorizer = TfidfVectorizer()
    vector_index = vectorizer.fit_transform(chunks)
    
    return {
        "vectorizer": vectorizer,
        "documents": chunks,
        "index": vector_index
    }

def find_similar_chunks(store: Dict, query: str, top_k: int = 3) -> List[str]:
    """Finds the top_k most similar chunks to a query from the vector store."""
    if not all(k in store for k in ["vectorizer", "documents", "index"]):
        raise TypeError("Provided store is not a valid vector store.")
        
    query_vector = store["vectorizer"].transform([query])
    similarities = cosine_similarity(query_vector, store["index"]).flatten()
    
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    return [store["documents"][i] for i in top_indices] 