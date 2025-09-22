# backend/services/llm_handler.py
import os
import requests
from typing import List, Dict, Any

API_KEY = os.getenv("OPENROUTER_API_KEY")  # prefer OpenRouter; can reuse existing key
API_URL = os.getenv("OPENROUTER_API_URL", "https://api.openrouter.ai/v1/chat/completions")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")  # use a lightweight OpenRouter model by default

def call_llm_messages(messages: List[Dict[str, str]],
                      model: str = None,
                      max_tokens: int = 500,
                      temperature: float = 0.2) -> Dict[str, Any]:
    """
    Generic call to OpenRouter-compatible chat endpoint.
    messages: list of {"role": "system/user/assistant", "content": "..."}
    Returns parsed JSON response.
    """
    if not API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable not set.")

    model_to_use = model or DEFAULT_MODEL

    payload = {
        "model": model_to_use,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    r = requests.post(API_URL, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()

def get_llm_response(system_prompt: str, context: str, user_question: str,
                     model: str = None, request_type: str = "chat") -> str:
    """
    Higher-level wrapper to construct messages and return text content.
    request_type can be used to vary max_tokens.
    """
    if request_type == "chat":
        max_tokens = 500
    elif request_type == "pdf":
        max_tokens = 1000
    elif request_type == "summary":
        max_tokens = 2000
    else:
        max_tokens = 500

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    user_message = f"Context:\n---\n{context}\n---\n\nQuestion: {user_question}" if context else user_question
    messages.append({"role": "user", "content": user_message})

    data = call_llm_messages(messages, model=model, max_tokens=max_tokens, temperature=0.7)
    # Parse response text robustly based on provider shape
    # OpenRouter-like: data["choices"][0]["message"]["content"]
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        # fallback: try top-level 'output' or 'text'
        if isinstance(data.get("output"), str):
            return data["output"].strip()
        # last fallback: stringify
        return str(data)
