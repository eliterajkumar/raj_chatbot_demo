# backend/services/llm_handler.py
import os
import requests

API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o")

def get_llm_response(
    system_prompt: str,
    context: str,
    user_question: str,
    model: str = None,
    request_type: str = "chat",  # chat | pdf | summary
) -> str:
    """
    Get LLM response via OpenRouter.
    Dynamically adjusts max_tokens based on use-case:
    - chat: 500
    - pdf: 1000
    - summary/analysis: 2000
    """
    if not API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable not set.")

    model_to_use = model or DEFAULT_MODEL

    # 🎯 Dynamic token rules
    if request_type == "chat":
        max_tokens = 500
    elif request_type == "pdf":
        max_tokens = 1000
    elif request_type == "summary":
        max_tokens = 2000
    else:
        max_tokens = 500  # fallback

    # Construct user message
    user_message = (
        f"Context:\n---\n{context}\n---\n\nQuestion: {user_question}"
        if context else user_question
    )

    try:
        response = requests.post(
            url=API_URL,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={
                "model": model_to_use,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.7,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"LLM API request failed: {e}")
    except (KeyError, IndexError):
        raise ValueError(f"Unexpected LLM response: {data}")
