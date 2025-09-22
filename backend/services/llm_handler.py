# backend/services/llm_handler.py
import os
from typing import Optional
from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL = "openai/gpt-oss-120b"

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set.")

# Setup OpenRouter client via OpenAI SDK
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)

def get_llm_response(
    system_prompt: str,
    context: str,
    user_question: str,
    model: Optional[str] = None,
    request_type: str = "chat",  # chat | pdf | summary
) -> str:
    """
    Get LLM response via OpenRouter (OpenAI SDK).
    Dynamic max_tokens by request_type.
    """
    model_to_use = model or DEFAULT_MODEL

    # Dynamic token rules
    if request_type == "chat":
        max_tokens = 500
    elif request_type == "pdf":
        max_tokens = 1000
    elif request_type == "summary":
        max_tokens = 2000
    else:
        max_tokens = 500  # fallback

    # Construct user message with optional context
    user_message = (
        f"Context:\n---\n{context}\n---\n\nQuestion: {user_question}"
        if context else user_question
    )

    completion = client.chat.completions.create(
        model=model_to_use,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max_tokens,
        temperature=0.7,
        extra_headers={
            "HTTP-Referer": "https://fynorra.com",   # optional
            "X-Title": "Fynorra AI Assistant",       # optional
        },
    )

    return completion.choices[0].message.content.strip()
