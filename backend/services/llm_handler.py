import os
import requests

API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

def get_llm_response(system_prompt: str, context: str, user_question: str, model: str = "mistralai/mistral-small-3.2-24b-instruct:free") -> str:
    """
    A unified function to get a response from an LLM via OpenRouter.
    Defaulting to a specific free model.
    """
    if not API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable not set.")

    # Construct the user message, including context only if it's provided
    user_message = f"Context:\n---\n{context}\n---\n\nQuestion: {user_question}" if context else user_question

    try:
        response = requests.post(
            url=API_URL,
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ]
            }
        )
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"LLM API request failed: {e}")
    except (KeyError, IndexError):
        raise ValueError("Failed to parse LLM API response.") 