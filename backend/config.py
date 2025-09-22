import os

# Site config
API_KEY = os.environ.get("API_KEY", "dev-site-key")
SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", 600))  # 10 minutes
MAX_MEMORY_PERCENT = int(os.environ.get("MAX_MEMORY_PERCENT", 85))

# LLM / OpenRouter
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o")  # change as appropriate

# App
UPLOADS_ENABLED = False  # explicit guard for site-only version
