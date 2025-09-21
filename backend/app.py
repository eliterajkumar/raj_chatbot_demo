# app.py  (place at repo root)
"""
Deployment entrypoint wrapper for OpenShift / Knative / Render.
Keeps your existing backend/main.py unchanged.
"""

import os

# Import the FastAPI ASGI app object. Change this path only if your app lives somewhere else.
from backend.main import app  # <- ensure backend/main.py defines `app`

# Optional: provide a simple CLI for local testing
if __name__ == "__main__":
    # Use PORT from environment (OpenShift sets a PORT), fallback to 8080 locally.
    port = int(os.environ.get("PORT", 8080))
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, log_level="info")
