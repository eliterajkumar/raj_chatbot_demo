# backend/api/admin_router.py
from fastapi import APIRouter, Request, Depends, HTTPException, Query
from typing import List, Dict
from datetime import datetime, timedelta
import sqlite3
from ..services import db as db_service

router = APIRouter()

def require_key(request: Request):
    from ..config import API_KEY
    key = request.headers.get("x-api-key")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

@router.get("/conversations", dependencies=[Depends(require_key)])
def list_conversations(limit: int = 50) -> List[Dict]:
    """
    List most recent conversations with last_activity.
    """
    conn = db_service._get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, session_id, title, created_at, last_activity FROM conversations ORDER BY last_activity DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

@router.get("/cleanup/preview", dependencies=[Depends(require_key)])
def cleanup_preview(ttl_seconds: int = None):
    """
    Preview conversations that WOULD be deleted by cleanup (no deletion).
    """
    ttl = ttl_seconds if ttl_seconds is not None else db_service.SESSION_TTL_SECONDS
    cutoff = datetime.utcnow() - timedelta(seconds=ttl)
    cutoff_iso = cutoff.isoformat() + "Z"
    conn = db_service._get_conn()
    cur = conn.cursor()
    # ensure column exists
    cur.execute("PRAGMA table_info(conversations);")
    cols = [r["name"] for r in cur.fetchall()]
    if "last_activity" not in cols:
        conn.close()
        return {"preview": [], "note": "no last_activity column present"}
    cur.execute("SELECT id, session_id, last_activity FROM conversations WHERE last_activity < ? ORDER BY last_activity ASC", (cutoff_iso,))
    rows = cur.fetchall()
    conn.close()
    return {"preview": [dict(r) for r in rows], "cutoff_iso": cutoff_iso, "ttl_seconds": ttl}

@router.post("/cleanup/run", dependencies=[Depends(require_key)])
def cleanup_run():
    """
    Run actual cleanup (deletes old conversations). Returns deleted ids.
    """
    deleted = db_service.cleanup_old_sessions()
    return {"deleted": deleted}

@router.delete("/conversation/{conversation_id}", dependencies=[Depends(require_key)])
def delete_conversation(conversation_id: str):
    ok = db_service.delete_conversation(conversation_id)
    return {"deleted": ok, "conversation_id": conversation_id}
