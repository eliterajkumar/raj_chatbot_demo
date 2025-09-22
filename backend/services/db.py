# backend/services/db.py
import os
import sqlite3
import uuid
import json
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.environ.get("DB_PATH", os.path.join(BASE_DIR, "..", "data.db"))
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/tmp/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Session TTL (seconds) default: 10 minutes
SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", 600))

# ---------------------------
# DB init + migrations
# ---------------------------
def _get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def _ensure_migrations(conn: sqlite3.Connection):
    """
    Ensure DB schema includes expected columns. Add missing columns if needed.
    This performs lightweight, idempotent migrations.
    """
    cur = conn.cursor()
    # check conversations columns
    cur.execute("PRAGMA table_info(conversations);")
    cols = [r["name"] for r in cur.fetchall()]
    if "last_activity" not in cols:
        cur.execute("ALTER TABLE conversations ADD COLUMN last_activity TEXT;")
        cur.execute("UPDATE conversations SET last_activity = created_at WHERE last_activity IS NULL;")
        conn.commit()

def _init():
    conn = _get_conn()
    cur = conn.cursor()
    # create tables if not exists (safe to run repeatedly)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
      id TEXT PRIMARY KEY,
      session_id TEXT UNIQUE,
      title TEXT,
      created_at TEXT
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
      id TEXT PRIMARY KEY,
      conversation_id TEXT,
      role TEXT,
      text TEXT,
      file_url TEXT,
      created_at TEXT
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS leads (
      id TEXT PRIMARY KEY,
      conversation_id TEXT,
      name TEXT,
      email TEXT,
      phone TEXT,
      interest TEXT,
      score REAL,
      metadata TEXT,
      created_at TEXT
    );
    """)
    conn.commit()

    # Run lightweight migrations (idempotent)
    _ensure_migrations(conn)

    # Ensure conversations table has last_activity column (for new installs, add column now)
    cur.execute("PRAGMA table_info(conversations);")
    cols = [r["name"] for r in cur.fetchall()]
    if "last_activity" not in cols:
        cur.execute("ALTER TABLE conversations ADD COLUMN last_activity TEXT;")
        cur.execute("UPDATE conversations SET last_activity = created_at WHERE last_activity IS NULL;")
        conn.commit()

    conn.close()

_init()

# ---------------------------
# Helpers
# ---------------------------
def _now():
    return datetime.utcnow().isoformat() + "Z"

def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", ""))
    except Exception:
        return None

# ---------------------------
# Core DB API (used by routers)
# ---------------------------
def upsert_conversation(session_id: Optional[str]):
    """
    Get or create a conversation. Always updates last_activity to now.
    Returns: {"id": <id>, "session_id": <session_id>}
    """
    conn = _get_conn()
    cur = conn.cursor()
    if session_id:
        cur.execute("SELECT id, session_id FROM conversations WHERE session_id = ?", (session_id,))
        row = cur.fetchone()
        if row:
            # update last_activity
            cur.execute("UPDATE conversations SET last_activity = ? WHERE id = ?", (_now(), row["id"]))
            conn.commit()
            conn.close()
            return {"id": row["id"], "session_id": row["session_id"]}

    # create new conversation
    new_id = str(uuid.uuid4())
    new_session = session_id or f"sess_{uuid.uuid4().hex[:12]}"
    cur.execute(
        "INSERT INTO conversations (id, session_id, title, created_at, last_activity) VALUES (?, ?, ?, ?, ?)",
        (new_id, new_session, "Chat", _now(), _now())
    )
    conn.commit()
    conn.close()
    return {"id": new_id, "session_id": new_session}

def save_message(conversation_id: str, role: str, text: str, file_url: Optional[str] = None):
    """
    Save a message and bump conversation last_activity.
    """
    conn = _get_conn()
    cur = conn.cursor()
    msg_id = str(uuid.uuid4())
    cur.execute(
        "INSERT INTO messages (id, conversation_id, role, text, file_url, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (msg_id, conversation_id, role, text, file_url, _now())
    )
    # update conversation last_activity
    cur.execute("UPDATE conversations SET last_activity = ? WHERE id = ?", (_now(), conversation_id))
    conn.commit()
    conn.close()
    return msg_id

def get_last_messages(conversation_id: str, limit: int = 6) -> List[Dict]:
    """
    Return last `limit` messages ordered oldest->newest.
    """
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT role, text, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at DESC LIMIT ?",
        (conversation_id, limit)
    )
    rows = cur.fetchall()
    conn.close()
    # reverse to return oldest->newest
    rows = list(reversed(rows))
    return [{"role": r["role"], "text": r["text"], "created_at": r["created_at"]} for r in rows]

def save_file_to_storage(file_path: str, conversation_id: Optional[str] = None) -> str:
    """
    Move file into UPLOAD_DIR with a unique name, return its local path.
    (Site version: file uploads are disabled; kept for demo compatibility.)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError("file not found: " + file_path)
    ext = os.path.splitext(file_path)[1]
    dest_name = f"{uuid.uuid4().hex}{ext}"
    dest_path = os.path.join(UPLOAD_DIR, dest_name)
    os.replace(file_path, dest_path)
    return dest_path

def create_lead(conversation_id: str, snippet: str, score: float = 0.5, metadata: Optional[dict] = None):
    conn = _get_conn()
    cur = conn.cursor()
    lead_id = str(uuid.uuid4())
    meta_json = json.dumps(metadata or {})
    cur.execute(
        "INSERT INTO leads (id, conversation_id, interest, score, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (lead_id, conversation_id, snippet[:120], score, meta_json, _now())
    )
    conn.commit()
    # fetch inserted
    cur.execute("SELECT * FROM leads WHERE id = ?", (lead_id,))
    row = cur.fetchone()
    conn.close()
    lead = dict(row) if row else {}
    return lead

def notify_sales(lead: dict):
    # Minimal notifier: print to logs. Replace with Slack/webhook/email as needed.
    print("=== NEW LEAD ===")
    print(json.dumps(lead, indent=2, ensure_ascii=False))
    print("Notify your sales team (hook this function to Slack/email).")
    return True

def delete_conversation(conversation_id: str):
    """
    Delete a conversation and its messages and leads.
    """
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    cur.execute("DELETE FROM leads WHERE conversation_id = ?", (conversation_id,))
    cur.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()
    conn.close()
    return True

# ---------------------------
# Cleanup / maintenance
# ---------------------------
def cleanup_old_sessions(ttl_seconds: Optional[int] = None):
    """
    Delete conversations whose last_activity is older than TTL.
    Returns list of deleted conversation ids.
    """
    ttl = ttl_seconds if ttl_seconds is not None else SESSION_TTL_SECONDS
    cutoff = datetime.utcnow() - timedelta(seconds=ttl)
    cutoff_iso = cutoff.isoformat() + "Z"

    conn = _get_conn()
    cur = conn.cursor()
    # Guard: ensure last_activity column exists
    cur.execute("PRAGMA table_info(conversations);")
    cols = [r["name"] for r in cur.fetchall()]
    if "last_activity" not in cols:
        conn.close()
        return []  # nothing to cleanup if column missing

    # find conversations to delete
    cur.execute("SELECT id FROM conversations WHERE last_activity < ?", (cutoff_iso,))
    rows = cur.fetchall()
    ids = [r["id"] for r in rows]

    for cid in ids:
        cur.execute("DELETE FROM messages WHERE conversation_id = ?", (cid,))
        cur.execute("DELETE FROM leads WHERE conversation_id = ?", (cid,))
        cur.execute("DELETE FROM conversations WHERE id = ?", (cid,))

    conn.commit()
    conn.close()
    return ids
