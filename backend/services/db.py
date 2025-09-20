# backend/services/db.py
import os
import sqlite3
import uuid
import json
from datetime import datetime
from typing import Optional, List, Dict

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.environ.get("DB_PATH", os.path.join(BASE_DIR, "..", "data.db"))
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/tmp/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize DB and tables
def _get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def _init():
    conn = _get_conn()
    cur = conn.cursor()
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
    conn.close()

_init()

# Helpers
def _now():
    return datetime.utcnow().isoformat() + "Z"

def upsert_conversation(session_id: Optional[str]):
    conn = _get_conn()
    cur = conn.cursor()
    if session_id:
        cur.execute("SELECT id, session_id FROM conversations WHERE session_id = ?", (session_id,))
        row = cur.fetchone()
        if row:
            conn.close()
            return {"id": row["id"], "session_id": row["session_id"]}
    # create new conversation
    new_id = str(uuid.uuid4())
    new_session = session_id or f"sess_{uuid.uuid4().hex[:12]}"
    cur.execute(
        "INSERT INTO conversations (id, session_id, title, created_at) VALUES (?, ?, ?, ?)",
        (new_id, new_session, "Chat", _now())
    )
    conn.commit()
    conn.close()
    return {"id": new_id, "session_id": new_session}

def save_message(conversation_id: str, role: str, text: str, file_url: Optional[str] = None):
    conn = _get_conn()
    cur = conn.cursor()
    msg_id = str(uuid.uuid4())
    cur.execute(
        "INSERT INTO messages (id, conversation_id, role, text, file_url, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (msg_id, conversation_id, role, text, file_url, _now())
    )
    conn.commit()
    conn.close()
    return msg_id

def get_last_messages(conversation_id: str, limit: int = 6) -> List[Dict]:
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
    # Move file into UPLOAD_DIR with a unique name, return its local path (or URL if you add S3)
    if not os.path.exists(file_path):
        raise FileNotFoundError("file not found: " + file_path)
    ext = os.path.splitext(file_path)[1]
    dest_name = f"{uuid.uuid4().hex}{ext}"
    dest_path = os.path.join(UPLOAD_DIR, dest_name)
    os.replace(file_path, dest_path)
    # Optionally return a public URL — for now return local path
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
    lead = dict(row)
    return lead

def notify_sales(lead: dict):
    # Minimal notifier: print to logs. Replace with Slack/webhook/email as needed.
    print("=== NEW LEAD ===")
    print(json.dumps(lead, indent=2, ensure_ascii=False))
    print("Notify your sales team (hook this function to Slack/email).")
    return True
