import json
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "state.db"


def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init():
    with conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS sessions(
            id TEXT PRIMARY KEY, created REAL, status TEXT DEFAULT 'draft');
        CREATE TABLE IF NOT EXISTS transcripts(
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, filename TEXT, text TEXT, created REAL);
        CREATE TABLE IF NOT EXISTS extractions(
            session_id TEXT PRIMARY KEY, intake_json TEXT, research_json TEXT, plan_json TEXT);
        CREATE TABLE IF NOT EXISTS sections(
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, key TEXT, title TEXT, role TEXT,
            position INTEGER, version INTEGER, text TEXT, created REAL);
        CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, created REAL);
        """)


def ensure_session(sid: str):
    with conn() as c:
        c.execute("INSERT OR IGNORE INTO sessions(id, created) VALUES(?,?)", (sid, time.time()))


def add_transcript(sid, filename, text):
    with conn() as c:
        c.execute("INSERT INTO transcripts(session_id, filename, text, created) VALUES(?,?,?,?)",
                  (sid, filename, text, time.time()))


def all_transcript_text(sid) -> str:
    with conn() as c:
        rows = c.execute("SELECT filename, text FROM transcripts WHERE session_id=? ORDER BY id", (sid,)).fetchall()
    return "\n\n".join(f"### FILE: {r['filename']}\n{r['text']}" for r in rows)


def has_transcript(sid) -> bool:
    with conn() as c:
        return c.execute("SELECT 1 FROM transcripts WHERE session_id=?", (sid,)).fetchone() is not None


def save_extraction(sid, intake: dict, research: dict, plan: dict):
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO extractions VALUES(?,?,?,?)",
                  (sid, json.dumps(intake), json.dumps(research, default=str), json.dumps(plan)))


def get_extraction(sid):
    with conn() as c:
        r = c.execute("SELECT * FROM extractions WHERE session_id=?", (sid,)).fetchone()
    if not r:
        return None
    return {"intake": json.loads(r["intake_json"]), "research": json.loads(r["research_json"]),
            "plan": json.loads(r["plan_json"])}


def save_section(sid, key, title, role, position, text):
    """Insert a new version; older versions are kept as history."""
    with conn() as c:
        v = c.execute("SELECT COALESCE(MAX(version),0) FROM sections WHERE session_id=? AND key=?",
                      (sid, key)).fetchone()[0] + 1
        c.execute("INSERT INTO sections(session_id,key,title,role,position,version,text,created) "
                  "VALUES(?,?,?,?,?,?,?,?)", (sid, key, title, role, position, v, text, time.time()))
    return v


def current_sections(sid) -> list[dict]:
    with conn() as c:
        rows = c.execute("""SELECT s.* FROM sections s
            JOIN (SELECT key, MAX(version) mv FROM sections WHERE session_id=? GROUP BY key) m
            ON s.key=m.key AND s.version=m.mv WHERE s.session_id=? ORDER BY s.position""",
                         (sid, sid)).fetchall()
    return [dict(r) for r in rows]


def log_message(sid, role, content):
    with conn() as c:
        c.execute("INSERT INTO messages(session_id, role, content, created) VALUES(?,?,?,?)",
                  (sid, role, content, time.time()))