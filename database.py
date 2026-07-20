import sqlite3, os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "distill.db")

def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def init_db():
    conn = _conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            balance INTEGER NOT NULL DEFAULT 30,
            feedback_given INTEGER NOT NULL DEFAULT 0,
            feedback_date TEXT,
            feedback_text TEXT,
            referral_code TEXT NOT NULL,
            referral_count INTEGER NOT NULL DEFAULT 0,
            last_daily_refill TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            love TEXT DEFAULT '',
            improve TEXT DEFAULT '',
            rating INTEGER DEFAULT 0,
            text TEXT DEFAULT '',
            name TEXT DEFAULT '',
            email TEXT DEFAULT '',
            timestamp TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()

def generate_code():
    import random, string
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

def create_session():
    conn = _conn()
    code = generate_code()
    sid = os.urandom(16).hex()
    conn.execute("INSERT INTO sessions (session_id, referral_code) VALUES (?, ?)", (sid, code))
    conn.commit()
    row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (sid,)).fetchone()
    conn.close()
    return _row_to_dict(row)

def _row_to_dict(row):
    d = dict(row)
    d["balance"] = d["balance"]
    d["feedback_given"] = bool(d["feedback_given"])
    d["referral_count"] = d["referral_count"]
    return d

def _apply_daily_refill(conn, sid):
    row = conn.execute("SELECT balance, last_daily_refill FROM sessions WHERE session_id = ?", (sid,)).fetchone()
    if not row: return
    today = datetime.now().date().isoformat()
    if row["last_daily_refill"] == today: return
    conn.execute("UPDATE sessions SET balance = MAX(balance, 30), last_daily_refill = ? WHERE session_id = ?", (today, sid))
    conn.commit()

def get_session(sid: str):
    conn = _conn()
    _apply_daily_refill(conn, sid)
    row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (sid,)).fetchone()
    conn.close()
    if not row: return None
    return _row_to_dict(row)

def deduct_credits(sid: str, cost: int):
    conn = _conn()
    _apply_daily_refill(conn, sid)
    cur = conn.execute("UPDATE sessions SET balance = balance - ? WHERE session_id = ? AND balance >= ?", (cost, sid, cost))
    if cur.rowcount == 0:
        conn.close()
        return None
    conn.commit()
    row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (sid,)).fetchone()
    conn.close()
    return _row_to_dict(row)

def add_credits(sid: str, amount: int):
    conn = _conn()
    conn.execute("UPDATE sessions SET balance = balance + ? WHERE session_id = ?", (amount, sid))
    conn.commit()
    row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (sid,)).fetchone()
    conn.close()
    return _row_to_dict(row)

def record_feedback(sid: str, love: str, improve: str, rating: int, text: str, name: str = "", email: str = ""):
    conn = _conn()
    conn.execute("INSERT INTO feedback (session_id, love, improve, rating, text, name, email) VALUES (?, ?, ?, ?, ?, ?, ?)", (sid, love, improve, rating, text, name, email))
    conn.execute("UPDATE sessions SET feedback_given = 1, feedback_date = datetime('now'), feedback_text = ? WHERE session_id = ?", (text, sid))
    conn.commit()
    conn.close()
