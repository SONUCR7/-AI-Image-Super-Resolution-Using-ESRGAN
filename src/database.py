"""
Database Layer for SR Vision Lab
Provides SQLite persistence for user authentication, account management,
and user-isolated image enhancement history.
Configured with WAL mode, parameterized queries, and foreign key enforcement.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict, Any, Optional
import bcrypt

import config

DB_PATH: Path = getattr(config, "DATA_DIR", Path("data")) / "sr_vision_lab.db"


@contextmanager
def get_db_connection():
    """Context manager yielding a SQLite connection with row_factory set to sqlite3.Row."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=20.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Initializes the database schema and creates tables, indexes, and initial seeds."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                reset_token TEXT,
                reset_token_expiry TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            );
        """)

        # User-isolated enhancement history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS enhancement_history (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                orig_width INTEGER NOT NULL,
                orig_height INTEGER NOT NULL,
                enhanced_width INTEGER NOT NULL,
                enhanced_height INTEGER NOT NULL,
                scale_factor TEXT NOT NULL,
                model_name TEXT NOT NULL,
                elapsed_sec REAL NOT NULL,
                device TEXT NOT NULL,
                orig_thumb_path TEXT NOT NULL,
                enh_thumb_path TEXT NOT NULL,
                enhanced_file_path TEXT NOT NULL,
                file_size_kb REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
        """)

        # Performance indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_user_id ON enhancement_history(user_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_created_at ON enhancement_history(created_at);")

        # Seed default demo account if no users exist
        cursor.execute("SELECT COUNT(*) AS cnt FROM users;")
        row = cursor.fetchone()
        if row and row["cnt"] == 0:
            demo_email = "demo@srvisionlab.ai"
            demo_pw = "Password@123"
            pw_hash = bcrypt.hashpw(demo_pw.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
            cursor.execute("""
                INSERT INTO users (full_name, email, password_hash, role)
                VALUES (?, ?, ?, 'researcher');
            """, ("Dr. Jessica Vance", demo_email, pw_hash))


def execute_query(sql: str, params: tuple = ()) -> int:
    """Executes an INSERT, UPDATE, or DELETE query and returns the last row id or affected row count."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        return cursor.lastrowid or cursor.rowcount


def fetch_one(sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
    """Fetches a single row as a dictionary."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        row = cursor.fetchone()
        return dict(row) if row else None


def fetch_all(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    """Fetches multiple rows as a list of dictionaries."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
