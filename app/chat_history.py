
import sqlite3
from pathlib import Path
from datetime import datetime, timezone


# --------------------------------
# DATABASE CONFIGURATION
# --------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DB_PATH = PROJECT_DIR / "chat_history.db"

DB_TIMEOUT = 10


# --------------------------------
# DATABASE CONNECTION
# --------------------------------

def get_connection():

    connection = sqlite3.connect(
        DB_PATH,
        timeout=DB_TIMEOUT
    )

    connection.row_factory = sqlite3.Row

    return connection


# --------------------------------
# INITIALIZE DATABASE
# --------------------------------

def init_db():

    try:

        with get_connection() as connection:

            connection.execute("""
                               CREATE TABLE IF NOT EXISTS chat_messages (

                                                                            id INTEGER PRIMARY KEY AUTOINCREMENT,

                                                                            session_id TEXT NOT NULL,

                                                                            role TEXT NOT NULL
                                                                            CHECK(role IN ('user', 'assistant')),

                                   content TEXT NOT NULL,

                                   created_at TEXT NOT NULL
                                   )
                               """)

            # Index for faster session-based retrieval
            connection.execute("""
                               CREATE INDEX IF NOT EXISTS
                                   idx_session_id_id
                                   ON chat_messages(session_id, id)
                               """)

        return True

    except sqlite3.Error as e:

        print(f"Database initialization error: {e}")

        raise


# --------------------------------
# SAVE MESSAGE
# --------------------------------

def save_message(session_id, role, content):

    if not session_id or role not in ("user", "assistant"):
        raise ValueError("Invalid session ID or message role.")

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Message content cannot be empty.")

    created_at = datetime.now(timezone.utc).isoformat()

    try:

        with get_connection() as connection:

            connection.execute("""
                               INSERT INTO chat_messages
                                   (session_id, role, content, created_at)

                               VALUES (?, ?, ?, ?)
                               """, (
                                   session_id,
                                   role,
                                   content,
                                   created_at
                               ))

    except sqlite3.Error as e:

        print(f"Message saving error: {e}")

        raise


# --------------------------------
# GET CONVERSATION HISTORY
# --------------------------------

def get_messages(session_id):

    if not session_id:
        raise ValueError("Session ID is required.")

    try:

        with get_connection() as connection:

            rows = connection.execute("""
                                      SELECT role, content, created_at

                                      FROM chat_messages

                                      WHERE session_id = ?

                                      ORDER BY id ASC
                                      """, (session_id,)).fetchall()

            return [dict(row) for row in rows]

    except sqlite3.Error as e:

        print(f"Message retrieval error: {e}")

        raise


# --------------------------------
# DELETE CONVERSATION HISTORY
# --------------------------------

def delete_history(session_id):

    if not session_id:
        raise ValueError("Session ID is required.")

    try:

        with get_connection() as connection:

            cursor = connection.execute("""
                                        DELETE FROM chat_messages

                                        WHERE session_id = ?
                                        """, (session_id,))

            return cursor.rowcount

    except sqlite3.Error as e:

        print(f"Message deletion error: {e}")

        raise
