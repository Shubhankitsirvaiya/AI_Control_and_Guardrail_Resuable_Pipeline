import sqlite3
from pathlib import Path
from typing import Any
import os
import json

from dotenv import load_dotenv

from .models import DocumentRecord, ValidationResult


class AuditStore:
    def __init__(self, db_path: str | None = None):
        load_dotenv()
        db_path = db_path or os.getenv("GATEWAY_DB_PATH", "data/gateway.db")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    request_id TEXT,
                    user_role TEXT,
                    action TEXT NOT NULL,
                    decision TEXT,
                    risk_level TEXT,
                    detail TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    request_id TEXT NOT NULL,
                    rating TEXT NOT NULL,
                    comment TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prompt_audits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    request_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    system_prompt TEXT NOT NULL,
                    user_prompt TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    owner_id TEXT NOT NULL DEFAULT 'system',
                    validation_json TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(documents)").fetchall()
            }
            if "owner_id" not in columns:
                conn.execute("ALTER TABLE documents ADD COLUMN owner_id TEXT NOT NULL DEFAULT 'system'")
            conn.execute(
                "UPDATE documents SET owner_id = 'analyst' WHERE source = 'uploaded' AND owner_id = 'system'"
            )

    def log(
        self,
        action: str,
        request_id: str | None = None,
        user_role: str | None = None,
        decision: str | None = None,
        risk_level: str | None = None,
        detail: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_events (request_id, user_role, action, decision, risk_level, detail)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (request_id, user_role, action, decision, risk_level, detail),
            )

    def add_feedback(self, request_id: str, rating: str, comment: str | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO feedback (request_id, rating, comment) VALUES (?, ?, ?)",
                (request_id, rating, comment),
            )
            conn.execute(
                """
                INSERT INTO audit_events (request_id, action, decision, detail)
                VALUES (?, 'FEEDBACK_CAPTURED', ?, ?)
                """,
                (request_id, rating, comment),
            )

    def log_prompt(self, request_id: str, model: str, system_prompt: str, user_prompt: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO prompt_audits (request_id, model, system_prompt, user_prompt)
                VALUES (?, ?, ?, ?)
                """,
                (request_id, model, system_prompt, user_prompt),
            )
            conn.execute(
                """
                INSERT INTO audit_events (request_id, action, decision, detail)
                VALUES (?, 'PROMPT_AUDITED', ?, ?)
                """,
                (request_id, model, "Hidden prompt saved to prompt audit table."),
            )

    def save_document(self, document: DocumentRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents (id, name, content, source, owner_id, validation_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    document.id,
                    document.name,
                    document.content,
                    document.source,
                    document.owner_id,
                    document.validation.model_dump_json(),
                ),
            )

    def list_documents(self) -> list[DocumentRecord]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
            documents: list[DocumentRecord] = []
            for row in rows:
                validation_data = json.loads(row["validation_json"])
                documents.append(
                    DocumentRecord(
                        id=row["id"],
                        name=row["name"],
                        content=row["content"],
                        source=row["source"],
                        owner_id=row["owner_id"],
                        validation=ValidationResult.model_validate(validation_data),
                    )
                )
            return documents

    def list_audit_events(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM audit_events ORDER BY id DESC LIMIT 200").fetchall()
            return [dict(row) for row in rows]

    def list_prompt_audits(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM prompt_audits ORDER BY id DESC LIMIT 100").fetchall()
            return [dict(row) for row in rows]

    def list_feedback(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM feedback ORDER BY id DESC LIMIT 200").fetchall()
            return [dict(row) for row in rows]

