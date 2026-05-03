from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite


LOGGER = logging.getLogger("ai_firewall_proxy.log_store")


@dataclass(slots=True)
class FirewallLogRecord:
    created_at: str
    agent_id: str
    user_input: str
    decision: str
    trigger_layer: str
    scope_score: float
    prompt_injection_score: float
    raw_scores: dict[str, Any]
    request_payload: dict[str, Any]
    response_payload: dict[str, Any]
    pg2_main: float = 0.0
    scope_main: float = 0.0
    pii_main: float = 0.0
    doc_pg2_max: float = 0.0
    doc_scope_min: float = 1.0
    doc_flagged_ratio: float = 0.0
    lg4_unsafe: int = 0
    lg4_code_abuse: int = 0
    final_risk: float = 0.0
    attachment_summary: dict[str, Any] | None = None
    decision_reasons: list[str] | None = None
    chunk_summaries: list[dict[str, Any]] | None = None
    model_versions: dict[str, Any] | None = None


class AsyncSQLiteLogger:
    def __init__(self, db_path: Path, max_rows: int) -> None:
        self._db_path = db_path
        self._max_rows = max_rows
        self._queue: asyncio.Queue[FirewallLogRecord | None] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._worker_task = asyncio.create_task(self._worker(), name="firewall-log-worker")

    async def stop(self) -> None:
        if self._worker_task is None:
            return
        await self._queue.put(None)
        await self._worker_task
        self._worker_task = None

    def enqueue(self, record: FirewallLogRecord) -> None:
        self._queue.put_nowait(record)

    async def _worker(self) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS firewall_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    user_input TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    trigger_layer TEXT NOT NULL,
                    scope_score REAL NOT NULL,
                    prompt_injection_score REAL NOT NULL,
                    raw_scores TEXT NOT NULL,
                    request_payload TEXT NOT NULL,
                    response_payload TEXT NOT NULL,
                    pg2_main REAL NOT NULL DEFAULT 0.0,
                    scope_main REAL NOT NULL DEFAULT 0.0,
                    pii_main REAL NOT NULL DEFAULT 0.0,
                    doc_pg2_max REAL NOT NULL DEFAULT 0.0,
                    doc_scope_min REAL NOT NULL DEFAULT 1.0,
                    doc_flagged_ratio REAL NOT NULL DEFAULT 0.0,
                    lg4_unsafe INTEGER NOT NULL DEFAULT 0,
                    lg4_code_abuse INTEGER NOT NULL DEFAULT 0,
                    final_risk REAL NOT NULL DEFAULT 0.0,
                    attachment_summary TEXT NOT NULL DEFAULT '{}',
                    decision_reasons TEXT NOT NULL DEFAULT '[]',
                    chunk_summaries TEXT NOT NULL DEFAULT '[]',
                    model_versions TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            await self._ensure_columns(db)
            await db.commit()

            while True:
                record = await self._queue.get()
                if record is None:
                    break

                try:
                    await db.execute(
                        """
                        INSERT INTO firewall_logs (
                            created_at,
                            agent_id,
                            user_input,
                            decision,
                            trigger_layer,
                            scope_score,
                            prompt_injection_score,
                            raw_scores,
                            request_payload,
                            response_payload,
                            pg2_main,
                            scope_main,
                            pii_main,
                            doc_pg2_max,
                            doc_scope_min,
                            doc_flagged_ratio,
                            lg4_unsafe,
                            lg4_code_abuse,
                            final_risk,
                            attachment_summary,
                            decision_reasons,
                            chunk_summaries,
                            model_versions
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record.created_at,
                            record.agent_id,
                            record.user_input,
                            record.decision,
                            record.trigger_layer,
                            record.scope_score,
                            record.prompt_injection_score,
                            json.dumps(record.raw_scores, ensure_ascii=False),
                            json.dumps(record.request_payload, ensure_ascii=False),
                            json.dumps(record.response_payload, ensure_ascii=False),
                            record.pg2_main,
                            record.scope_main,
                            record.pii_main,
                            record.doc_pg2_max,
                            record.doc_scope_min,
                            record.doc_flagged_ratio,
                            record.lg4_unsafe,
                            record.lg4_code_abuse,
                            record.final_risk,
                            json.dumps(record.attachment_summary or {}, ensure_ascii=False),
                            json.dumps(record.decision_reasons or [], ensure_ascii=False),
                            json.dumps(record.chunk_summaries or [], ensure_ascii=False),
                            json.dumps(record.model_versions or {}, ensure_ascii=False),
                        ),
                    )
                    await db.execute(
                        """
                        DELETE FROM firewall_logs
                        WHERE id NOT IN (
                            SELECT id
                            FROM firewall_logs
                            ORDER BY id DESC
                            LIMIT ?
                        )
                        """,
                        (self._max_rows,),
                    )
                    await db.commit()
                except Exception:
                    LOGGER.exception("Failed to persist firewall log record.")

    @staticmethod
    async def _ensure_columns(db: aiosqlite.Connection) -> None:
        cursor = await db.execute("PRAGMA table_info(firewall_logs)")
        rows = await cursor.fetchall()
        existing = {row[1] for row in rows}
        columns = {
            "pg2_main": "REAL NOT NULL DEFAULT 0.0",
            "scope_main": "REAL NOT NULL DEFAULT 0.0",
            "pii_main": "REAL NOT NULL DEFAULT 0.0",
            "doc_pg2_max": "REAL NOT NULL DEFAULT 0.0",
            "doc_scope_min": "REAL NOT NULL DEFAULT 1.0",
            "doc_flagged_ratio": "REAL NOT NULL DEFAULT 0.0",
            "lg4_unsafe": "INTEGER NOT NULL DEFAULT 0",
            "lg4_code_abuse": "INTEGER NOT NULL DEFAULT 0",
            "final_risk": "REAL NOT NULL DEFAULT 0.0",
            "attachment_summary": "TEXT NOT NULL DEFAULT '{}'",
            "decision_reasons": "TEXT NOT NULL DEFAULT '[]'",
            "chunk_summaries": "TEXT NOT NULL DEFAULT '[]'",
            "model_versions": "TEXT NOT NULL DEFAULT '{}'",
        }
        for column_name, column_type in columns.items():
            if column_name in existing:
                continue
            await db.execute(f"ALTER TABLE firewall_logs ADD COLUMN {column_name} {column_type}")
