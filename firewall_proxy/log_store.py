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
                    response_payload TEXT NOT NULL
                )
                """
            )
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
                            response_payload
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
