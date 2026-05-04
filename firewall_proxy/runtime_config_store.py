from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from firewall_proxy.config import AgentConfig, UpstreamConfig


@dataclass(slots=True)
class RuntimeAgentSetup:
    agent_id: str
    description: str
    allowed_examples: list[str]
    denied_examples: list[str]
    use_local_mock: bool
    base_url: str | None
    timeout_seconds: float
    default_model: str
    updated_at: str

    def to_agent_config(self) -> AgentConfig:
        return AgentConfig(
            agent_id=self.agent_id,
            description=self.description,
            allowed_examples=self.allowed_examples,
            denied_examples=self.denied_examples,
        )

    def to_upstream_config(self) -> UpstreamConfig:
        return UpstreamConfig(
            use_local_mock=self.use_local_mock,
            base_url=self.base_url,
            timeout_seconds=self.timeout_seconds,
            default_model=self.default_model,
        )


def ensure_runtime_agent_setup_table(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS runtime_agent_setup (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                agent_id TEXT NOT NULL,
                description TEXT NOT NULL,
                allowed_examples TEXT NOT NULL,
                denied_examples TEXT NOT NULL,
                use_local_mock INTEGER NOT NULL,
                base_url TEXT,
                timeout_seconds REAL NOT NULL,
                default_model TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.commit()


def load_runtime_agent_setup(db_path: Path) -> RuntimeAgentSetup | None:
    ensure_runtime_agent_setup_table(db_path)
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT
                agent_id,
                description,
                allowed_examples,
                denied_examples,
                use_local_mock,
                base_url,
                timeout_seconds,
                default_model,
                updated_at
            FROM runtime_agent_setup
            WHERE id = 1
            """
        ).fetchone()

    if row is None:
        return None

    return RuntimeAgentSetup(
        agent_id=str(row[0]),
        description=str(row[1]),
        allowed_examples=_json_list(row[2]),
        denied_examples=_json_list(row[3]),
        use_local_mock=bool(row[4]),
        base_url=str(row[5]).strip() if row[5] else None,
        timeout_seconds=float(row[6]),
        default_model=str(row[7]),
        updated_at=str(row[8]),
    )


def get_runtime_agent_setup_versions(db_path: Path) -> dict[str, str]:
    setup = load_runtime_agent_setup(db_path)
    if setup is None:
        return {}
    return {setup.agent_id: setup.updated_at}


def upsert_runtime_agent_setup(
    db_path: Path,
    *,
    agent_id: str,
    description: str,
    allowed_examples: list[str],
    denied_examples: list[str],
    use_local_mock: bool,
    base_url: str | None,
    timeout_seconds: float,
    default_model: str,
) -> RuntimeAgentSetup:
    setup = RuntimeAgentSetup(
        agent_id=agent_id.strip(),
        description=description.strip(),
        allowed_examples=[example.strip() for example in allowed_examples if example.strip()],
        denied_examples=[example.strip() for example in denied_examples if example.strip()],
        use_local_mock=use_local_mock,
        base_url=(base_url or "").strip() or None,
        timeout_seconds=float(timeout_seconds),
        default_model=default_model.strip(),
        updated_at=_utc_now(),
    )
    _validate_runtime_agent_setup(setup)
    ensure_runtime_agent_setup_table(db_path)

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO runtime_agent_setup (
                id,
                agent_id,
                description,
                allowed_examples,
                denied_examples,
                use_local_mock,
                base_url,
                timeout_seconds,
                default_model,
                updated_at
            ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                agent_id = excluded.agent_id,
                description = excluded.description,
                allowed_examples = excluded.allowed_examples,
                denied_examples = excluded.denied_examples,
                use_local_mock = excluded.use_local_mock,
                base_url = excluded.base_url,
                timeout_seconds = excluded.timeout_seconds,
                default_model = excluded.default_model,
                updated_at = excluded.updated_at
            """,
            (
                setup.agent_id,
                setup.description,
                json.dumps(setup.allowed_examples, ensure_ascii=False),
                json.dumps(setup.denied_examples, ensure_ascii=False),
                1 if setup.use_local_mock else 0,
                setup.base_url,
                setup.timeout_seconds,
                setup.default_model,
                setup.updated_at,
            ),
        )
        # Drop stale split-table settings from the earlier dashboard design.
        _delete_from_table_if_exists(connection, "runtime_upstream_settings")
        _delete_from_table_if_exists(connection, "agent_profiles")
        connection.commit()

    return setup


def clear_runtime_agent_setup(db_path: Path) -> None:
    ensure_runtime_agent_setup_table(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute("DELETE FROM runtime_agent_setup WHERE id = 1")
        _delete_from_table_if_exists(connection, "runtime_upstream_settings")
        _delete_from_table_if_exists(connection, "agent_profiles")
        connection.commit()


def split_examples(raw_value: str) -> list[str]:
    return [line.strip(" -\t") for line in raw_value.splitlines() if line.strip(" -\t")]


def _validate_runtime_agent_setup(setup: RuntimeAgentSetup) -> None:
    if not setup.agent_id:
        raise ValueError("Agent ID is required.")
    if not setup.description:
        raise ValueError("Description is required.")
    if not setup.allowed_examples:
        raise ValueError("At least one allowed example is required.")
    if not setup.denied_examples:
        raise ValueError("At least one denied example is required.")
    if not setup.default_model:
        raise ValueError("Default model is required.")
    if setup.timeout_seconds <= 0:
        raise ValueError("Timeout must be greater than zero.")
    if not setup.use_local_mock and not setup.base_url:
        raise ValueError("Base URL is required when local mock mode is off.")


def _delete_from_table_if_exists(connection: sqlite3.Connection, table_name: str) -> None:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    if exists:
        connection.execute(f"DELETE FROM {table_name}")


def _json_list(raw_value: Any) -> list[str]:
    if isinstance(raw_value, list):
        return [str(value) for value in raw_value if str(value).strip()]
    try:
        loaded = json.loads(raw_value or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(loaded, list):
        return []
    return [str(value) for value in loaded if str(value).strip()]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
