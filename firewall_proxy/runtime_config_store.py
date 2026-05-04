from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from firewall_proxy.config import AgentConfig, UpstreamConfig


@dataclass(slots=True)
class ToolRegistryEntry:
    name: str
    category: str
    purpose: str
    risk: str
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "purpose": self.purpose,
            "risk": self.risk,
            "enabled": self.enabled,
        }


@dataclass(slots=True)
class RuntimeAgentSetup:
    agent_id: str
    description: str
    allowed_examples: list[str]
    denied_examples: list[str]
    tool_registry: list[ToolRegistryEntry]
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
                tool_registry TEXT NOT NULL DEFAULT '[]',
                use_local_mock INTEGER NOT NULL,
                base_url TEXT,
                timeout_seconds REAL NOT NULL,
                default_model TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        _ensure_columns(connection)
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
                tool_registry,
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
        tool_registry=_tool_registry_from_json(row[4]),
        use_local_mock=bool(row[5]),
        base_url=str(row[6]).strip() if row[6] else None,
        timeout_seconds=float(row[7]),
        default_model=str(row[8]),
        updated_at=str(row[9]),
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
    tool_registry: list[ToolRegistryEntry | dict[str, Any]] | None = None,
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
        tool_registry=_normalize_tool_registry(tool_registry or []),
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
                tool_registry,
                use_local_mock,
                base_url,
                timeout_seconds,
                default_model,
                updated_at
            ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                agent_id = excluded.agent_id,
                description = excluded.description,
                allowed_examples = excluded.allowed_examples,
                denied_examples = excluded.denied_examples,
                tool_registry = excluded.tool_registry,
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
                json.dumps([tool.to_dict() for tool in setup.tool_registry], ensure_ascii=False),
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


def parse_tool_registry(raw_value: str) -> list[ToolRegistryEntry]:
    entries: list[ToolRegistryEntry] = []
    seen: set[str] = set()
    for line in _split_tool_lines(raw_value):
        parts = [part.strip() for part in line.split("|")]
        name = _clean_tool_name(parts[0])
        if not name or name in seen:
            continue
        category = parts[1].strip() if len(parts) > 1 and parts[1].strip() else _infer_tool_category(name)
        provided_purpose = parts[2].strip() if len(parts) > 2 and parts[2].strip() else ""
        if _requires_tool_purpose(name, category) and not provided_purpose:
            raise ValueError(
                f"Tool `{name}` is ambiguous. Add a short purpose using: "
                f"{name} | category | what this tool is supposed to do"
            )
        purpose = provided_purpose or _infer_tool_purpose(name, category)
        risk = parts[3].strip() if len(parts) > 3 and parts[3].strip() else _infer_tool_risk(name, category)
        entries.append(
            ToolRegistryEntry(
                name=name,
                category=category,
                purpose=purpose,
                risk=risk,
            )
        )
        seen.add(name)
    return entries


def format_tool_registry(tool_registry: list[ToolRegistryEntry]) -> str:
    lines = []
    for tool in tool_registry:
        if tool.purpose:
            lines.append(f"{tool.name} | {tool.category} | {tool.purpose} | {tool.risk}")
        else:
            lines.append(f"{tool.name} | {tool.category} | {tool.risk}")
    return "\n".join(lines)


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


def _ensure_columns(connection: sqlite3.Connection) -> None:
    rows = connection.execute("PRAGMA table_info(runtime_agent_setup)").fetchall()
    existing = {str(row[1]) for row in rows}
    if "tool_registry" not in existing:
        connection.execute("ALTER TABLE runtime_agent_setup ADD COLUMN tool_registry TEXT NOT NULL DEFAULT '[]'")


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


def _tool_registry_from_json(raw_value: Any) -> list[ToolRegistryEntry]:
    try:
        loaded = json.loads(raw_value or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(loaded, list):
        return []
    return _normalize_tool_registry(loaded)


def _normalize_tool_registry(raw_entries: list[ToolRegistryEntry | dict[str, Any]]) -> list[ToolRegistryEntry]:
    entries: list[ToolRegistryEntry] = []
    seen: set[str] = set()
    for raw_entry in raw_entries:
        if isinstance(raw_entry, ToolRegistryEntry):
            entry = ToolRegistryEntry(
                name=_clean_tool_name(raw_entry.name),
                category=raw_entry.category,
                purpose=raw_entry.purpose,
                risk=raw_entry.risk,
                enabled=raw_entry.enabled,
            )
        elif isinstance(raw_entry, dict):
            name = _clean_tool_name(str(raw_entry.get("name") or ""))
            if not name:
                continue
            category = str(raw_entry.get("category") or _infer_tool_category(name)).strip()
            purpose = str(raw_entry.get("purpose") or _infer_tool_purpose(name, category)).strip()
            risk = str(raw_entry.get("risk") or _infer_tool_risk(name, category)).strip()
            entry = ToolRegistryEntry(
                name=name,
                category=category,
                purpose=purpose,
                risk=risk,
                enabled=bool(raw_entry.get("enabled", True)),
            )
        else:
            continue

        if not entry.name or entry.name in seen:
            continue
        entries.append(entry)
        seen.add(entry.name)
    return entries


def _split_tool_lines(raw_value: str) -> list[str]:
    lines: list[str] = []
    for raw_line in raw_value.splitlines():
        line = raw_line.strip(" -\t,")
        if not line:
            continue
        if "|" in line:
            lines.append(line)
            continue
        lines.extend(part.strip() for part in line.split(",") if part.strip())
    return lines


def _clean_tool_name(name: str) -> str:
    cleaned = str(name).strip().strip("`").strip()
    for _ in range(2):
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
            cleaned = cleaned[1:-1].strip()
    return cleaned


def _infer_tool_category(name: str) -> str:
    lowered = name.lower()
    if any(term in lowered for term in ("email", "mail", "send")):
        return "external_action"
    if any(term in lowered for term in ("search", "web", "retriev", "lookup", "law")):
        return "retrieval"
    if any(term in lowered for term in ("summar", "pdf", "document", "doc")):
        return "document_processing"
    if any(term in lowered for term in ("shell", "bash", "powershell", "python", "code", "execute", "sandbox")):
        return "code_execution"
    return "custom"


def _requires_tool_purpose(name: str, category: str) -> bool:
    lowered = name.lower()
    generic_markers = (
        "tool",
        "function",
        "action",
        "api",
        "custom",
        "handler",
        "operation",
    )
    if category == "custom":
        return True
    return any(marker in lowered for marker in generic_markers)


def _infer_tool_purpose(name: str, category: str) -> str:
    purposes = {
        "external_action": "Perform an external side effect such as sending a message.",
        "retrieval": "Retrieve information for the protected agent's allowed scope.",
        "document_processing": "Read or summarize user-provided documents.",
        "code_execution": "Execute code or interact with a runtime environment.",
        "custom": "Developer-provided tool for this agent.",
    }
    return purposes.get(category, f"Use the {name} tool.")


def _infer_tool_risk(name: str, category: str) -> str:
    if category == "code_execution":
        return "high_risk"
    if category == "external_action":
        return "external_write"
    if category in {"retrieval", "document_processing"}:
        return "read_only"
    return "needs_review"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
