from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_STATE_PATH = PROJECT_ROOT / "firewall_runtime_state.json"
DEFAULT_STATE = {"global_firewall_enabled": True}


def load_runtime_state() -> dict[str, Any]:
    if not RUNTIME_STATE_PATH.exists():
        return dict(DEFAULT_STATE)

    try:
        raw = json.loads(RUNTIME_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_STATE)

    if not isinstance(raw, dict):
        return dict(DEFAULT_STATE)

    return {
        "global_firewall_enabled": bool(
            raw.get("global_firewall_enabled", DEFAULT_STATE["global_firewall_enabled"])
        )
    }


def save_runtime_state(state: dict[str, Any]) -> None:
    normalized = {
        "global_firewall_enabled": bool(
            state.get("global_firewall_enabled", DEFAULT_STATE["global_firewall_enabled"])
        )
    }
    temp_path = RUNTIME_STATE_PATH.with_suffix(".tmp")
    temp_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(RUNTIME_STATE_PATH)


def is_global_firewall_enabled() -> bool:
    return bool(load_runtime_state().get("global_firewall_enabled", True))


def set_global_firewall_enabled(enabled: bool) -> None:
    save_runtime_state({"global_firewall_enabled": bool(enabled)})
