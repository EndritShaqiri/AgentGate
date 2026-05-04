from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass(slots=True)
class AgentConfig:
    agent_id: str
    description: str
    allowed_examples: list[str]
    denied_examples: list[str]


@dataclass(slots=True)
class ModelConfig:
    scope_model_name: str
    prompt_guard_model_name: str
    prompt_guard_max_length: int
    prompt_guard_malicious_label: str
    pii_model_name: str
    pii_enabled: bool
    llama_guard_model_name: str
    llama_guard_max_new_tokens: int
    llama_guard_fail_closed: bool
    include_recent_context: bool
    recent_context_messages: int
    recent_context_chars: int
    device: str
    attachment_chunk_chars: int
    attachment_chunk_overlap: int
    attachment_max_pages: int
    attachment_max_lg4_images: int


@dataclass(slots=True)
class ThresholdConfig:
    pg2_warn: float
    pg2_deny: float
    scope_warn: float
    scope_deny: float
    doc_pg2_warn: float
    doc_pg2_deny: float
    doc_scope_deny: float
    doc_flagged_ratio_warn: float
    final_warn: float
    final_deny: float
    pii_high: float
    denied_similarity_weight: float


@dataclass(slots=True)
class DatabaseConfig:
    path: Path
    max_rows: int


@dataclass(slots=True)
class UpstreamConfig:
    use_local_mock: bool
    base_url: str | None
    timeout_seconds: float
    default_model: str


@dataclass(slots=True)
class ProtectedRouteConfig:
    name: str
    path_pattern: str
    methods: list[str]
    content_sources: list[str]
    block_response_format: str
    block_status_code: int
    pass_through_if_unextractable: bool


@dataclass(slots=True)
class FirewallProxyConfig:
    protected_routes: list[ProtectedRouteConfig]
    default_content_sources: list[str]
    generic_block_status_code: int
    request_toggle_enabled: bool
    request_toggle_default_protect: bool
    request_toggle_sources: list[str]
    request_toggle_protect_values: list[str]
    request_toggle_bypass_values: list[str]


@dataclass(slots=True)
class LoggingConfig:
    level: str


@dataclass(slots=True)
class AppConfig:
    default_agent: str
    models: ModelConfig
    thresholds: ThresholdConfig
    database: DatabaseConfig
    upstream: UpstreamConfig
    firewall: FirewallProxyConfig
    logging: LoggingConfig
    agents: dict[str, AgentConfig]


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _default_protected_routes() -> list[dict[str, Any]]:
    return [
        {
            "name": "openai_chat_completions",
            "path_pattern": "/v1/chat/completions",
            "methods": ["POST"],
            "content_sources": ["body.messages", "body.prompt", "body.input"],
            "block_response_format": "openai_chat",
            "block_status_code": 200,
            "pass_through_if_unextractable": False,
        },
        {
            "name": "openai_responses",
            "path_pattern": "/v1/responses",
            "methods": ["POST"],
            "content_sources": ["body.input", "body.messages", "body.prompt"],
            "block_response_format": "openai_response",
            "block_status_code": 200,
            "pass_through_if_unextractable": False,
        },
    ]


def _expand_toggle_values(values: list[Any], kind: str) -> list[str]:
    truthy_aliases = {"1", "true", "on", "yes", "protect", "enabled"}
    falsy_aliases = {"0", "false", "off", "no", "bypass", "disabled"}

    expanded: set[str] = set()
    for value in values:
        normalized = str(value).strip().lower()
        if kind == "protect" and normalized in truthy_aliases:
            expanded.update(truthy_aliases)
        elif kind == "bypass" and normalized in falsy_aliases:
            expanded.update(falsy_aliases)
        elif normalized:
            expanded.add(normalized)

    if not expanded:
        expanded = truthy_aliases if kind == "protect" else falsy_aliases

    return sorted(expanded)


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    agent_configs = {
        agent_id: AgentConfig(
            agent_id=agent_id,
            description=agent_values["description"],
            allowed_examples=list(agent_values.get("allowed_examples", [])),
            denied_examples=list(agent_values.get("denied_examples", [])),
        )
        for agent_id, agent_values in raw.get("agents", {}).items()
    }

    firewall_values = raw.get("firewall", {})
    route_values = firewall_values.get("protected_routes") or _default_protected_routes()
    protected_routes = [
        ProtectedRouteConfig(
            name=str(route.get("name") or route["path_pattern"]),
            path_pattern=str(route["path_pattern"]),
            methods=[str(method).upper() for method in route.get("methods", ["POST"])],
            content_sources=list(
                route.get("content_sources")
                or firewall_values.get("default_content_sources")
                or ["body.messages", "body.input", "body.prompt", "body.question", "body.query"]
            ),
            block_response_format=str(route.get("block_response_format", "generic_json")),
            block_status_code=int(
                route.get(
                    "block_status_code",
                    firewall_values.get("generic_block_status_code", 403),
                )
            ),
            pass_through_if_unextractable=bool(route.get("pass_through_if_unextractable", True)),
        )
        for route in route_values
    ]

    return AppConfig(
        default_agent=raw["default_agent"],
        models=ModelConfig(
            scope_model_name=raw["models"]["scope_model_name"],
            prompt_guard_model_name=str(
                raw["models"].get("prompt_guard_model_name", "meta-llama/Llama-Prompt-Guard-2-86M")
            ),
            prompt_guard_max_length=int(raw["models"].get("prompt_guard_max_length", 512)),
            prompt_guard_malicious_label=str(raw["models"].get("prompt_guard_malicious_label", "MALICIOUS")),
            pii_model_name=str(raw["models"].get("pii_model_name", "dslim/bert-base-NER")),
            pii_enabled=bool(raw["models"].get("pii_enabled", True)),
            llama_guard_model_name=str(
                raw["models"].get("llama_guard_model_name", "meta-llama/Llama-Guard-4-12B")
            ),
            llama_guard_max_new_tokens=int(raw["models"].get("llama_guard_max_new_tokens", 32)),
            llama_guard_fail_closed=bool(raw["models"].get("llama_guard_fail_closed", True)),
            include_recent_context=bool(raw["models"].get("include_recent_context", False)),
            recent_context_messages=int(raw["models"].get("recent_context_messages", 3)),
            recent_context_chars=int(raw["models"].get("recent_context_chars", 600)),
            device=raw["models"].get("device", "auto"),
            attachment_chunk_chars=int(raw["models"].get("attachment_chunk_chars", 1800)),
            attachment_chunk_overlap=int(raw["models"].get("attachment_chunk_overlap", 240)),
            attachment_max_pages=int(raw["models"].get("attachment_max_pages", 12)),
            attachment_max_lg4_images=int(raw["models"].get("attachment_max_lg4_images", 4)),
        ),
        thresholds=ThresholdConfig(
            pg2_warn=float(raw["thresholds"].get("pg2_warn", 0.75)),
            pg2_deny=float(raw["thresholds"].get("pg2_deny", 0.90)),
            scope_warn=float(raw["thresholds"].get("scope_warn", 0.45)),
            scope_deny=float(raw["thresholds"].get("scope_deny", 0.30)),
            doc_pg2_warn=float(raw["thresholds"].get("doc_pg2_warn", 0.75)),
            doc_pg2_deny=float(raw["thresholds"].get("doc_pg2_deny", 0.90)),
            doc_scope_deny=float(raw["thresholds"].get("doc_scope_deny", 0.25)),
            doc_flagged_ratio_warn=float(raw["thresholds"].get("doc_flagged_ratio_warn", 0.15)),
            final_warn=float(raw["thresholds"].get("final_warn", 0.60)),
            final_deny=float(raw["thresholds"].get("final_deny", 0.78)),
            pii_high=float(raw["thresholds"].get("pii_high", 0.70)),
            denied_similarity_weight=float(raw["thresholds"]["denied_similarity_weight"]),
        ),
        database=DatabaseConfig(
            path=_resolve_path(raw["database"]["path"]),
            max_rows=int(raw["database"]["max_rows"]),
        ),
        upstream=UpstreamConfig(
            use_local_mock=bool(raw["upstream"]["use_local_mock"]),
            base_url=raw["upstream"].get("base_url") or raw["upstream"].get("url"),
            timeout_seconds=float(raw["upstream"]["timeout_seconds"]),
            default_model=raw["upstream"]["default_model"],
        ),
        firewall=FirewallProxyConfig(
            protected_routes=protected_routes,
            default_content_sources=list(
                firewall_values.get("default_content_sources")
                or ["body.messages", "body.input", "body.prompt", "body.question", "body.query"]
            ),
            generic_block_status_code=int(firewall_values.get("generic_block_status_code", 403)),
            request_toggle_enabled=bool(firewall_values.get("request_toggle_enabled", True)),
            request_toggle_default_protect=bool(firewall_values.get("request_toggle_default_protect", True)),
            request_toggle_sources=list(
                firewall_values.get("request_toggle_sources")
                or [
                    "header.x-slashid-firewall",
                    "header.x-slashid-firewall-enabled",
                    "query.slashid_firewall",
                    "query.firewall",
                    "body.metadata.slashid_firewall",
                    "body.metadata.firewall_enabled",
                    "body.firewall.enabled",
                ]
            ),
            request_toggle_protect_values=_expand_toggle_values(
                list(
                    firewall_values.get(
                        "request_toggle_protect_values",
                        ["1", "true", "on", "yes", "protect", "enabled"],
                    )
                ),
                "protect",
            ),
            request_toggle_bypass_values=_expand_toggle_values(
                list(
                    firewall_values.get(
                        "request_toggle_bypass_values",
                        ["0", "false", "off", "no", "bypass", "disabled"],
                    )
                ),
                "bypass",
            ),
        ),
        logging=LoggingConfig(level=raw["logging"]["level"]),
        agents=agent_configs,
    )
