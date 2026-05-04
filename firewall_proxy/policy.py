from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from firewall_proxy.runtime_config_store import RuntimeAgentSetup, ToolRegistryEntry


SCHEMA_VERSION = "agentgate.pbac.v1"
COMPILER_VERSION = "local-structured-policy-compiler-v1"

GENERIC_TERMS = {
    "a",
    "about",
    "access",
    "action",
    "agent",
    "allowed",
    "and",
    "api",
    "approved",
    "assistant",
    "can",
    "configured",
    "control",
    "custom",
    "data",
    "document",
    "documents",
    "do",
    "does",
    "email",
    "external",
    "file",
    "fixed",
    "for",
    "from",
    "generated",
    "help",
    "helps",
    "if",
    "information",
    "input",
    "law",
    "legal",
    "lookup",
    "message",
    "names",
    "only",
    "or",
    "pdf",
    "policy",
    "processing",
    "protected",
    "purpose",
    "read",
    "recipient",
    "registry",
    "request",
    "requests",
    "retrieve",
    "retrieval",
    "scope",
    "search",
    "send",
    "source",
    "sources",
    "summarize",
    "summary",
    "the",
    "this",
    "to",
    "tool",
    "tools",
    "upstream",
    "use",
    "user",
    "using",
    "with",
}

RETRIEVAL_TERMS = {
    "answer",
    "answers",
    "cite",
    "cites",
    "corpus",
    "find",
    "lookup",
    "question",
    "questions",
    "research",
    "retrieve",
    "retrieved",
    "search",
    "source",
    "sources",
}
DOCUMENT_TERMS = {
    "attachment",
    "attachments",
    "doc",
    "document",
    "documents",
    "file",
    "lease",
    "notice",
    "pdf",
    "read",
    "summarize",
    "summary",
    "uploaded",
}
EXTERNAL_ACTION_TERMS = {
    "deliver",
    "email",
    "forward",
    "mail",
    "message",
    "notify",
    "recipient",
    "send",
}
CODE_TERMS = {
    "bash",
    "code",
    "code_execution",
    "execute",
    "interpreter",
    "notebook",
    "powershell",
    "python",
    "run",
    "sandbox",
    "shell",
    "terminal",
}


@dataclass(slots=True)
class ToolReference:
    name: str
    source: str
    usage: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PBACDecision:
    agent_id: str
    decision: str
    trigger: str
    reason: str
    policy_id: int | None
    policy_hash: str | None
    requested_tools: list[ToolReference]
    allowed_tools: list[str]
    denied_tools: list[str]
    required_tools: list[str]
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "decision": self.decision,
            "trigger": self.trigger,
            "reason": self.reason,
            "policy_id": self.policy_id,
            "policy_hash": self.policy_hash,
            "requested_tools": [tool.to_dict() for tool in self.requested_tools],
            "allowed_tools": self.allowed_tools,
            "denied_tools": self.denied_tools,
            "required_tools": self.required_tools,
            "details": self.details,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_policy_source_hash(setup: RuntimeAgentSetup) -> str:
    payload = {
        "agent_id": setup.agent_id,
        "description": setup.description,
        "allowed_examples": setup.allowed_examples,
        "denied_examples": setup.denied_examples,
        "tool_registry": [tool.to_dict() for tool in setup.tool_registry],
    }
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def compile_policy_document(setup: RuntimeAgentSetup) -> dict[str, Any]:
    positive_text = "\n".join([setup.description, *setup.allowed_examples])
    denied_text = "\n".join(setup.denied_examples)
    source_tokens = _tokens(positive_text)
    domain_terms = sorted(
        token
        for token in source_tokens
        if token not in GENERIC_TERMS
        and token not in RETRIEVAL_TERMS
        and token not in DOCUMENT_TERMS
        and token not in EXTERNAL_ACTION_TERMS
        and token not in CODE_TERMS
    )
    domain_label = _domain_label(domain_terms)

    tool_rules: list[dict[str, Any]] = []
    allowed_by_intent: dict[str, list[str]] = {}
    denied_tools: list[dict[str, str]] = []

    for tool in setup.tool_registry:
        effect, intent_kind, conditions, reason = _classify_tool_rule(
            tool=tool,
            positive_text=positive_text,
            denied_text=denied_text,
            source_tokens=source_tokens,
            domain_terms=set(domain_terms),
        )
        if effect == "allow" and intent_kind:
            allowed_by_intent.setdefault(intent_kind, []).append(tool.name)
        else:
            denied_tools.append({"tool": tool.name, "reason": reason})

        tool_rules.append(
            {
                "tool": tool.name,
                "effect": effect,
                "intents": [_intent_name(domain_label, intent_kind)] if effect == "allow" and intent_kind else [],
                "conditions": conditions,
                "reason": reason,
            }
        )

    tool_rules.append(
        {
            "tool": "*",
            "effect": "deny",
            "intents": [],
            "conditions": [],
            "reason": "Default deny: tools are denied unless an explicit active policy rule allows the exact tool name.",
        }
    )

    intents = [
        _intent_document(domain_label, intent_kind, tool_names)
        for intent_kind, tool_names in sorted(allowed_by_intent.items())
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "agent_id": setup.agent_id,
        "source_hash": compute_policy_source_hash(setup),
        "generated_at": utc_now(),
        "default_effect": "deny",
        "compiler": {
            "type": "llm_policy_pipeline",
            "mode": "local_structured_fallback",
            "version": COMPILER_VERSION,
            "note": (
                "This deterministic compiler produces the reviewable PBAC artifact. "
                "It is the fallback stage for an LLM policy-generation pipeline and keeps runtime enforcement deterministic."
            ),
        },
        "source_summary": {
            "description_chars": len(setup.description),
            "allowed_example_count": len(setup.allowed_examples),
            "denied_example_count": len(setup.denied_examples),
            "registered_tool_count": len(setup.tool_registry),
            "domain_terms": domain_terms[:12],
        },
        "default_tool_policy": {
            "effect": "deny",
            "reason": "Default deny all tools not explicitly allowed by an active intent rule.",
        },
        "intents": intents,
        "tool_rules": tool_rules,
        "denied_tools": denied_tools,
        "runtime_enforcement": {
            "plane": "PBAC",
            "decision_type": "binary",
            "content_scores_used": False,
            "tool_gateway_required": True,
            "policy_mode": "tool_gateway_only",
        },
    }


def validate_policy_document(policy: dict[str, Any], setup: RuntimeAgentSetup) -> list[str]:
    errors: list[str] = []
    if policy.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}.")
    if policy.get("agent_id") != setup.agent_id:
        errors.append("agent_id must match the active runtime setup.")
    if policy.get("source_hash") != compute_policy_source_hash(setup):
        errors.append("source_hash does not match the active runtime setup.")
    if policy.get("default_effect") != "deny":
        errors.append("default_effect must be deny.")

    registry_names = {tool.name for tool in setup.tool_registry}
    rules = policy.get("tool_rules")
    if not isinstance(rules, list):
        errors.append("tool_rules must be a list.")
        return errors

    seen_tools: set[str] = set()
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            errors.append(f"tool_rules[{index}] must be an object.")
            continue
        tool_name = str(rule.get("tool") or "").strip()
        effect = str(rule.get("effect") or "").strip().lower()
        if not tool_name:
            errors.append(f"tool_rules[{index}].tool is required.")
            continue
        if effect not in {"allow", "deny"}:
            errors.append(f"tool_rules[{index}].effect must be allow or deny.")
        if tool_name == "*" and effect != "deny":
            errors.append("Wildcard tool rule may only deny.")
        if tool_name != "*" and tool_name not in registry_names:
            errors.append(f"tool `{tool_name}` is not present in the runtime tool_registry.")
        seen_tools.add(tool_name)

    enabled_tools = {tool.name for tool in setup.tool_registry if tool.enabled}
    missing_rules = sorted(enabled_tools - seen_tools)
    if missing_rules:
        errors.append("Every enabled registered tool needs an explicit allow or deny rule: " + ", ".join(missing_rules))
    if "*" not in seen_tools:
        errors.append("A wildcard default-deny tool rule is required.")

    return errors


def format_policy_json(policy: dict[str, Any]) -> str:
    return json.dumps(policy, ensure_ascii=False, indent=2)


def ensure_policy_tables(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS pbac_policy_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                policy_json TEXT NOT NULL,
                status TEXT NOT NULL,
                compiler_model TEXT,
                compiler_version TEXT,
                validation_errors TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                activated_at TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS pbac_decision_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                policy_id INTEGER,
                policy_hash TEXT,
                decision TEXT NOT NULL,
                trigger TEXT NOT NULL,
                reason TEXT NOT NULL,
                requested_tools TEXT NOT NULL DEFAULT '[]',
                required_tools TEXT NOT NULL DEFAULT '[]',
                allowed_tools TEXT NOT NULL DEFAULT '[]',
                denied_tools TEXT NOT NULL DEFAULT '[]',
                request_payload TEXT NOT NULL DEFAULT '{}',
                response_payload TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        _ensure_policy_columns(connection)
        connection.commit()


def _ensure_policy_columns(connection: sqlite3.Connection) -> None:
    rows = connection.execute("PRAGMA table_info(pbac_decision_logs)").fetchall()
    existing = {str(row[1]) for row in rows}
    if "required_tools" not in existing:
        connection.execute("ALTER TABLE pbac_decision_logs ADD COLUMN required_tools TEXT NOT NULL DEFAULT '[]'")


def save_active_policy_document(db_path: Path, policy: dict[str, Any], setup: RuntimeAgentSetup) -> int:
    validation_errors = validate_policy_document(policy, setup)
    if validation_errors:
        raise ValueError("Policy validation failed: " + " ".join(validation_errors))

    ensure_policy_tables(db_path)
    created_at = str(policy.get("generated_at") or utc_now())
    activated_at = utc_now()
    compiler = policy.get("compiler") if isinstance(policy.get("compiler"), dict) else {}
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            UPDATE pbac_policy_documents
            SET status = 'superseded'
            WHERE agent_id = ? AND status = 'active'
            """,
            (setup.agent_id,),
        )
        cursor = connection.execute(
            """
            INSERT INTO pbac_policy_documents (
                agent_id,
                source_hash,
                policy_json,
                status,
                compiler_model,
                compiler_version,
                validation_errors,
                created_at,
                activated_at
            ) VALUES (?, ?, ?, 'active', ?, ?, '[]', ?, ?)
            """,
            (
                setup.agent_id,
                compute_policy_source_hash(setup),
                json.dumps(policy, ensure_ascii=False),
                str(compiler.get("type") or ""),
                str(compiler.get("version") or ""),
                created_at,
                activated_at,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def load_active_policy_document(db_path: Path, agent_id: str, source_hash: str) -> tuple[int, dict[str, Any]] | None:
    ensure_policy_tables(db_path)
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT id, policy_json
            FROM pbac_policy_documents
            WHERE agent_id = ? AND source_hash = ? AND status = 'active'
            ORDER BY id DESC
            LIMIT 1
            """,
            (agent_id, source_hash),
        ).fetchone()
    if row is None:
        return None
    try:
        policy = json.loads(row[1])
    except json.JSONDecodeError:
        return None
    if not isinstance(policy, dict):
        return None
    return int(row[0]), policy


def load_latest_policy_document(db_path: Path, agent_id: str) -> dict[str, Any] | None:
    ensure_policy_tables(db_path)
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT id, source_hash, policy_json, status, created_at, activated_at
            FROM pbac_policy_documents
            WHERE agent_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (agent_id,),
        ).fetchone()
    if row is None:
        return None
    try:
        policy = json.loads(row[2])
    except json.JSONDecodeError:
        policy = {}
    return {
        "id": int(row[0]),
        "source_hash": row[1],
        "policy": policy,
        "status": row[3],
        "created_at": row[4],
        "activated_at": row[5],
    }


def evaluate_pbac_request(
    *,
    db_path: Path,
    setup: RuntimeAgentSetup | None,
    agent_id: str,
    body: Any,
    user_text: str = "",
    headers: dict[str, Any] | None = None,
) -> PBACDecision:
    if setup is None:
        return PBACDecision(
            agent_id=agent_id,
            decision="DENY",
            trigger="PBAC_SETUP_MISSING",
            reason="No runtime agent setup is available for PBAC evaluation.",
            policy_id=None,
            policy_hash=None,
            requested_tools=[],
            allowed_tools=[],
            denied_tools=[],
            required_tools=[],
            details={"content_scores_used": False},
        )

    enabled_tools = [tool for tool in setup.tool_registry if tool.enabled]
    explicit_tools = extract_tool_references(body)
    inferred_tools = infer_required_tool_references(setup, user_text, body)
    requested_tools = _dedupe_tool_references([*explicit_tools, *inferred_tools])
    source_hash = compute_policy_source_hash(setup)

    if not enabled_tools and not requested_tools:
        return _allow_decision(
            agent_id=agent_id,
            policy_id=None,
            policy_hash=None,
            requested_tools=requested_tools,
            allowed_tools=[],
            required_tools=[],
            details={"policy_required": False, "content_scores_used": False},
        )

    active = load_active_policy_document(db_path, agent_id, source_hash)
    if active is None:
        return PBACDecision(
            agent_id=agent_id,
            decision="DENY",
            trigger="PBAC_POLICY_MISSING",
            reason="No accepted PBAC policy exists for the current runtime setup.",
            policy_id=None,
            policy_hash=source_hash,
            requested_tools=requested_tools,
            allowed_tools=[],
            denied_tools=[tool.name for tool in enabled_tools],
            required_tools=[reference.name for reference in requested_tools if reference.usage != "offered"],
            details={
                "content_scores_used": False,
                "registered_tool_count": len(enabled_tools),
                "explicit_tools": [tool.to_dict() for tool in explicit_tools],
                "inferred_tools": [tool.to_dict() for tool in inferred_tools],
            },
        )

    policy_id, policy = active
    allowed_tool_names = _allowed_tools_from_policy(policy)
    registry_names = {tool.name for tool in enabled_tools}
    denied_tool_names: list[str] = []
    required_tool_names: list[str] = []

    for reference in requested_tools:
        if reference.name not in registry_names:
            denied_tool_names.append(reference.name)
            continue
        if reference.name not in allowed_tool_names:
            denied_tool_names.append(reference.name)
            continue
        if reference.usage != "offered":
            required_tool_names.append(reference.name)

    if denied_tool_names:
        denied_unique = sorted(set(denied_tool_names))
        return PBACDecision(
            agent_id=agent_id,
            decision="DENY",
            trigger="PBAC_TOOL_DENY",
            reason="PBAC denied one or more requested tools: " + ", ".join(denied_unique),
            policy_id=policy_id,
            policy_hash=source_hash,
            requested_tools=requested_tools,
            allowed_tools=sorted(allowed_tool_names),
            denied_tools=denied_unique,
            required_tools=sorted(set(required_tool_names)),
            details={
                "content_scores_used": False,
                "policy_default_effect": policy.get("default_effect", "deny"),
                "explicit_tools": [tool.to_dict() for tool in explicit_tools],
                "inferred_tools": [tool.to_dict() for tool in inferred_tools],
            },
        )

    return _allow_decision(
        agent_id=agent_id,
        policy_id=policy_id,
        policy_hash=source_hash,
        requested_tools=requested_tools,
        allowed_tools=sorted(allowed_tool_names),
        required_tools=sorted(set(required_tool_names)),
        details={
            "content_scores_used": False,
            "policy_default_effect": policy.get("default_effect", "deny"),
            "explicit_tools": [tool.to_dict() for tool in explicit_tools],
            "inferred_tools": [tool.to_dict() for tool in inferred_tools],
        },
    )


def record_pbac_decision(
    db_path: Path,
    decision: PBACDecision,
    *,
    request_payload: Any,
    response_payload: Any,
    max_rows: int = 100,
) -> None:
    ensure_policy_tables(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO pbac_decision_logs (
                created_at,
                agent_id,
                policy_id,
                policy_hash,
                decision,
                trigger,
                reason,
                requested_tools,
                required_tools,
                allowed_tools,
                denied_tools,
                request_payload,
                response_payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now(),
                decision.agent_id,
                decision.policy_id,
                decision.policy_hash,
                decision.decision,
                decision.trigger,
                decision.reason,
                json.dumps([tool.to_dict() for tool in decision.requested_tools], ensure_ascii=False),
                json.dumps(decision.required_tools, ensure_ascii=False),
                json.dumps(decision.allowed_tools, ensure_ascii=False),
                json.dumps(decision.denied_tools, ensure_ascii=False),
                json.dumps(request_payload, ensure_ascii=False),
                json.dumps(response_payload, ensure_ascii=False),
            ),
        )
        connection.execute(
            """
            DELETE FROM pbac_decision_logs
            WHERE id NOT IN (
                SELECT id
                FROM pbac_decision_logs
                ORDER BY id DESC
                LIMIT ?
            )
            """,
            (max_rows,),
        )
        connection.commit()


def load_recent_pbac_decisions(db_path: Path, max_rows: int = 20) -> list[dict[str, Any]]:
    ensure_policy_tables(db_path)
    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute(
            """
            SELECT
                id,
                created_at,
                agent_id,
                policy_id,
                decision,
                trigger,
                reason,
                requested_tools,
                denied_tools,
                required_tools
            FROM pbac_decision_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (max_rows,),
        )
        rows = cursor.fetchall()
    keys = [
        "id",
        "created_at",
        "agent_id",
        "policy_id",
        "decision",
        "trigger",
        "reason",
        "requested_tools",
        "denied_tools",
        "required_tools",
    ]
    return [dict(zip(keys, row)) for row in rows]


def extract_tool_references(body: Any) -> list[ToolReference]:
    references: list[ToolReference] = []

    if not isinstance(body, dict):
        return references

    for tool_name in _extract_offered_tools(body.get("tools")):
        references.append(ToolReference(name=tool_name, source="body.tools", usage="offered"))

    tool_choice_name = _extract_named_tool_choice(body.get("tool_choice"))
    if tool_choice_name:
        references.append(ToolReference(name=tool_choice_name, source="body.tool_choice", usage="forced"))

    for source_path in (
        "tool_name",
        "function_name",
        "action_name",
        "payload.tool_name",
        "payload.function_name",
        "tool.name",
        "function.name",
        "action.name",
    ):
        value = _nested_value(body, source_path)
        if isinstance(value, str) and value.strip():
            references.append(ToolReference(name=value.strip(), source=f"body.{source_path}", usage="execution"))

    references.extend(_extract_message_tool_calls(body.get("messages")))
    references.extend(_extract_response_input_tool_calls(body.get("input")))

    return _dedupe_tool_references(references)


def infer_required_tool_references(
    setup: RuntimeAgentSetup,
    user_text: str,
    body: Any,
) -> list[ToolReference]:
    search_space = user_text.strip() or _extract_promptish_text(body)
    tokens = _tokens(search_space)
    lowered = search_space.lower()
    required_categories: set[str] = set()

    if _has_code_execution_intent(tokens, lowered):
        required_categories.add("code_execution")
    if _has_external_action_intent(tokens, lowered):
        required_categories.add("external_action")
    if _has_document_processing_intent(tokens, lowered, body):
        required_categories.add("document_processing")
    if _has_retrieval_intent(tokens, lowered, setup):
        required_categories.add("retrieval")

    references: list[ToolReference] = []
    enabled_by_category: dict[str, list[ToolRegistryEntry]] = {}
    for tool in setup.tool_registry:
        if tool.enabled:
            enabled_by_category.setdefault(tool.category.lower().strip(), []).append(tool)

    for category in sorted(required_categories):
        category_tools = enabled_by_category.get(category) or []
        if not category_tools:
            references.append(ToolReference(name=f"<{category}>", source="pbac.intent", usage="inferred"))
            continue
        for tool in category_tools:
            references.append(ToolReference(name=tool.name, source=f"pbac.intent.{category}", usage="inferred"))

    return _dedupe_tool_references(references)


def _allow_decision(
    *,
    agent_id: str,
    policy_id: int | None,
    policy_hash: str | None,
    requested_tools: list[ToolReference],
    allowed_tools: list[str],
    required_tools: list[str],
    details: dict[str, Any],
) -> PBACDecision:
    return PBACDecision(
        agent_id=agent_id,
        decision="ALLOW",
        trigger="PBAC_ALLOW",
        reason="PBAC structural policy allowed the request.",
        policy_id=policy_id,
        policy_hash=policy_hash,
        requested_tools=requested_tools,
        allowed_tools=allowed_tools,
        denied_tools=[],
        required_tools=required_tools,
        details=details,
    )


def _allowed_tools_from_policy(policy: dict[str, Any]) -> set[str]:
    allowed: set[str] = set()
    for rule in policy.get("tool_rules", []):
        if not isinstance(rule, dict):
            continue
        if str(rule.get("effect") or "").lower() != "allow":
            continue
        tool_name = str(rule.get("tool") or "").strip()
        if tool_name and tool_name != "*":
            allowed.add(tool_name)
    return allowed


def _has_code_execution_intent(tokens: set[str], lowered: str) -> bool:
    runtime_markers = {
        "bash",
        "eval",
        "exec",
        "interpreter",
        "notebook",
        "powershell",
        "python",
        "sandbox",
        "shell",
        "subprocess",
        "terminal",
    }
    if tokens & runtime_markers:
        return True
    if "code" in tokens and tokens & {"execute", "executing", "execution", "run", "running"}:
        return True
    return any(
        phrase in lowered
        for phrase in (
            "execute this",
            "execute code",
            "run this code",
            "run code",
            "python code",
            "code interpreter",
        )
    )


def _has_external_action_intent(tokens: set[str], lowered: str) -> bool:
    if tokens & {"email", "mail", "recipient"}:
        return True
    return any(
        phrase in lowered
        for phrase in (
            "send me",
            "send it",
            "send the result",
            "send this",
            "email me",
            "email the",
            "forward the",
        )
    )


def _has_document_processing_intent(tokens: set[str], lowered: str, body: Any) -> bool:
    if _body_has_inline_attachment(body):
        return True
    document_action_terms = {"summarize", "summary", "read", "review", "extract", "flag", "compare"}
    document_object_terms = {"attachment", "attached", "doc", "document", "file", "lease", "pdf", "uploaded"}
    if tokens & document_action_terms and tokens & document_object_terms:
        return True
    return any(
        phrase in lowered
        for phrase in (
            "attached lease",
            "uploaded lease",
            "lease pdf",
            "summarize this lease",
            "read this lease",
            "housing document",
        )
    )


def _has_retrieval_intent(tokens: set[str], lowered: str, setup: RuntimeAgentSetup) -> bool:
    if tokens & {"search", "lookup", "retrieve", "research", "cite", "cites", "citation", "source", "sources"}:
        return True
    has_retrieval_tool = any(tool.enabled and tool.category.lower().strip() == "retrieval" for tool in setup.tool_registry)
    if not has_retrieval_tool:
        return False
    question_starts = ("what ", "when ", "where ", "why ", "how ", "can ", "could ", "does ", "do ", "is ", "are ")
    return "?" in lowered or lowered.strip().startswith(question_starts)


def _tokens(text: str) -> set[str]:
    normalized = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    normalized = normalized.replace("_", " ").replace("-", " ")
    values = re.findall(r"[a-zA-Z][a-zA-Z0-9]{2,}", normalized.lower())
    tokens: set[str] = set()
    for value in values:
        tokens.add(value)
        if value.endswith("ies") and len(value) > 4:
            tokens.add(value[:-3] + "y")
        elif value.endswith("s") and len(value) > 4:
            tokens.add(value[:-1])
    return tokens


def _domain_label(domain_terms: list[str]) -> str:
    if not domain_terms:
        return "agent"
    preferred = [term for term in domain_terms if len(term) > 4]
    selected = (preferred or domain_terms)[:2]
    return "_".join(selected) or "agent"


def _intent_name(domain_label: str, intent_kind: str | None) -> str:
    suffixes = {
        "retrieval": "retrieval",
        "document_processing": "document_processing",
        "external_action": "external_action",
        "code_execution": "code_execution",
    }
    return f"{domain_label}_{suffixes.get(intent_kind or '', 'intent')}"


def _intent_document(domain_label: str, intent_kind: str, tool_names: list[str]) -> dict[str, Any]:
    descriptions = {
        "retrieval": "Retrieve information or sources needed for requests inside the agent's allowed scope.",
        "document_processing": "Read or summarize user-provided documents inside the agent's allowed scope.",
        "external_action": "Perform an explicitly requested external side effect inside the agent's allowed scope.",
        "code_execution": "Execute code only when the agent policy explicitly permits code execution.",
    }
    conditions = {
        "retrieval": ["request_intent_matches_allowed_scope"],
        "document_processing": ["request_intent_matches_allowed_scope", "request_has_document_or_attachment"],
        "external_action": [
            "request_intent_matches_allowed_scope",
            "user_explicitly_requested_external_action",
            "recipient_is_configured_or_approved",
        ],
        "code_execution": ["request_intent_matches_allowed_scope", "code_execution_explicitly_required"],
    }
    return {
        "name": _intent_name(domain_label, intent_kind),
        "description": descriptions.get(intent_kind, "Allowed policy intent."),
        "allowed_tools": sorted(tool_names),
        "conditions": conditions.get(intent_kind, ["request_intent_matches_allowed_scope"]),
    }


def _classify_tool_rule(
    *,
    tool: ToolRegistryEntry,
    positive_text: str,
    denied_text: str,
    source_tokens: set[str],
    domain_terms: set[str],
) -> tuple[str, str | None, list[str], str]:
    if not tool.enabled:
        return "deny", None, [], "Tool is disabled in the runtime registry."

    category = tool.category.lower().strip()
    tool_tokens = _tokens(f"{tool.name} {tool.category} {tool.purpose} {tool.risk}")
    name_tokens = _tokens(tool.name)
    positive_tokens = source_tokens
    denied_tokens = _tokens(denied_text)
    positive_lower = positive_text.lower()

    if category == "code_execution" or "high_risk" in tool.risk.lower():
        if positive_tokens & CODE_TERMS and not (denied_tokens & CODE_TERMS):
            return (
                "allow",
                "code_execution",
                ["request_intent_matches_allowed_scope", "code_execution_explicitly_required"],
                "Positive policy examples explicitly require code execution.",
            )
        return "deny", None, [], "No allowed policy intent requires code execution."

    if category == "retrieval":
        if not (positive_tokens & RETRIEVAL_TERMS):
            return "deny", None, [], "The policy source does not describe a retrieval or search intent."
        if _tool_domain_conflicts(name_tokens, domain_terms, {"search", "web", "lookup", "retrieval", "retrieve"}):
            return "deny", None, [], "Retrieval tool name appears unrelated to the allowed policy domain."
        return (
            "allow",
            "retrieval",
            ["request_intent_matches_allowed_scope", "tool_name_in_registry"],
            "Retrieval is needed by the allowed policy and this exact tool is domain-aligned or generic.",
        )

    if category == "document_processing":
        if not (positive_tokens & DOCUMENT_TERMS):
            return "deny", None, [], "The policy source does not describe document processing."
        if _tool_domain_conflicts(
            name_tokens,
            domain_terms,
            {"summarize", "summary", "pdf", "doc", "document", "file", "uploaded"},
        ):
            return "deny", None, [], "Document tool name appears unrelated to the allowed policy domain."
        return (
            "allow",
            "document_processing",
            ["request_intent_matches_allowed_scope", "request_has_document_or_attachment", "tool_name_in_registry"],
            "Document processing is needed by the allowed policy and this exact tool is domain-aligned or generic.",
        )

    if category == "external_action":
        if not (positive_tokens & EXTERNAL_ACTION_TERMS):
            return "deny", None, [], "The policy source does not describe an external action."
        recipient_terms = name_tokens - EXTERNAL_ACTION_TERMS - {"to", "result", "summary", "send"}
        configured_recipient_allowed = any(term in positive_lower for term in ("configured recipient", "approved recipient", "fixed recipient"))
        if recipient_terms and not (recipient_terms & domain_terms) and not configured_recipient_allowed:
            return "deny", None, [], "External-action tool appears to target a recipient or purpose outside the policy source."
        if recipient_terms and configured_recipient_allowed and not any(term in tool.purpose.lower() for term in ("configured", "approved", "fixed")):
            return "deny", None, [], "External-action tool does not state that it uses a configured or approved recipient."
        return (
            "allow",
            "external_action",
            [
                "request_intent_matches_allowed_scope",
                "user_explicitly_requested_external_action",
                "recipient_is_configured_or_approved",
                "tool_name_in_registry",
            ],
            "External action is allowed only for explicit in-scope requests and configured recipients.",
        )

    if _tool_domain_conflicts(name_tokens, domain_terms, set()):
        return "deny", None, [], "Custom tool name appears unrelated to the allowed policy domain."
    return "deny", None, [], "Custom tools require manual review before they can be allowed."


def _tool_domain_conflicts(name_tokens: set[str], domain_terms: set[str], category_terms: set[str]) -> bool:
    specific_terms = name_tokens - GENERIC_TERMS - category_terms
    if not specific_terms:
        return False
    return not bool(specific_terms & domain_terms)


def _extract_offered_tools(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    names: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("function"), dict) and item["function"].get("name"):
            names.append(str(item["function"]["name"]).strip())
        elif item.get("name"):
            names.append(str(item["name"]).strip())
    return [name for name in names if name]


def _extract_named_tool_choice(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    if isinstance(value.get("function"), dict) and value["function"].get("name"):
        return str(value["function"]["name"]).strip()
    if value.get("name"):
        return str(value["name"]).strip()
    return None


def _extract_message_tool_calls(value: Any) -> list[ToolReference]:
    references: list[ToolReference] = []
    if not isinstance(value, list):
        return references
    for message_index, message in enumerate(value):
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call_index, call in enumerate(tool_calls):
            if not isinstance(call, dict):
                continue
            name = None
            if isinstance(call.get("function"), dict):
                name = call["function"].get("name")
            if not name:
                name = call.get("name")
            if isinstance(name, str) and name.strip():
                references.append(
                    ToolReference(
                        name=name.strip(),
                        source=f"body.messages[{message_index}].tool_calls[{call_index}]",
                        usage="execution",
                    )
                )
    return references


def _extract_response_input_tool_calls(value: Any) -> list[ToolReference]:
    references: list[ToolReference] = []

    def walk(current: Any, path: str) -> None:
        if isinstance(current, dict):
            item_type = str(current.get("type") or "").lower()
            if item_type in {"function_call", "tool_call"} and isinstance(current.get("name"), str):
                references.append(ToolReference(name=current["name"].strip(), source=path, usage="execution"))
            for key, child in current.items():
                walk(child, f"{path}.{key}" if path else str(key))
            return
        if isinstance(current, list):
            for index, child in enumerate(current):
                walk(child, f"{path}[{index}]")

    walk(value, "body.input")
    return [reference for reference in references if reference.name]


def _dedupe_tool_references(references: list[ToolReference]) -> list[ToolReference]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[ToolReference] = []
    for reference in references:
        key = (reference.name, reference.source, reference.usage)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(reference)
    return deduped


def _extract_promptish_text(value: Any) -> str:
    if isinstance(value, dict):
        messages = value.get("messages")
        if isinstance(messages, list):
            parts: list[str] = []
            for message in messages:
                if not isinstance(message, dict):
                    continue
                if message.get("role") not in (None, "user"):
                    continue
                content = message.get("content")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and isinstance(item.get("text"), str):
                            parts.append(item["text"])
                        elif isinstance(item, str):
                            parts.append(item)
            if parts:
                return "\n".join(parts)

        for key in ("prompt", "question", "query", "message", "text", "user_request", "intent"):
            if isinstance(value.get(key), str):
                return str(value[key])
        if "input" in value:
            return _extract_promptish_text(value["input"])
        return ""

    if isinstance(value, list):
        parts = [_extract_promptish_text(item) for item in value]
        return "\n".join(part for part in parts if part)

    return value if isinstance(value, str) else ""


def _body_has_inline_attachment(value: Any) -> bool:
    if isinstance(value, dict):
        if any(key in value for key in ("file_data", "data", "content_base64", "base64", "bytes", "image_url")):
            item_type = str(value.get("type") or "").lower()
            if item_type in {"input_file", "file", "attachment", "document", "input_image", "image", "image_url"}:
                return True
            if any(key in value for key in ("filename", "file_name", "name", "media_type", "mime_type")):
                return True
        return any(_body_has_inline_attachment(child) for child in value.values())
    if isinstance(value, list):
        return any(_body_has_inline_attachment(child) for child in value)
    return False


def _nested_value(root: Any, dotted_path: str) -> Any:
    current = root
    for segment in dotted_path.split("."):
        if isinstance(current, dict):
            current = current.get(segment)
            continue
        return None
    return current
