from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qs

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from firewall_proxy.config import AppConfig, ProtectedRouteConfig, load_config
from firewall_proxy.firewall import (
    FirewallDecision,
    FirewallEngine,
    STRUCTURED_PII_PATTERNS,
    extract_first_matching_value,
    extract_user_text_from_sources,
    mask_sensitive_value,
    normalize_user_message,
)
from firewall_proxy.log_store import AsyncSQLiteLogger, FirewallLogRecord
from firewall_proxy.policy import (
    PBACDecision,
    evaluate_pbac_request,
    record_pbac_decision,
)
from firewall_proxy.runtime_config_store import clear_runtime_agent_setup, load_runtime_agent_setup
from firewall_proxy.runtime_state import is_global_firewall_enabled


LOGGER = logging.getLogger("ai_firewall_proxy")
HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
ATTACHMENT_PAYLOAD_KEYS = {
    "file_data",
    "data",
    "content_base64",
    "base64",
    "bytes",
    "image_url",
}


def sanitize_payload_for_logs(payload: Any, decision: FirewallDecision | None = None) -> Any:
    if isinstance(payload, dict):
        sanitized: dict[str, Any] = {}
        for key, value in payload.items():
            key_lower = str(key).lower()
            if key_lower in ATTACHMENT_PAYLOAD_KEYS and isinstance(value, str) and len(value) > 200:
                sanitized[key] = f"[redacted attachment payload: {len(value)} chars]"
                continue
            sanitized[key] = sanitize_payload_for_logs(value, decision)
        return sanitized

    if isinstance(payload, list):
        return [sanitize_payload_for_logs(item, decision) for item in payload]

    if isinstance(payload, str):
        if payload.startswith("data:") and len(payload) > 200:
            return f"[redacted inline data payload: {len(payload)} chars]"
        redacted = _redact_structured_pii(payload)
        if decision and decision.prompt_injection.evaluated_text:
            original = decision.prompt_injection.evaluated_text
            if original in redacted and decision.latest_user_message != original:
                redacted = redacted.replace(original, decision.latest_user_message)
        return redacted

    return payload


def _redact_structured_pii(text: str) -> str:
    redacted = text
    for pattern in STRUCTURED_PII_PATTERNS.values():
        matches = list(pattern.finditer(redacted))
        for match in reversed(matches):
            redacted = (
                redacted[: match.start()]
                + mask_sensitive_value(match.group(0))
                + redacted[match.end() :]
            )
    return redacted


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(message)s")


def log_event(event: str, **fields: Any) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    LOGGER.info(json.dumps(payload, ensure_ascii=False))


def build_firewall_details(decision: FirewallDecision) -> dict[str, Any]:
    return {
        "id": f"fw-{uuid.uuid4().hex}",
        "object": "firewall.decision",
        "created": int(time.time()),
        "agent_id": decision.agent_id,
        "decision": decision.decision,
        "forwarded": False,
        "trigger_layer": decision.trigger_layer,
        "message": decision.reason,
        "scores": {
            "scope_main": decision.risk.scope_main,
            "pg2_main": decision.risk.pg2_main,
            "pii_main": decision.risk.pii_main,
            "doc_pg2_max": decision.risk.doc_pg2_max,
            "doc_scope_min": decision.risk.doc_scope_min,
            "doc_flagged_ratio": decision.risk.doc_flagged_ratio,
            "lg4_unsafe": decision.risk.lg4_unsafe,
            "lg4_code_abuse": decision.risk.lg4_code_abuse,
            "final_risk": decision.risk.final_risk,
        },
        "details": {
            "scope": decision.scope.to_dict(),
            "prompt_injection": decision.prompt_injection.to_dict(),
            "pii": decision.pii.to_dict(),
            "attachments": decision.attachments.to_dict(),
            "llama_guard": decision.llama_guard.to_dict(),
            "risk": decision.risk.to_dict(),
            "model_versions": decision.model_versions,
        },
    }


def build_block_message(decision: FirewallDecision) -> str:
    return f"Request blocked by firewall: {decision.reason}"


def build_chat_block_response(
    decision: FirewallDecision,
    request_body: Any,
) -> dict[str, Any]:
    content = build_block_message(decision)
    completion_tokens = len(content.split())
    model_name = request_body.get("model") if isinstance(request_body, dict) else None
    return {
        "id": f"chatcmpl-fw-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name or app.state.config.upstream.default_model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": completion_tokens,
            "total_tokens": completion_tokens,
        },
        "firewall": build_firewall_details(decision),
    }


def build_responses_block_response(
    decision: FirewallDecision,
    request_body: Any,
) -> dict[str, Any]:
    content = build_block_message(decision)
    output_tokens = len(content.split())
    model_name = request_body.get("model") if isinstance(request_body, dict) else None
    return {
        "id": f"resp-fw-{uuid.uuid4().hex[:12]}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model_name or app.state.config.upstream.default_model,
        "output": [
            {
                "id": f"msg-fw-{uuid.uuid4().hex[:12]}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                        "annotations": [],
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": 0,
            "output_tokens": output_tokens,
            "total_tokens": output_tokens,
        },
        "firewall": build_firewall_details(decision),
    }


def build_generic_block_response(
    decision: FirewallDecision,
    request_body: Any,
) -> dict[str, Any]:
    return {
        "id": f"fw-block-{uuid.uuid4().hex[:12]}",
        "object": "firewall.block",
        "status": "blocked",
        "message": build_block_message(decision),
        "request": request_body if isinstance(request_body, dict) else {"raw_body": str(request_body)[:2000]},
        "firewall": build_firewall_details(decision),
    }


def build_pbac_details(decision: PBACDecision) -> dict[str, Any]:
    return {
        "id": f"pbac-{uuid.uuid4().hex}",
        "object": "pbac.decision",
        "created": int(time.time()),
        "agent_id": decision.agent_id,
        "decision": decision.decision,
        "forwarded": False,
        "trigger": decision.trigger,
        "message": decision.reason,
        "policy_id": decision.policy_id,
        "policy_hash": decision.policy_hash,
        "requested_tools": [tool.to_dict() for tool in decision.requested_tools],
        "allowed_tools": decision.allowed_tools,
        "denied_tools": decision.denied_tools,
        "required_tools": decision.required_tools,
        "details": {
            **decision.details,
            "plane": "PBAC",
            "decision_type": "binary",
            "content_scores_used": False,
        },
    }


def build_pbac_block_message(decision: PBACDecision) -> str:
    return f"Request blocked by PBAC policy: {decision.reason}"


def build_pbac_chat_block_response(decision: PBACDecision, request_body: Any) -> dict[str, Any]:
    content = build_pbac_block_message(decision)
    completion_tokens = len(content.split())
    model_name = request_body.get("model") if isinstance(request_body, dict) else None
    return {
        "id": f"chatcmpl-pbac-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name or app.state.config.upstream.default_model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": completion_tokens,
            "total_tokens": completion_tokens,
        },
        "pbac": build_pbac_details(decision),
    }


def build_pbac_responses_block_response(decision: PBACDecision, request_body: Any) -> dict[str, Any]:
    content = build_pbac_block_message(decision)
    output_tokens = len(content.split())
    model_name = request_body.get("model") if isinstance(request_body, dict) else None
    return {
        "id": f"resp-pbac-{uuid.uuid4().hex[:12]}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model_name or app.state.config.upstream.default_model,
        "output": [
            {
                "id": f"msg-pbac-{uuid.uuid4().hex[:12]}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                        "annotations": [],
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": 0,
            "output_tokens": output_tokens,
            "total_tokens": output_tokens,
        },
        "pbac": build_pbac_details(decision),
    }


def build_pbac_generic_block_response(decision: PBACDecision, request_body: Any) -> dict[str, Any]:
    return {
        "id": f"pbac-block-{uuid.uuid4().hex[:12]}",
        "object": "pbac.block",
        "status": "blocked",
        "message": build_pbac_block_message(decision),
        "request": request_body if isinstance(request_body, dict) else {"raw_body": str(request_body)[:2000]},
        "pbac": build_pbac_details(decision),
    }


def resolve_agent_id(body: Any, request: Request, config: AppConfig) -> str:
    metadata = body.get("metadata") if isinstance(body, dict) and isinstance(body.get("metadata"), dict) else {}
    requested_agent = (
        request.headers.get("x-agent-id")
        or metadata.get("agent_id")
        or (body.get("agent_id") if isinstance(body, dict) else None)
        or config.default_agent
    )

    if requested_agent not in config.agents:
        available = ", ".join(sorted(config.agents))
        if not available:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown agent_id '{requested_agent}'. No runtime agent setup is configured yet. "
                    "Create one in the dashboard Runtime Agent Setup panel first."
                ),
            )
        raise HTTPException(
            status_code=400,
            detail=f"Unknown agent_id '{requested_agent}'. Available agents: {available}",
        )

    return requested_agent


def build_upstream_url(request: Request, config: AppConfig) -> str:
    if config.upstream.use_local_mock:
        base_url = str(request.base_url).rstrip("/")
        return f"{base_url}/mock{request.url.path}"

    if not config.upstream.base_url:
        raise RuntimeError("No upstream base URL configured.")

    upstream_base = httpx.URL(config.upstream.base_url)
    base_path = upstream_base.path.rstrip("/")
    request_path = request.url.path

    if base_path and request_path.startswith(f"{base_path}/"):
        upstream_path = request_path
    elif base_path and request_path == base_path:
        upstream_path = request_path
    elif base_path:
        upstream_path = f"{base_path}{request_path}"
    else:
        upstream_path = request_path

    return str(upstream_base.copy_with(path=upstream_path))


def forwardable_headers(request: Request) -> dict[str, str]:
    headers: dict[str, str] = {}
    for name, value in request.headers.items():
        if name.lower() in HOP_BY_HOP_HEADERS:
            continue
        headers[name] = value

    client_host = request.client.host if request.client else "unknown"
    headers["x-forwarded-for"] = client_host
    headers["x-forwarded-proto"] = request.url.scheme
    headers["x-forwarded-host"] = request.headers.get("host", "")
    return headers


async def forward_upstream(
    request: Request,
    config: AppConfig,
    extra_headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    upstream_url = build_upstream_url(request, config)
    request_body = await request.body()
    request_headers = forwardable_headers(request)
    if extra_headers:
        request_headers.update(extra_headers)

    async with httpx.AsyncClient(timeout=config.upstream.timeout_seconds) as client:
        upstream_response = await client.request(
            method=request.method,
            url=upstream_url,
            params=request.query_params,
            content=request_body,
            headers=request_headers,
        )

    response_headers = {
        name: value
        for name, value in upstream_response.headers.items()
        if name.lower() not in HOP_BY_HOP_HEADERS
    }
    return upstream_response.status_code, response_headers, upstream_response.content


def parse_response_payload(response_headers: dict[str, str], response_body: bytes) -> Any:
    content_type = response_headers.get("content-type", "")
    if "application/json" not in content_type.lower():
        return {"raw_text": response_body.decode("utf-8", errors="replace")}

    try:
        return json.loads(response_body)
    except ValueError:
        return {"raw_text": response_body.decode("utf-8", errors="replace")}


def build_proxy_response(
    status_code: int,
    response_headers: dict[str, str],
    response_body: bytes,
    *,
    firewall_decision: str = "ALLOW",
    forwarded: bool = True,
) -> Response:
    headers = {
        name: value
        for name, value in response_headers.items()
        if name.lower() != "content-length"
    }
    headers["x-firewall-decision"] = firewall_decision
    headers["x-firewall-forwarded"] = str(forwarded).lower()
    return Response(content=response_body, status_code=status_code, headers=headers)


def should_apply_firewall(
    *,
    body: Any,
    request: Request,
    config: AppConfig,
) -> tuple[bool, str]:
    if not is_global_firewall_enabled():
        return False, "Firewall bypassed by global dashboard toggle."

    if not config.firewall.request_toggle_enabled:
        return True, "Firewall toggle disabled in config."

    toggle_value = extract_first_matching_value(
        body=body,
        query_params=dict(request.query_params),
        headers=dict(request.headers),
        sources=config.firewall.request_toggle_sources,
    )
    if toggle_value is None:
        if config.firewall.request_toggle_default_protect:
            return True, "No toggle supplied; defaulting to protected."
        return False, "No toggle supplied; defaulting to bypass."

    normalized = str(toggle_value).strip().lower()
    if normalized in config.firewall.request_toggle_bypass_values:
        return False, f"Bypassed by request toggle value '{toggle_value}'."
    if normalized in config.firewall.request_toggle_protect_values:
        return True, f"Protected by request toggle value '{toggle_value}'."

    if config.firewall.request_toggle_default_protect:
        return True, f"Unrecognized toggle value '{toggle_value}'; defaulting to protected."
    return False, f"Unrecognized toggle value '{toggle_value}'; defaulting to bypass."


def build_firewall_sdk_response(
    route_config: ProtectedRouteConfig,
    request_body: Any,
    decision: FirewallDecision,
) -> dict[str, Any]:
    response_format = route_config.block_response_format.lower()

    if response_format == "openai_response":
        return build_responses_block_response(decision, request_body)

    if response_format == "openai_chat":
        return build_chat_block_response(decision, request_body)

    if response_format == "auto" and isinstance(request_body, dict):
        if "input" in request_body and "messages" not in request_body:
            return build_responses_block_response(decision, request_body)
        if "messages" in request_body:
            return build_chat_block_response(decision, request_body)

    return build_generic_block_response(decision, request_body)


def build_pbac_sdk_response(
    route_config: ProtectedRouteConfig,
    request_body: Any,
    decision: PBACDecision,
) -> dict[str, Any]:
    response_format = route_config.block_response_format.lower()

    if response_format == "openai_response":
        return build_pbac_responses_block_response(decision, request_body)

    if response_format == "openai_chat":
        return build_pbac_chat_block_response(decision, request_body)

    if response_format == "auto" and isinstance(request_body, dict):
        if "input" in request_body and "messages" not in request_body:
            return build_pbac_responses_block_response(decision, request_body)
        if "messages" in request_body:
            return build_pbac_chat_block_response(decision, request_body)

    return build_pbac_generic_block_response(decision, request_body)


def parse_request_body(request: Request, body_bytes: bytes) -> Any:
    content_type = request.headers.get("content-type", "").lower()
    if not body_bytes:
        return {}

    if "application/json" in content_type:
        try:
            return json.loads(body_bytes)
        except ValueError:
            return {"raw_body": body_bytes.decode("utf-8", errors="replace")}

    if "application/x-www-form-urlencoded" in content_type:
        parsed = parse_qs(body_bytes.decode("utf-8", errors="replace"), keep_blank_values=True)
        return {
            key: values[0] if len(values) == 1 else values
            for key, values in parsed.items()
        }

    return {"raw_body": body_bytes.decode("utf-8", errors="replace")}


def request_payload_for_logs(parsed_body: Any) -> Any:
    if isinstance(parsed_body, dict):
        return parsed_body
    return {"raw_body": str(parsed_body)}


def extract_tool_gateway_request(body: Any) -> tuple[str, Any]:
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Tool gateway body must be a JSON object.")

    tool_name = (
        body.get("tool_name")
        or body.get("name")
        or body.get("function_name")
        or (
            body.get("tool", {}).get("name")
            if isinstance(body.get("tool"), dict)
            else None
        )
    )
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise HTTPException(status_code=400, detail="Tool gateway requires `tool_name`.")

    arguments = body.get("arguments")
    if arguments is None:
        arguments = body.get("args", {})

    return tool_name.strip(), arguments


def build_mock_tool_result(tool_name: str, arguments: Any) -> dict[str, Any]:
    return {
        "status": "ok",
        "mock": True,
        "tool_name": tool_name,
        "arguments": arguments if isinstance(arguments, dict) else {"value": arguments},
        "message": "PBAC allowed this tool call. Local mock mode returned a mock tool result.",
    }


def build_tool_gateway_response(
    *,
    tool_name: str,
    arguments: Any,
    decision: PBACDecision,
    config: AppConfig,
) -> tuple[int, dict[str, Any]]:
    if config.upstream.use_local_mock:
        return 200, {
            "status": "ok",
            "tool_name": tool_name,
            "result": build_mock_tool_result(tool_name, arguments),
            "pbac": build_pbac_details(decision),
        }

    return 501, {
        "status": "authorized_but_not_executed",
        "tool_name": tool_name,
        "message": (
            "PBAC allowed this tool call, but no real tool executor is configured inside AgentGate. "
            "Route your tool implementation through this endpoint or add an executor adapter here."
        ),
        "pbac": build_pbac_details(decision),
    }


def decision_payload_for_logs(decision: FirewallDecision) -> dict[str, Any]:
    return {
        "model_versions": decision.model_versions,
        "scope": decision.scope.to_dict(),
        "prompt_injection": decision.prompt_injection.to_dict(),
        "pii": decision.pii.to_dict(),
        "attachments": decision.attachments.to_dict(),
        "llama_guard": decision.llama_guard.to_dict(),
        "risk": decision.risk.to_dict(),
    }


def log_record_from_decision(
    *,
    decision: FirewallDecision,
    request_body: Any,
    response_payload: Any,
) -> FirewallLogRecord:
    raw_scores = decision_payload_for_logs(decision)
    return FirewallLogRecord(
        created_at=datetime.now(timezone.utc).isoformat(),
        agent_id=decision.agent_id,
        user_input=decision.latest_user_message,
        decision=decision.decision,
        trigger_layer=decision.trigger_layer,
        scope_score=decision.risk.scope_main,
        prompt_injection_score=decision.risk.pg2_main,
        raw_scores=raw_scores,
        request_payload=sanitize_payload_for_logs(request_payload_for_logs(request_body), decision),
        response_payload=response_payload,
        pg2_main=decision.risk.pg2_main,
        scope_main=decision.risk.scope_main,
        pii_main=decision.risk.pii_main,
        doc_pg2_max=decision.risk.doc_pg2_max,
        doc_scope_min=decision.risk.doc_scope_min,
        doc_flagged_ratio=decision.risk.doc_flagged_ratio,
        lg4_unsafe=decision.risk.lg4_unsafe,
        lg4_code_abuse=decision.risk.lg4_code_abuse,
        final_risk=decision.risk.final_risk,
        attachment_summary=decision.attachments.summary,
        decision_reasons=decision.risk.reasons,
        chunk_summaries=[chunk.to_log_dict() for chunk in decision.attachments.chunks],
        model_versions=decision.model_versions,
    )


def match_protected_route(request: Request, config: AppConfig) -> ProtectedRouteConfig | None:
    request_path = request.url.path
    request_method = request.method.upper()

    for route in config.firewall.protected_routes:
        if route.methods and request_method not in route.methods:
            continue
        if fnmatch.fnmatchcase(request_path, route.path_pattern):
            return route

    return None


def configured_protected_endpoints(config: AppConfig) -> list[str]:
    return sorted(f"{','.join(route.methods)} {route.path_pattern}" for route in config.firewall.protected_routes)


def refresh_runtime_agent_setup(config: AppConfig) -> bool:
    setup = load_runtime_agent_setup(config.database.path)
    if setup is None:
        config.agents = {}
        return False

    config.default_agent = setup.agent_id
    config.agents = {setup.agent_id: setup.to_agent_config()}
    config.upstream = setup.to_upstream_config()
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    configure_logging(config.logging.level)
    runtime_setup_configured = refresh_runtime_agent_setup(config)

    engine = FirewallEngine(config)
    logger = AsyncSQLiteLogger(config.database.path, config.database.max_rows)

    await logger.start()
    try:
        await asyncio.gather(
            asyncio.to_thread(engine.scope_service.initialize),
            asyncio.to_thread(engine.prompt_injection_service.initialize),
            asyncio.to_thread(engine.pii_service.initialize),
        )
    except Exception:
        await logger.stop()
        raise

    app.state.config = config
    app.state.engine = engine
    app.state.db_logger = logger

    log_event(
        "startup.complete",
        default_agent=config.default_agent,
        agents=list(config.agents),
        runtime_setup_configured=runtime_setup_configured,
        database_path=str(config.database.path),
    )

    try:
        yield
    finally:
        await logger.stop()
        clear_runtime_agent_setup(config.database.path)
        log_event("shutdown.complete")


app = FastAPI(title="AI Firewall Proxy", version="1.0.0", lifespan=lifespan)


@app.get("/")
async def root() -> dict[str, Any]:
    config: AppConfig = app.state.config
    runtime_setup_configured = refresh_runtime_agent_setup(config)
    app.state.engine.scope_service.refresh_profiles_if_changed()
    return {
        "service": "ai_firewall_proxy",
        "status": "ok",
        "runtime_setup_configured": runtime_setup_configured,
        "default_agent": config.default_agent,
        "upstream_mode": "local_mock" if config.upstream.use_local_mock else "remote",
        "upstream_base_url": config.upstream.base_url,
        "upstream_default_model": config.upstream.default_model,
        "protected_endpoints": configured_protected_endpoints(config),
        "request_toggle_enabled": config.firewall.request_toggle_enabled,
        "global_firewall_enabled": is_global_firewall_enabled(),
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    config: AppConfig = app.state.config
    runtime_setup_configured = refresh_runtime_agent_setup(config)
    app.state.engine.scope_service.refresh_profiles_if_changed()
    return {
        "status": "ok",
        "runtime_setup_configured": runtime_setup_configured,
        "default_agent": config.default_agent,
        "agents": list(config.agents),
        "upstream_mode": "local_mock" if config.upstream.use_local_mock else "remote",
        "upstream_base_url": config.upstream.base_url,
        "upstream_default_model": config.upstream.default_model,
        "database": str(config.database.path),
        "protected_endpoints": configured_protected_endpoints(config),
        "request_toggle_enabled": config.firewall.request_toggle_enabled,
        "request_toggle_default_protect": config.firewall.request_toggle_default_protect,
        "global_firewall_enabled": is_global_firewall_enabled(),
    }


async def handle_protected_request(request: Request, route_config: ProtectedRouteConfig) -> Response:
    body_bytes = await request.body()
    body = parse_request_body(request, body_bytes)

    config: AppConfig = app.state.config
    engine: FirewallEngine = app.state.engine
    db_logger: AsyncSQLiteLogger = app.state.db_logger

    refresh_runtime_agent_setup(config)
    engine.scope_service.refresh_profiles_if_changed()
    agent_id = resolve_agent_id(body, request, config)
    runtime_setup = load_runtime_agent_setup(config.database.path)
    try:
        pbac_user_text = normalize_user_message(
            extract_user_text_from_sources(
                body=body,
                query_params=dict(request.query_params),
                headers=dict(request.headers),
                content_sources=route_config.content_sources,
            )
        )
    except ValueError:
        pbac_user_text = ""
    pbac_decision = evaluate_pbac_request(
        db_path=config.database.path,
        setup=runtime_setup,
        agent_id=agent_id,
        body=body,
        user_text=pbac_user_text,
        headers=dict(request.headers),
    )

    if pbac_decision.decision == "DENY":
        pbac_response = build_pbac_sdk_response(route_config, body, pbac_decision)
        record_pbac_decision(
            config.database.path,
            pbac_decision,
            request_payload=sanitize_payload_for_logs(request_payload_for_logs(body)),
            response_payload=pbac_response,
        )
        log_event(
            "pbac.blocked",
            agent_id=agent_id,
            route_name=route_config.name,
            path=request.url.path,
            trigger=pbac_decision.trigger,
            denied_tools=pbac_decision.denied_tools,
        )
        response = JSONResponse(status_code=route_config.block_status_code, content=pbac_response)
        response.headers["x-pbac-decision"] = pbac_decision.decision
        response.headers["x-pbac-trigger"] = pbac_decision.trigger
        response.headers["x-firewall-decision"] = "NOT_EVALUATED"
        response.headers["x-firewall-forwarded"] = "false"
        response.headers["x-firewall-bypassed"] = "false"
        return response

    should_protect, toggle_reason = should_apply_firewall(body=body, request=request, config=config)

    if not should_protect:
        try:
            bypass_user_input = normalize_user_message(
                extract_user_text_from_sources(
                    body=body,
                    query_params=dict(request.query_params),
                    headers=dict(request.headers),
                    content_sources=route_config.content_sources,
                )
            )
        except ValueError:
            bypass_user_input = "Firewall bypassed by request toggle."

        try:
            status_code, response_headers, upstream_body = await forward_upstream(request, config)
        except httpx.HTTPError as exc:
            log_event("upstream.error", agent_id=agent_id, error=str(exc))
            raise HTTPException(status_code=502, detail=f"Failed to reach upstream endpoint: {exc}") from exc

        upstream_payload = parse_response_payload(response_headers, upstream_body)
        db_logger.enqueue(
            FirewallLogRecord(
                created_at=datetime.now(timezone.utc).isoformat(),
                agent_id=agent_id,
                user_input=bypass_user_input,
                decision="BYPASS",
                trigger_layer=(
                    "GLOBAL_TOGGLE"
                    if "global dashboard toggle" in toggle_reason.lower()
                    else "USER_TOGGLE"
                ),
                scope_score=0.0,
                prompt_injection_score=0.0,
                raw_scores={"firewall_toggle": {"applied": False, "reason": toggle_reason}},
                request_payload=sanitize_payload_for_logs(request_payload_for_logs(body)),
                response_payload=upstream_payload,
                decision_reasons=[toggle_reason],
            )
        )
        record_pbac_decision(
            config.database.path,
            pbac_decision,
            request_payload=sanitize_payload_for_logs(request_payload_for_logs(body)),
            response_payload=upstream_payload,
        )
        log_event(
            "firewall.bypass",
            agent_id=agent_id,
            route_name=route_config.name,
            path=request.url.path,
            reason=toggle_reason,
        )
        response = build_proxy_response(
            status_code,
            response_headers,
            upstream_body,
            firewall_decision="BYPASS",
            forwarded=True,
        )
        response.headers["x-firewall-bypassed"] = "true"
        response.headers["x-pbac-decision"] = pbac_decision.decision
        response.headers["x-pbac-trigger"] = pbac_decision.trigger
        return response

    try:
        latest_user_message = normalize_user_message(
            extract_user_text_from_sources(
                body=body,
                query_params=dict(request.query_params),
                headers=dict(request.headers),
                content_sources=route_config.content_sources,
            )
        )
    except ValueError as exc:
        if route_config.pass_through_if_unextractable:
            log_event(
                "firewall.skip",
                path=request.url.path,
                route_name=route_config.name,
                reason=str(exc),
            )
            try:
                status_code, response_headers, upstream_body = await forward_upstream(request, config)
            except httpx.HTTPError as upstream_exc:
                log_event("upstream.error", path=request.url.path, error=str(upstream_exc))
                raise HTTPException(
                    status_code=502,
                    detail=f"Failed to reach upstream endpoint: {upstream_exc}",
                ) from upstream_exc

            upstream_payload = parse_response_payload(response_headers, upstream_body)
            record_pbac_decision(
                config.database.path,
                pbac_decision,
                request_payload=sanitize_payload_for_logs(request_payload_for_logs(body)),
                response_payload=upstream_payload,
            )
            response = build_proxy_response(status_code, response_headers, upstream_body)
            response.headers["x-firewall-bypassed"] = "false"
            response.headers["x-pbac-decision"] = pbac_decision.decision
            response.headers["x-pbac-trigger"] = pbac_decision.trigger
            return response

        raise HTTPException(status_code=400, detail=str(exc)) from exc

    messages_for_scoring = body.get("messages") if isinstance(body, dict) else None
    if not isinstance(messages_for_scoring, list):
        messages_for_scoring = [{"role": "user", "content": latest_user_message}]

    try:
        decision = await asyncio.to_thread(engine.evaluate, agent_id, messages_for_scoring, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if decision.decision == "ALLOW":
        try:
            status_code, response_headers, upstream_body = await forward_upstream(request, config)
        except httpx.HTTPError as exc:
            log_event("upstream.error", agent_id=agent_id, error=str(exc))
            raise HTTPException(status_code=502, detail=f"Failed to reach upstream endpoint: {exc}") from exc

        upstream_payload = parse_response_payload(response_headers, upstream_body)

        db_logger.enqueue(
            log_record_from_decision(
                decision=decision,
                request_body=body,
                response_payload=upstream_payload,
            )
        )
        record_pbac_decision(
            config.database.path,
            pbac_decision,
            request_payload=sanitize_payload_for_logs(request_payload_for_logs(body), decision),
            response_payload=upstream_payload,
        )
        log_event(
            "firewall.allow",
            agent_id=agent_id,
            route_name=route_config.name,
            path=request.url.path,
            scope_main=decision.risk.scope_main,
            pg2_main=decision.risk.pg2_main,
            final_risk=decision.risk.final_risk,
        )
        response = build_proxy_response(
            status_code,
            response_headers,
            upstream_body,
            firewall_decision=decision.decision,
            forwarded=True,
        )
        response.headers["x-firewall-bypassed"] = "false"
        response.headers["x-pbac-decision"] = pbac_decision.decision
        response.headers["x-pbac-trigger"] = pbac_decision.trigger
        return response

    firewall_response = build_firewall_sdk_response(route_config, body, decision)
    db_logger.enqueue(
        log_record_from_decision(
            decision=decision,
            request_body=body,
            response_payload=firewall_response,
        )
    )
    record_pbac_decision(
        config.database.path,
        pbac_decision,
        request_payload=sanitize_payload_for_logs(request_payload_for_logs(body), decision),
        response_payload=firewall_response,
    )
    log_event(
        "firewall.blocked",
        agent_id=agent_id,
        route_name=route_config.name,
        path=request.url.path,
        decision=decision.decision,
        trigger_layer=decision.trigger_layer,
        scope_main=decision.risk.scope_main,
        pg2_main=decision.risk.pg2_main,
        final_risk=decision.risk.final_risk,
    )
    response = JSONResponse(status_code=route_config.block_status_code, content=firewall_response)
    response.headers["x-firewall-decision"] = decision.decision
    response.headers["x-firewall-forwarded"] = "false"
    response.headers["x-firewall-bypassed"] = "false"
    response.headers["x-pbac-decision"] = pbac_decision.decision
    response.headers["x-pbac-trigger"] = pbac_decision.trigger
    return response


@app.post("/agentgate/tools/execute", name="agentgate_tool_execute")
async def agentgate_tool_execute(request: Request) -> Response:
    body_bytes = await request.body()
    body = parse_request_body(request, body_bytes)

    config: AppConfig = app.state.config
    refresh_runtime_agent_setup(config)
    runtime_setup = load_runtime_agent_setup(config.database.path)
    agent_id = resolve_agent_id(body, request, config)
    tool_name, arguments = extract_tool_gateway_request(body)
    tool_body = {
        "tool_name": tool_name,
        "arguments": arguments,
    }
    user_text = str(
        body.get("user_request")
        or body.get("intent")
        or body.get("reason")
        or tool_name
    ) if isinstance(body, dict) else tool_name
    pbac_decision = evaluate_pbac_request(
        db_path=config.database.path,
        setup=runtime_setup,
        agent_id=agent_id,
        body=tool_body,
        user_text=user_text,
        headers=dict(request.headers),
    )

    if pbac_decision.decision == "DENY":
        response_payload = {
            "status": "blocked",
            "message": f"Tool blocked by PBAC policy: {pbac_decision.reason}",
            "pbac": build_pbac_details(pbac_decision),
        }
        record_pbac_decision(
            config.database.path,
            pbac_decision,
            request_payload=sanitize_payload_for_logs(request_payload_for_logs(body)),
            response_payload=response_payload,
        )
        response = JSONResponse(status_code=403, content=response_payload)
        response.headers["x-pbac-decision"] = pbac_decision.decision
        response.headers["x-pbac-trigger"] = pbac_decision.trigger
        return response

    status_code, response_payload = build_tool_gateway_response(
        tool_name=tool_name,
        arguments=arguments,
        decision=pbac_decision,
        config=config,
    )
    record_pbac_decision(
        config.database.path,
        pbac_decision,
        request_payload=sanitize_payload_for_logs(request_payload_for_logs(body)),
        response_payload=response_payload,
    )
    response = JSONResponse(status_code=status_code, content=response_payload)
    response.headers["x-pbac-decision"] = pbac_decision.decision
    response.headers["x-pbac-trigger"] = pbac_decision.trigger
    return response


@app.post("/mock/v1/chat/completions", name="mock_chat_completions")
async def mock_chat_completions(request: Request) -> dict[str, Any]:
    refresh_runtime_agent_setup(app.state.config)
    body = await request.json()
    messages = body.get("messages", [])
    try:
        latest_user_message = normalize_user_message(
            extract_user_text_from_sources(body=body, content_sources=["body.messages", "body.input", "body.prompt"])
        )
    except ValueError:
        latest_user_message = "No user message supplied."

    model_name = body.get("model") or app.state.config.upstream.default_model
    response_text = (
        "Mock upstream reply because local mock mode is enabled. Turn it off in the dashboard "
        "Runtime Agent Setup panel and enter a real upstream base URL to receive your chatbot's "
        f"actual answer. Latest user message: {latest_user_message[:240]}"
    )
    prompt_tokens = len(json.dumps(messages))
    completion_tokens = len(response_text.split())

    return {
        "id": f"chatcmpl-mock-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.post("/mock/v1/responses", name="mock_responses")
async def mock_responses(request: Request) -> dict[str, Any]:
    refresh_runtime_agent_setup(app.state.config)
    body = await request.json()
    try:
        latest_user_message = normalize_user_message(
            extract_user_text_from_sources(body=body, content_sources=["body.input", "body.messages", "body.prompt"])
        )
    except ValueError:
        latest_user_message = "No user input supplied."

    model_name = body.get("model") or app.state.config.upstream.default_model
    response_text = (
        "Mock upstream reply because local mock mode is enabled. Turn it off in the dashboard "
        "Runtime Agent Setup panel and enter a real upstream base URL to receive your agent's "
        f"actual answer. Latest user message: {latest_user_message[:240]}"
    )
    output_tokens = len(response_text.split())

    return {
        "id": f"resp-mock-{uuid.uuid4().hex[:12]}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model_name,
        "output": [
            {
                "id": f"msg-mock-{uuid.uuid4().hex[:12]}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": response_text,
                        "annotations": [],
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": 0,
            "output_tokens": output_tokens,
            "total_tokens": output_tokens,
        },
    }


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def proxy_all(path: str, request: Request) -> Response:
    config: AppConfig = app.state.config
    refresh_runtime_agent_setup(config)
    route_config = match_protected_route(request, config)

    if route_config is not None:
        return await handle_protected_request(request, route_config)

    try:
        status_code, response_headers, upstream_body = await forward_upstream(request, config)
    except httpx.HTTPError as exc:
        log_event("upstream.error", path=request.url.path, error=str(exc))
        raise HTTPException(status_code=502, detail=f"Failed to reach upstream endpoint: {exc}") from exc

    return build_proxy_response(
        status_code,
        response_headers,
        upstream_body,
        firewall_decision="BYPASS",
        forwarded=True,
    )
