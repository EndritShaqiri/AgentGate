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
    extract_first_matching_value,
    extract_user_text_from_sources,
    normalize_user_message,
)
from firewall_proxy.log_store import AsyncSQLiteLogger, FirewallLogRecord
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
            "scope_score": decision.scope.scope_score,
            "prompt_injection_score": decision.prompt_injection.malicious_probability,
        },
        "details": {
            "scope": decision.scope.to_dict(),
            "prompt_injection": decision.prompt_injection.to_dict(),
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
) -> tuple[int, dict[str, str], bytes]:
    upstream_url = build_upstream_url(request, config)
    request_body = await request.body()
    request_headers = forwardable_headers(request)

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    configure_logging(config.logging.level)

    engine = FirewallEngine(config)
    logger = AsyncSQLiteLogger(config.database.path, config.database.max_rows)

    await logger.start()
    try:
        await asyncio.gather(
            asyncio.to_thread(engine.scope_service.initialize),
            asyncio.to_thread(engine.prompt_injection_service.initialize),
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
        database_path=str(config.database.path),
    )

    try:
        yield
    finally:
        await logger.stop()
        log_event("shutdown.complete")


app = FastAPI(title="AI Firewall Proxy", version="1.0.0", lifespan=lifespan)


@app.get("/")
async def root() -> dict[str, Any]:
    config: AppConfig = app.state.config
    return {
        "service": "ai_firewall_proxy",
        "status": "ok",
        "default_agent": config.default_agent,
        "protected_endpoints": configured_protected_endpoints(config),
        "request_toggle_enabled": config.firewall.request_toggle_enabled,
        "global_firewall_enabled": is_global_firewall_enabled(),
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    config: AppConfig = app.state.config
    return {
        "status": "ok",
        "default_agent": config.default_agent,
        "agents": list(config.agents),
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

    agent_id = resolve_agent_id(body, request, config)
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
                request_payload=request_payload_for_logs(body),
                response_payload=upstream_payload,
            )
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

            response = build_proxy_response(status_code, response_headers, upstream_body)
            response.headers["x-firewall-bypassed"] = "false"
            return response

        raise HTTPException(status_code=400, detail=str(exc)) from exc

    messages_for_scoring = body.get("messages") if isinstance(body, dict) else None
    if not isinstance(messages_for_scoring, list):
        messages_for_scoring = [{"role": "user", "content": latest_user_message}]

    try:
        decision = await asyncio.to_thread(engine.evaluate, agent_id, messages_for_scoring)
        decision.latest_user_message = latest_user_message
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    decision_payload = {
        "scope": decision.scope.to_dict(),
        "prompt_injection": decision.prompt_injection.to_dict(),
    }

    if decision.decision == "ALLOW":
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
                user_input=decision.latest_user_message,
                decision=decision.decision,
                trigger_layer=decision.trigger_layer,
                scope_score=decision.scope.scope_score,
                prompt_injection_score=decision.prompt_injection.malicious_probability,
                raw_scores=decision_payload,
                request_payload=request_payload_for_logs(body),
                response_payload=upstream_payload,
            )
        )
        log_event(
            "firewall.allow",
            agent_id=agent_id,
            route_name=route_config.name,
            path=request.url.path,
            scope_score=decision.scope.scope_score,
            prompt_injection_score=decision.prompt_injection.malicious_probability,
        )
        response = build_proxy_response(
            status_code,
            response_headers,
            upstream_body,
            firewall_decision=decision.decision,
            forwarded=True,
        )
        response.headers["x-firewall-bypassed"] = "false"
        return response

    firewall_response = build_firewall_sdk_response(route_config, body, decision)
    db_logger.enqueue(
        FirewallLogRecord(
            created_at=datetime.now(timezone.utc).isoformat(),
            agent_id=agent_id,
            user_input=decision.latest_user_message,
            decision=decision.decision,
            trigger_layer=decision.trigger_layer,
            scope_score=decision.scope.scope_score,
            prompt_injection_score=decision.prompt_injection.malicious_probability,
            raw_scores=decision_payload,
            request_payload=request_payload_for_logs(body),
            response_payload=firewall_response,
        )
    )
    log_event(
        "firewall.blocked",
        agent_id=agent_id,
        route_name=route_config.name,
        path=request.url.path,
        decision=decision.decision,
        trigger_layer=decision.trigger_layer,
        scope_score=decision.scope.scope_score,
        prompt_injection_score=decision.prompt_injection.malicious_probability,
    )
    response = JSONResponse(status_code=route_config.block_status_code, content=firewall_response)
    response.headers["x-firewall-decision"] = decision.decision
    response.headers["x-firewall-forwarded"] = "false"
    response.headers["x-firewall-bypassed"] = "false"
    return response


@app.post("/mock/v1/chat/completions", name="mock_chat_completions")
async def mock_chat_completions(request: Request) -> dict[str, Any]:
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
        "Mock upstream reply. Configure a real upstream URL to receive your chatbot's "
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
    body = await request.json()
    try:
        latest_user_message = normalize_user_message(
            extract_user_text_from_sources(body=body, content_sources=["body.input", "body.messages", "body.prompt"])
        )
    except ValueError:
        latest_user_message = "No user input supplied."

    model_name = body.get("model") or app.state.config.upstream.default_model
    response_text = (
        "Mock upstream reply. Configure a real upstream base URL to receive your agent's "
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
    route_config = match_protected_route(request, config)

    if route_config is not None:
        return await handle_protected_request(request, route_config)

    try:
        status_code, response_headers, upstream_body = await forward_upstream(request, config)
    except httpx.HTTPError as exc:
        log_event("upstream.error", path=request.url.path, error=str(exc))
        raise HTTPException(status_code=502, detail=f"Failed to reach upstream endpoint: {exc}") from exc

    return build_proxy_response(status_code, response_headers, upstream_body)
