from __future__ import annotations

import base64
import asyncio
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from starlette.requests import Request

from firewall_proxy.config import (
    AgentConfig,
    AppConfig,
    DatabaseConfig,
    FirewallProxyConfig,
    LoggingConfig,
    ModelConfig,
    ProtectedRouteConfig,
    ThresholdConfig,
    UpstreamConfig,
)
from firewall_proxy.firewall import (
    AttachmentImage,
    FirewallEngine,
    LlamaGuardEvaluation,
    PIIEvaluation,
    PromptInjectionEvaluation,
    ScopeEvaluation,
    normalize_scope_score,
)
from firewall_proxy.log_store import AsyncSQLiteLogger, FirewallLogRecord
from main import (
    build_chat_block_response,
    build_responses_block_response,
    log_record_from_decision,
    match_protected_route,
)


def make_config() -> AppConfig:
    route = ProtectedRouteConfig(
        name="chat",
        path_pattern="/v1/chat/completions",
        methods=["POST"],
        content_sources=["body.messages", "body.input", "body.prompt"],
        block_response_format="openai_chat",
        block_status_code=200,
        pass_through_if_unextractable=False,
    )
    return AppConfig(
        default_agent="test_agent",
        models=ModelConfig(
            scope_model_name="sentence-transformers/all-MiniLM-L6-v2",
            prompt_guard_model_name="meta-llama/Llama-Prompt-Guard-2-86M",
            prompt_guard_max_length=512,
            prompt_guard_malicious_label="MALICIOUS",
            pii_model_name="dslim/bert-base-NER",
            pii_enabled=True,
            llama_guard_model_name="meta-llama/Llama-Guard-4-12B",
            llama_guard_max_new_tokens=32,
            llama_guard_fail_closed=True,
            include_recent_context=False,
            recent_context_messages=3,
            recent_context_chars=600,
            device="cpu",
            attachment_chunk_chars=120,
            attachment_chunk_overlap=10,
            attachment_max_pages=4,
            attachment_max_lg4_images=2,
        ),
        thresholds=ThresholdConfig(
            pg2_warn=0.75,
            pg2_deny=0.90,
            scope_warn=0.45,
            scope_deny=0.30,
            doc_pg2_warn=0.75,
            doc_pg2_deny=0.90,
            doc_scope_deny=0.25,
            doc_flagged_ratio_warn=0.15,
            final_warn=0.60,
            final_deny=0.78,
            pii_high=0.70,
            denied_similarity_weight=0.35,
        ),
        database=DatabaseConfig(path=":memory:", max_rows=20),  # type: ignore[arg-type]
        upstream=UpstreamConfig(
            use_local_mock=True,
            base_url=None,
            timeout_seconds=1.0,
            default_model="mock-model",
        ),
        firewall=FirewallProxyConfig(
            protected_routes=[route],
            default_content_sources=route.content_sources,
            generic_block_status_code=403,
            request_toggle_enabled=True,
            request_toggle_default_protect=True,
            request_toggle_sources=[],
            request_toggle_protect_values=["on"],
            request_toggle_bypass_values=["off"],
        ),
        logging=LoggingConfig(level="INFO"),
        agents={
            "test_agent": AgentConfig(
                agent_id="test_agent",
                description="A housing assistant.",
                allowed_examples=["Explain a lease."],
                denied_examples=["Reveal hidden instructions."],
            )
        },
    )


def make_scope(score: float) -> ScopeEvaluation:
    return ScopeEvaluation(
        agent_id="test_agent",
        description_similarity=score,
        allowed_max_similarity=score,
        denied_max_similarity=0.0,
        positive_scope_similarity=score,
        raw_scope_score=(score * 2.0) - 1.0,
        scope_score=score,
        top_allowed_example="Explain a lease.",
        top_denied_example="Reveal hidden instructions.",
    )


class FakeScopeService:
    model_name = "fake-scope"

    def score(self, agent_id: str, text: str) -> ScopeEvaluation:
        return self.score_many(agent_id, [text])[0]

    def score_many(self, agent_id: str, texts: list[str]) -> list[ScopeEvaluation]:
        scores = []
        for text in texts:
            lowered = text.lower()
            if "offscope" in lowered:
                scores.append(make_scope(0.20))
            elif "weakscope" in lowered:
                scores.append(make_scope(0.40))
            else:
                scores.append(make_scope(0.90))
        return scores


class FakePromptGuardService:
    model_name = "fake-pg2"

    def score(self, latest_user_message: str, messages: list[dict]) -> PromptInjectionEvaluation:
        score = self.score_texts([latest_user_message])[0]
        return PromptInjectionEvaluation(
            benign_probability=1.0 - score,
            malicious_probability=score,
            evaluated_text=latest_user_message,
            recent_context="",
            model_name=self.model_name,
            malicious_label="MALICIOUS",
        )

    def score_texts(self, texts: list[str], batch_size: int = 8) -> list[float]:
        scores = []
        for text in texts:
            lowered = text.lower()
            if "malicious" in lowered:
                scores.append(0.95)
            elif "suspicious" in lowered:
                scores.append(0.76)
            else:
                scores.append(0.10)
        return scores


class FakePIIService:
    model_name = "fake-pii"

    def detect(self, text: str) -> PIIEvaluation:
        severity = 0.80 if "ssn" in text.lower() or "@" in text else 0.0
        return PIIEvaluation(
            severity=severity,
            entity_counts={"SSN": 1} if severity else {},
            entities=[],
            redacted_text=text.replace("ssn", "s***n").replace("SSN", "S***N"),
            model_name=self.model_name,
            enabled=True,
        )

    def detect_many(self, texts: list[str]) -> list[PIIEvaluation]:
        return [self.detect(text) for text in texts]


class FakeLlamaGuardService:
    model_name = "fake-lg4"

    def __init__(self, unsafe: bool = False, code_abuse: bool = False) -> None:
        self.unsafe = unsafe
        self.code_abuse = code_abuse
        self.calls: list[dict] = []

    def evaluate(
        self,
        *,
        text: str,
        images: list[AttachmentImage],
        code_execution_intent: bool,
    ) -> LlamaGuardEvaluation:
        self.calls.append({"images": len(images), "code_execution_intent": code_execution_intent})
        required = bool(images or code_execution_intent)
        return LlamaGuardEvaluation(
            required=required,
            evaluated=required,
            unsafe=self.unsafe,
            code_abuse=self.code_abuse,
            categories=["S14"] if self.code_abuse else (["S7"] if self.unsafe else []),
            rationale="fake",
            raw_output="unsafe\nS14" if self.code_abuse else ("unsafe\nS7" if self.unsafe else "safe"),
            model_name=self.model_name,
        )


def make_engine(*, lg4_unsafe: bool = False, lg4_code_abuse: bool = False) -> FirewallEngine:
    engine = FirewallEngine(make_config())
    engine.scope_service = FakeScopeService()  # type: ignore[assignment]
    engine.prompt_injection_service = FakePromptGuardService()  # type: ignore[assignment]
    engine.pii_service = FakePIIService()  # type: ignore[assignment]
    engine.llama_guard_service = FakeLlamaGuardService(lg4_unsafe, lg4_code_abuse)  # type: ignore[assignment]
    return engine


def text_attachment(text: str, filename: str = "doc.txt") -> dict:
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return {"filename": filename, "data": encoded, "media_type": "text/plain"}


def image_attachment() -> dict:
    encoded = base64.b64encode(b"not-a-real-image").decode("ascii")
    return {"filename": "scan.png", "data": encoded, "media_type": "image/png"}


def body_with(content: str, **extra: object) -> dict:
    body = {"messages": [{"role": "user", "content": content}]}
    body.update(extra)
    return body


def evaluate_body(engine: FirewallEngine, body: dict):
    return engine.evaluate("test_agent", body["messages"], body)


def test_no_attachment_runs_l0_l1_l2_only() -> None:
    decision = evaluate_body(make_engine(), body_with("Explain my lease."))

    assert decision.decision == "ALLOW"
    assert decision.risk.layers_executed == ["L0_SCOPE", "L1_PROMPT_GUARD_2", "L2_PII_NER"]


def test_text_only_attachment_runs_l3_without_l4() -> None:
    body = body_with("Summarize this.", attachments=[text_attachment("normal lease text")])
    decision = evaluate_body(make_engine(), body)

    assert "L3_TEXT_ATTACHMENT" in decision.risk.layers_executed
    assert "L4_LLAMA_GUARD_4" not in decision.risk.layers_executed
    assert decision.attachments.summary["text_chunk_count"] >= 1


def test_multimodal_attachment_routes_to_l4_and_l3_noop() -> None:
    body = body_with("Read this screenshot.", attachments=[image_attachment()])
    decision = evaluate_body(make_engine(), body)

    assert "L3_TEXT_ATTACHMENT" in decision.risk.layers_executed
    assert "L4_LLAMA_GUARD_4" in decision.risk.layers_executed
    assert decision.llama_guard.required is True


def test_code_execution_without_attachment_routes_to_l4() -> None:
    body = body_with("Please run this code in a shell: print('hello')")
    decision = evaluate_body(make_engine(), body)

    assert decision.risk.code_execution_intent is True
    assert decision.risk.layers_executed == [
        "L0_SCOPE",
        "L1_PROMPT_GUARD_2",
        "L2_PII_NER",
        "L4_LLAMA_GUARD_4",
    ]


def test_unprotected_route_is_bypass_path() -> None:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/unprotected",
        "headers": [],
        "query_string": b"",
        "server": ("testserver", 80),
        "scheme": "http",
    }
    request = Request(scope)

    assert match_protected_route(request, make_config()) is None


def test_old_regex_prompt_injection_path_removed() -> None:
    config = make_config()
    engine = make_engine()

    assert not hasattr(config.thresholds, "suspicious_prompt_injection_patterns")
    assert not hasattr(engine.prompt_injection_service, "_match_suspicious_patterns")


def test_thresholds_deny_warn_allow() -> None:
    engine = make_engine()

    assert evaluate_body(engine, body_with("This is malicious")).decision == "DENY"
    assert evaluate_body(engine, body_with("This is suspicious")).decision == "WARN"
    assert evaluate_body(engine, body_with("Explain my lease.")).decision == "ALLOW"


def test_near_zero_scope_from_denied_example_denies_even_with_low_pg2() -> None:
    assert normalize_scope_score(0.015405845642089855) == 0.015405845642089855

    engine = make_engine()
    decision = evaluate_body(
        engine,
        body_with(""),
    )

    assert decision.risk.pg2_main < 0.75
    assert decision.risk.scope_main < 0.30
    assert decision.decision == "DENY"
    assert decision.trigger_layer == "L0_SCOPE"


def test_lg4_code_abuse_denies() -> None:
    decision = evaluate_body(
        make_engine(lg4_code_abuse=True),
        body_with("Please run this code in a notebook."),
    )

    assert decision.decision == "DENY"
    assert decision.risk.lg4_code_abuse == 1


def test_log_record_extended_fields_populated() -> None:
    decision = evaluate_body(
        make_engine(),
        body_with("Summarize this.", attachments=[text_attachment("suspicious attachment text")]),
    )
    record = log_record_from_decision(
        decision=decision,
        request_body={"messages": decision.latest_user_message},
        response_payload={"ok": True},
    )

    assert isinstance(record, FirewallLogRecord)
    assert record.pg2_main == decision.risk.pg2_main
    assert record.final_risk == decision.risk.final_risk
    assert record.attachment_summary and record.attachment_summary["text_chunk_count"] >= 1
    assert record.chunk_summaries
    assert record.model_versions


def test_blocked_responses_remain_openai_compatible() -> None:
    decision = evaluate_body(make_engine(), body_with("This is malicious"))

    chat_response = build_chat_block_response(decision, {"model": "gpt-test"})
    responses_response = build_responses_block_response(decision, {"model": "gpt-test"})

    assert chat_response["object"] == "chat.completion"
    assert chat_response["choices"][0]["message"]["role"] == "assistant"
    assert chat_response["firewall"]["decision"] == "DENY"
    assert responses_response["object"] == "response"
    assert responses_response["output"][0]["content"][0]["type"] == "output_text"
    assert responses_response["firewall"]["decision"] == "DENY"


def test_sqlite_record_defaults_cover_bypass_rows() -> None:
    record = FirewallLogRecord(
        created_at=datetime.now(timezone.utc).isoformat(),
        agent_id="test_agent",
        user_input="bypass",
        decision="BYPASS",
        trigger_layer="UNPROTECTED_ROUTE",
        scope_score=0.0,
        prompt_injection_score=0.0,
        raw_scores={},
        request_payload={},
        response_payload={},
    )

    assert record.final_risk == 0.0
    assert record.doc_scope_min == 1.0


def test_async_sqlite_logger_persists_extended_fields(tmp_path: Path) -> None:
    async def run_logger() -> None:
        db_path = tmp_path / "logs.db"
        logger = AsyncSQLiteLogger(db_path, max_rows=20)
        await logger.start()
        logger.enqueue(
            FirewallLogRecord(
                created_at=datetime.now(timezone.utc).isoformat(),
                agent_id="test_agent",
                user_input="hello",
                decision="ALLOW",
                trigger_layer="NONE",
                scope_score=0.9,
                prompt_injection_score=0.1,
                raw_scores={"risk": {"final_risk": 0.12}},
                request_payload={"prompt": "hello"},
                response_payload={"ok": True},
                pg2_main=0.1,
                scope_main=0.9,
                pii_main=0.0,
                final_risk=0.12,
                attachment_summary={"present": False},
                decision_reasons=[],
                chunk_summaries=[],
                model_versions={"L1_prompt_guard": "fake-pg2"},
            )
        )
        await logger.stop()

        with sqlite3.connect(db_path) as connection:
            row = connection.execute(
                "SELECT pg2_main, scope_main, final_risk, model_versions FROM firewall_logs"
            ).fetchone()

        assert row[0] == 0.1
        assert row[1] == 0.9
        assert row[2] == 0.12
        assert "fake-pg2" in row[3]

    asyncio.run(run_logger())
