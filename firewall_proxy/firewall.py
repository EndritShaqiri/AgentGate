from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from firewall_proxy.config import AgentConfig, AppConfig


def flatten_message_content(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    parts.append(stripped)
                continue

            if isinstance(item, dict):
                text_candidate = (
                    item.get("text")
                    or item.get("input_text")
                    or item.get("content")
                    or item.get("value")
                )
                if isinstance(text_candidate, str):
                    stripped = text_candidate.strip()
                    if stripped:
                        parts.append(stripped)

        return "\n".join(parts).strip()

    return str(content).strip()


def extract_latest_user_message(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = flatten_message_content(message.get("content"))
            if content:
                return content
    raise ValueError("No user message found in the conversation.")


def _extract_text_from_response_input_item(item: Any) -> str:
    if item is None:
        return ""

    if isinstance(item, str):
        return item.strip()

    if isinstance(item, list):
        parts = [_extract_text_from_response_input_item(value) for value in item]
        return "\n".join(part for part in parts if part).strip()

    if not isinstance(item, dict):
        return str(item).strip()

    if item.get("role") == "user":
        return flatten_message_content(item.get("content"))

    for key in ("text", "input_text", "prompt", "question", "query", "message", "user_request"):
        value = item.get(key)
        if value is None:
            continue
        extracted = _extract_text_from_response_input_item(value)
        if extracted:
            return extracted

    content = item.get("content")
    if content is not None:
        extracted = flatten_message_content(content)
        if extracted:
            return extracted
        extracted = _extract_text_from_response_input_item(content)
        if extracted:
            return extracted

    return ""


def extract_latest_user_input(input_value: Any) -> str:
    if input_value is None:
        raise ValueError("No user input found in the request.")

    if isinstance(input_value, str):
        text = input_value.strip()
        if text:
            return text
        raise ValueError("No user input found in the request.")

    if isinstance(input_value, list):
        for item in reversed(input_value):
            extracted = _extract_text_from_response_input_item(item)
            if extracted:
                return extracted
        raise ValueError("No user input found in the request.")

    extracted = _extract_text_from_response_input_item(input_value)
    if extracted:
        return extracted

    raise ValueError("No user input found in the request.")


def extract_text_candidate(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, dict):
        messages = value.get("messages")
        if isinstance(messages, list) and messages:
            try:
                return extract_latest_user_message(messages)
            except ValueError:
                pass

        if "input" in value:
            try:
                return extract_latest_user_input(value.get("input"))
            except ValueError:
                pass

        for key in ("prompt", "question", "query", "message", "text", "content", "user_request"):
            if key not in value:
                continue
            extracted = _extract_text_from_response_input_item(value.get(key))
            if extracted:
                return extracted

    return _extract_text_from_response_input_item(value)


def extract_user_message_from_request_body(body: dict[str, Any]) -> str:
    extracted = extract_text_candidate(body)
    if extracted:
        return extracted
    raise ValueError("No user message found in the request body.")


def _get_nested_value(root: Any, dotted_path: str) -> Any:
    if not dotted_path:
        return root

    current = root
    for segment in dotted_path.split("."):
        if isinstance(current, dict):
            if segment not in current:
                return None
            current = current.get(segment)
            continue

        if isinstance(current, list):
            try:
                index = int(segment)
            except ValueError:
                return None
            if index < 0 or index >= len(current):
                return None
            current = current[index]
            continue

        return None

    return current


def get_source_value(
    *,
    body: Any,
    query_params: dict[str, Any] | None = None,
    headers: dict[str, Any] | None = None,
    source: str,
) -> Any:
    query_params = query_params or {}
    normalized_headers = {str(key).lower(): value for key, value in (headers or {}).items()}

    if source == "body":
        return body
    if source.startswith("body."):
        return _get_nested_value(body, source[5:])
    if source.startswith("query."):
        return query_params.get(source[6:])
    if source.startswith("header."):
        return normalized_headers.get(source[7:].lower())
    return None


def extract_first_matching_value(
    *,
    body: Any,
    query_params: dict[str, Any] | None = None,
    headers: dict[str, Any] | None = None,
    sources: list[str] | None = None,
) -> Any:
    for source in sources or []:
        value = get_source_value(
            body=body,
            query_params=query_params,
            headers=headers,
            source=source,
        )
        if value not in (None, "", []):
            return value
    return None


def extract_user_text_from_sources(
    *,
    body: Any,
    query_params: dict[str, Any] | None = None,
    headers: dict[str, Any] | None = None,
    content_sources: list[str] | None = None,
) -> str:
    sources = content_sources or [
        "body.messages",
        "body.input",
        "body.prompt",
        "body.question",
        "body.query",
        "body.message",
        "body.text",
    ]
    query_params = query_params or {}

    for source in sources:
        source_value = get_source_value(
            body=body,
            query_params=query_params,
            headers=headers,
            source=source,
        )

        extracted = extract_text_candidate(source_value)
        if extracted:
            return extracted

    raise ValueError("No user text found in the configured content sources.")


def normalize_user_message(user_text: str) -> str:
    text = user_text.strip()
    if not text:
        return text

    request_match = re.search(r"(?is)\buser request:\s*(.+)$", text)
    if request_match:
        extracted = request_match.group(1).strip()
        if extracted:
            return extracted

    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        lowered = stripped.lower()
        if lowered.startswith("preferred role:"):
            continue
        if lowered.startswith("no uploaded pdf"):
            continue
        if lowered.startswith("uploaded pdf"):
            continue
        if lowered.startswith("user request:"):
            remainder = stripped.split(":", 1)[1].strip()
            if remainder:
                cleaned_lines.append(remainder)
            continue

        cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines).strip() or text


def find_latest_user_message_index(messages: list[dict[str, Any]]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            content = flatten_message_content(messages[index].get("content"))
            if content:
                return index
    raise ValueError("No user message found in the conversation.")


def build_compact_recent_context(
    messages: list[dict[str, Any]],
    max_messages: int,
    max_chars: int,
) -> str:
    recent: list[str] = []
    for message in reversed(messages):
        role = message.get("role", "unknown")
        content = flatten_message_content(message.get("content"))
        if not content:
            continue
        recent.append(f"{role}: {content}")
        if len(recent) >= max_messages:
            break

    recent.reverse()
    compact = "\n".join(recent)
    return compact[-max_chars:]


@dataclass(slots=True)
class ScopeEvaluation:
    agent_id: str
    description_similarity: float
    allowed_max_similarity: float
    denied_max_similarity: float
    positive_scope_similarity: float
    scope_score: float
    top_allowed_example: str
    top_denied_example: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PromptInjectionEvaluation:
    benign_probability: float
    malicious_probability: float
    evaluated_text: str
    recent_context: str
    suspicious_pattern_match: bool
    matched_patterns: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FirewallDecision:
    agent_id: str
    latest_user_message: str
    decision: str
    trigger_layer: str
    reason: str
    scope: ScopeEvaluation
    prompt_injection: PromptInjectionEvaluation

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentEmbeddingCache:
    agent_id: str
    description_embedding: np.ndarray
    allowed_embeddings: np.ndarray
    denied_embeddings: np.ndarray
    allowed_examples: list[str]
    denied_examples: list[str]


class ScopeService:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._model: SentenceTransformer | None = None
        self._cache: dict[str, AgentEmbeddingCache] = {}

    def initialize(self) -> None:
        model = SentenceTransformer(
            self._config.models.scope_model_name,
            device=self._resolve_device(),
        )

        cache: dict[str, AgentEmbeddingCache] = {}
        for agent_id, agent in self._config.agents.items():
            texts = [agent.description, *agent.allowed_examples, *agent.denied_examples]
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            description_embedding = embeddings[0]
            allowed_end = 1 + len(agent.allowed_examples)
            allowed_embeddings = embeddings[1:allowed_end]
            denied_embeddings = embeddings[allowed_end:]
            cache[agent_id] = AgentEmbeddingCache(
                agent_id=agent_id,
                description_embedding=description_embedding,
                allowed_embeddings=allowed_embeddings,
                denied_embeddings=denied_embeddings,
                allowed_examples=agent.allowed_examples,
                denied_examples=agent.denied_examples,
            )

        self._model = model
        self._cache = cache

    def _resolve_device(self) -> str:
        configured = self._config.models.device.lower()
        if configured == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return configured

    def score(self, agent_id: str, user_text: str) -> ScopeEvaluation:
        if self._model is None or agent_id not in self._cache:
            raise RuntimeError("Scope service has not been initialized.")

        user_embedding = self._model.encode(
            [user_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        cache = self._cache[agent_id]
        description_similarity = float(np.dot(cache.description_embedding, user_embedding))

        allowed_max_similarity, top_allowed_example = self._max_similarity(
            cache.allowed_embeddings,
            cache.allowed_examples,
            user_embedding,
        )
        denied_max_similarity, top_denied_example = self._max_similarity(
            cache.denied_embeddings,
            cache.denied_examples,
            user_embedding,
        )

        positive_scope_similarity = max(description_similarity, allowed_max_similarity)
        scope_score = positive_scope_similarity - (
            self._config.thresholds.denied_similarity_weight * denied_max_similarity
        )
        scope_score = float(max(-1.0, min(1.0, scope_score)))

        return ScopeEvaluation(
            agent_id=agent_id,
            description_similarity=description_similarity,
            allowed_max_similarity=allowed_max_similarity,
            denied_max_similarity=denied_max_similarity,
            positive_scope_similarity=positive_scope_similarity,
            scope_score=scope_score,
            top_allowed_example=top_allowed_example,
            top_denied_example=top_denied_example,
        )

    @staticmethod
    def _max_similarity(
        embeddings: np.ndarray,
        examples: list[str],
        user_embedding: np.ndarray,
    ) -> tuple[float, str]:
        if embeddings.size == 0 or not examples:
            return 0.0, ""

        similarities = embeddings @ user_embedding
        top_index = int(np.argmax(similarities))
        return float(similarities[top_index]), examples[top_index]


class PromptInjectionService:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: torch.device | None = None

    def initialize(self) -> None:
        resolved_device = self._resolve_device()
        tokenizer = AutoTokenizer.from_pretrained(
            self._config.models.prompt_injection_model_path,
            use_fast=False,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self._config.models.prompt_injection_model_path,
        )
        model.to(resolved_device)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        self._device = torch.device(resolved_device)

    def _resolve_device(self) -> str:
        configured = self._config.models.device.lower()
        if configured == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return configured

    def score(
        self,
        latest_user_message: str,
        messages: list[dict[str, Any]],
    ) -> PromptInjectionEvaluation:
        if self._tokenizer is None or self._model is None or self._device is None:
            raise RuntimeError("Prompt injection service has not been initialized.")

        recent_context = ""
        evaluated_text = latest_user_message
        if self._config.models.include_recent_context:
            latest_user_index = find_latest_user_message_index(messages)
            recent_context = build_compact_recent_context(
                messages[:latest_user_index],
                max_messages=self._config.models.recent_context_messages,
                max_chars=self._config.models.recent_context_chars,
            )
            if recent_context:
                evaluated_text = (
                    "RECENT CONTEXT\n"
                    f"{recent_context}\n\n"
                    "LATEST USER MESSAGE\n"
                    f"{latest_user_message}"
                )

        tokenized = self._tokenizer(
            evaluated_text,
            return_tensors="pt",
            truncation=True,
            max_length=self._config.models.classifier_max_length,
        )
        tokenized = {key: value.to(self._device) for key, value in tokenized.items()}

        with torch.no_grad():
            logits = self._model(**tokenized).logits
            probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        malicious_index = self._config.models.malicious_label_index
        malicious_probability = float(probabilities[malicious_index])
        benign_probability = (
            float(probabilities[1 - malicious_index])
            if len(probabilities) == 2
            else float(max(0.0, 1.0 - malicious_probability))
        )
        matched_patterns = self._match_suspicious_patterns(latest_user_message)

        return PromptInjectionEvaluation(
            benign_probability=benign_probability,
            malicious_probability=malicious_probability,
            evaluated_text=evaluated_text,
            recent_context=recent_context,
            suspicious_pattern_match=bool(matched_patterns),
            matched_patterns=matched_patterns,
        )

    def _match_suspicious_patterns(self, text: str) -> list[str]:
        lowered = text.lower()
        matches: list[str] = []
        for pattern in self._config.thresholds.suspicious_prompt_injection_patterns:
            candidate = pattern.strip().lower()
            if candidate and candidate in lowered:
                matches.append(pattern)
        return matches


class FirewallEngine:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self.scope_service = ScopeService(config)
        self.prompt_injection_service = PromptInjectionService(config)

    def validate_agent(self, agent_id: str) -> AgentConfig:
        agent = self._config.agents.get(agent_id)
        if agent is None:
            raise KeyError(
                f"Unknown agent_id '{agent_id}'. Available agents: {', '.join(self._config.agents)}"
            )
        return agent

    def evaluate(self, agent_id: str, messages: list[dict[str, Any]]) -> FirewallDecision:
        self.validate_agent(agent_id)
        latest_user_message = normalize_user_message(extract_latest_user_message(messages))
        scope_result = self.scope_service.score(agent_id, latest_user_message)
        injection_result = self.prompt_injection_service.score(latest_user_message, messages)
        prompt_injection_should_block = (
            injection_result.malicious_probability
            >= self._config.thresholds.prompt_injection_deny
        )
        if (
            prompt_injection_should_block
            and self._config.thresholds.require_suspicious_pattern_for_prompt_injection_deny
        ):
            prompt_injection_should_block = (
                injection_result.suspicious_pattern_match
                or scope_result.scope_score < self._config.thresholds.scope_warn
            )

        if prompt_injection_should_block:
            return FirewallDecision(
                agent_id=agent_id,
                latest_user_message=latest_user_message,
                decision="DENY",
                trigger_layer="LAYER_1_PROMPT_INJECTION",
                reason="Potential direct prompt injection detected in the latest user message.",
                scope=scope_result,
                prompt_injection=injection_result,
            )

        if scope_result.scope_score < self._config.thresholds.scope_deny:
            return FirewallDecision(
                agent_id=agent_id,
                latest_user_message=latest_user_message,
                decision="DENY",
                trigger_layer="LAYER_0_SCOPE_CHECK",
                reason="The latest user message appears out of scope for the registered agent.",
                scope=scope_result,
                prompt_injection=injection_result,
            )

        if scope_result.scope_score < self._config.thresholds.scope_warn:
            return FirewallDecision(
                agent_id=agent_id,
                latest_user_message=latest_user_message,
                decision="WARN",
                trigger_layer="LAYER_0_SCOPE_CHECK",
                reason="The latest user message is only weakly aligned with the registered agent.",
                scope=scope_result,
                prompt_injection=injection_result,
            )

        return FirewallDecision(
            agent_id=agent_id,
            latest_user_message=latest_user_message,
            decision="ALLOW",
            trigger_layer="NONE",
            reason="The request passed the firewall checks and was forwarded upstream.",
            scope=scope_result,
            prompt_injection=injection_result,
        )
