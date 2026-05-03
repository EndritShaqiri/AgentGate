from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from firewall_proxy.attachments import (
    AttachmentExtractionResult,
    AttachmentImage,
    TextChunk,
    extract_attachments_from_body,
)
from firewall_proxy.config import AgentConfig, AppConfig


STRUCTURED_PII_PATTERNS = {
    "EMAIL": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "PHONE": re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\d)"),
}
CODE_EXECUTION_TERMS = (
    "code interpreter",
    "code_interpreter",
    "python sandbox",
    "sandbox",
    "notebook",
    "jupyter",
    "shell",
    "terminal",
    "powershell",
    "bash",
    "execute this",
    "execute code",
    "run this code",
    "run code",
    "subprocess",
    "os.system",
    "eval(",
    "exec(",
    "container escape",
)


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


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def normalize_scope_score(raw_score: float) -> float:
    # The scope formula already expresses "positive scope minus denied-example
    # pull" on a practical 0..1 similarity scale. Clamping preserves strong
    # denied-example matches instead of shifting near-zero risks up to ~0.5.
    return clamp01(raw_score)


def detect_code_execution_request(body: Any, text: str) -> bool:
    search_space = f"{text}\n{_compact_body_for_tool_scan(body)}".lower()
    if any(term in search_space for term in CODE_EXECUTION_TERMS):
        return True

    tools = body.get("tools") if isinstance(body, dict) else None
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            tool_type = str(tool.get("type") or "").lower()
            function_name = ""
            if isinstance(tool.get("function"), dict):
                function_name = str(tool["function"].get("name") or "").lower()
            if tool_type in {"code_interpreter", "computer", "shell", "python"}:
                return True
            if any(term in function_name for term in ("execute", "shell", "python", "sandbox", "notebook")):
                return True

    return False


def _compact_body_for_tool_scan(body: Any, max_chars: int = 5000) -> str:
    if body is None:
        return ""
    try:
        rendered = str(body)
    except Exception:
        return ""
    return rendered[:max_chars]


def mask_sensitive_value(value: str) -> str:
    stripped = value.strip()
    if len(stripped) <= 2:
        return "*" * len(stripped)
    if "@" in stripped:
        local, _, domain = stripped.partition("@")
        return f"{local[:1]}***@{domain}"
    return f"{stripped[:1]}***{stripped[-1:]}"


@dataclass(slots=True)
class ScopeEvaluation:
    agent_id: str
    description_similarity: float
    allowed_max_similarity: float
    denied_max_similarity: float
    positive_scope_similarity: float
    raw_scope_score: float
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
    model_name: str
    malicious_label: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PIIEntity:
    entity_type: str
    score: float
    start: int | None
    end: int | None
    masked_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PIIEvaluation:
    severity: float
    entity_counts: dict[str, int]
    entities: list[PIIEntity]
    redacted_text: str
    model_name: str
    enabled: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "entity_counts": self.entity_counts,
            "entities": [entity.to_dict() for entity in self.entities[:20]],
            "redacted_text": self.redacted_text,
            "model_name": self.model_name,
            "enabled": self.enabled,
            "error": self.error,
        }


@dataclass(slots=True)
class ChunkEvaluation:
    chunk: TextChunk
    prompt_guard_score: float
    scope: ScopeEvaluation
    pii: PIIEvaluation
    flagged: bool
    reasons: list[str]

    def to_log_dict(self) -> dict[str, Any]:
        payload = self.chunk.to_log_dict()
        payload.update(
            {
                "pg2_score": self.prompt_guard_score,
                "scope_score": self.scope.scope_score,
                "pii_severity": self.pii.severity,
                "flagged": self.flagged,
                "reasons": self.reasons,
                "redacted_excerpt": self.pii.redacted_text[:500],
            }
        )
        return payload


@dataclass(slots=True)
class AttachmentEvaluation:
    summary: dict[str, Any]
    doc_pg2_max: float
    doc_scope_min: float
    doc_scope_mean: float
    doc_flagged_chunks: int
    doc_flagged_ratio: float
    pii: PIIEvaluation
    chunks: list[ChunkEvaluation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "doc_pg2_max": self.doc_pg2_max,
            "doc_scope_min": self.doc_scope_min,
            "doc_scope_mean": self.doc_scope_mean,
            "doc_flagged_chunks": self.doc_flagged_chunks,
            "doc_flagged_ratio": self.doc_flagged_ratio,
            "pii": self.pii.to_dict(),
            "chunks": [chunk.to_log_dict() for chunk in self.chunks],
        }


@dataclass(slots=True)
class LlamaGuardEvaluation:
    required: bool
    evaluated: bool
    unsafe: bool
    code_abuse: bool
    categories: list[str]
    rationale: str
    raw_output: str
    model_name: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RiskEvaluation:
    pg2_main: float
    scope_main: float
    pii_main: float
    risk_main: float
    risk_doc: float | None
    risk_multimodal: float | None
    final_risk: float
    doc_pg2_max: float
    doc_scope_min: float
    doc_flagged_ratio: float
    lg4_unsafe: int
    lg4_code_abuse: int
    code_execution_intent: bool
    layers_executed: list[str]
    reasons: list[str]

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
    pii: PIIEvaluation
    attachments: AttachmentEvaluation
    llama_guard: LlamaGuardEvaluation
    risk: RiskEvaluation
    model_versions: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "latest_user_message": self.latest_user_message,
            "decision": self.decision,
            "trigger_layer": self.trigger_layer,
            "reason": self.reason,
            "scope": self.scope.to_dict(),
            "prompt_injection": self.prompt_injection.to_dict(),
            "pii": self.pii.to_dict(),
            "attachments": self.attachments.to_dict(),
            "llama_guard": self.llama_guard.to_dict(),
            "risk": self.risk.to_dict(),
            "model_versions": self.model_versions,
        }


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

    @property
    def model_name(self) -> str:
        return self._config.models.scope_model_name

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
        return self.score_many(agent_id, [user_text])[0]

    def score_many(self, agent_id: str, texts: list[str]) -> list[ScopeEvaluation]:
        if self._model is None or agent_id not in self._cache:
            raise RuntimeError("Scope service has not been initialized.")
        if not texts:
            return []

        user_embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        cache = self._cache[agent_id]
        return [self._build_evaluation(agent_id, cache, embedding) for embedding in user_embeddings]

    def _build_evaluation(
        self,
        agent_id: str,
        cache: AgentEmbeddingCache,
        user_embedding: np.ndarray,
    ) -> ScopeEvaluation:
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
        raw_scope_score = positive_scope_similarity - (
            self._config.thresholds.denied_similarity_weight * denied_max_similarity
        )
        raw_scope_score = float(max(-1.0, min(1.0, raw_scope_score)))

        return ScopeEvaluation(
            agent_id=agent_id,
            description_similarity=description_similarity,
            allowed_max_similarity=allowed_max_similarity,
            denied_max_similarity=denied_max_similarity,
            positive_scope_similarity=positive_scope_similarity,
            raw_scope_score=raw_scope_score,
            scope_score=normalize_scope_score(raw_scope_score),
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


class PromptGuardService:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: torch.device | None = None
        self._max_length = config.models.prompt_guard_max_length
        self._malicious_index = 1

    @property
    def model_name(self) -> str:
        return self._config.models.prompt_guard_model_name

    def initialize(self) -> None:
        resolved_device = self._resolve_device()
        tokenizer = AutoTokenizer.from_pretrained(
            self._config.models.prompt_guard_model_name,
            fix_mistral_regex=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self._config.models.prompt_guard_model_name,
        )
        model.to(resolved_device)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        self._device = torch.device(resolved_device)
        self._max_length = self._resolve_max_length(tokenizer)
        self._malicious_index = self._resolve_malicious_index(model)

    def _resolve_device(self) -> str:
        configured = self._config.models.device.lower()
        if configured == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return configured

    def _resolve_max_length(self, tokenizer: Any) -> int:
        tokenizer_limit = getattr(tokenizer, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 100_000:
            return tokenizer_limit
        return max(1, int(self._config.models.prompt_guard_max_length or 512))

    def _resolve_malicious_index(self, model: Any) -> int:
        desired = self._config.models.prompt_guard_malicious_label.lower()
        label2id = getattr(model.config, "label2id", {}) or {}
        for label, index in label2id.items():
            if str(label).lower() == desired:
                return int(index)

        id2label = getattr(model.config, "id2label", {}) or {}
        for index, label in id2label.items():
            if str(label).lower() == desired:
                return int(index)

        return 1 if getattr(model.config, "num_labels", 2) > 1 else 0

    def score(
        self,
        latest_user_message: str,
        messages: list[dict[str, Any]],
    ) -> PromptInjectionEvaluation:
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

        malicious_probability = self.score_texts([evaluated_text])[0]
        return PromptInjectionEvaluation(
            benign_probability=clamp01(1.0 - malicious_probability),
            malicious_probability=malicious_probability,
            evaluated_text=evaluated_text,
            recent_context=recent_context,
            model_name=self.model_name,
            malicious_label=self._config.models.prompt_guard_malicious_label,
        )

    def score_texts(self, texts: list[str], batch_size: int = 8) -> list[float]:
        if self._tokenizer is None or self._model is None or self._device is None:
            raise RuntimeError("Prompt Guard service has not been initialized.")
        if not texts:
            return []

        scores: list[float] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            tokenized = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._max_length,
            )
            tokenized = {key: value.to(self._device) for key, value in tokenized.items()}

            with torch.no_grad():
                logits = self._model(**tokenized).logits
                probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()

            for probability_row in probabilities:
                index = min(self._malicious_index, len(probability_row) - 1)
                scores.append(clamp01(float(probability_row[index])))

        return scores


class PIIService:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._pipeline: Any = None
        self._enabled = config.models.pii_enabled
        self._error: str | None = None

    @property
    def model_name(self) -> str:
        return self._config.models.pii_model_name

    def initialize(self) -> None:
        if not self._enabled:
            return
        try:
            from transformers import pipeline

            device = 0 if self._resolve_device() == "cuda" else -1
            self._pipeline = pipeline(
                "token-classification",
                model=self._config.models.pii_model_name,
                aggregation_strategy="simple",
                device=device,
            )
        except Exception as exc:
            self._error = str(exc)
            self._pipeline = None

    def _resolve_device(self) -> str:
        configured = self._config.models.device.lower()
        if configured == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return configured

    def detect_many(self, texts: list[str]) -> list[PIIEvaluation]:
        return [self.detect(text) for text in texts]

    def detect(self, text: str) -> PIIEvaluation:
        if not text:
            return self._empty(text)

        entities: list[PIIEntity] = []
        if self._pipeline is not None:
            try:
                for item in self._pipeline(text):
                    entity_type = str(item.get("entity_group") or item.get("entity") or "PII")
                    score = float(item.get("score") or 0.0)
                    start = item.get("start")
                    end = item.get("end")
                    if item.get("word"):
                        word = str(item["word"])
                    elif isinstance(start, int) and isinstance(end, int):
                        word = text[start:end]
                    else:
                        word = ""
                    if score >= 0.55:
                        entities.append(
                            PIIEntity(
                                entity_type=entity_type,
                                score=score,
                                start=int(start) if isinstance(start, int) else None,
                                end=int(end) if isinstance(end, int) else None,
                                masked_text=mask_sensitive_value(word),
                            )
                        )
            except Exception as exc:
                self._error = str(exc)

        entities.extend(self._detect_structured_pii(text))
        redacted_text = redact_text(text, entities)
        counts: dict[str, int] = {}
        for entity in entities:
            counts[entity.entity_type] = counts.get(entity.entity_type, 0) + 1

        return PIIEvaluation(
            severity=self._score_severity(counts),
            entity_counts=counts,
            entities=entities,
            redacted_text=redacted_text,
            model_name=self.model_name,
            enabled=self._enabled,
            error=self._error,
        )

    def _empty(self, text: str) -> PIIEvaluation:
        return PIIEvaluation(
            severity=0.0,
            entity_counts={},
            entities=[],
            redacted_text=text,
            model_name=self.model_name,
            enabled=self._enabled,
            error=self._error,
        )

    @staticmethod
    def _detect_structured_pii(text: str) -> list[PIIEntity]:
        entities: list[PIIEntity] = []
        for entity_type, pattern in STRUCTURED_PII_PATTERNS.items():
            for match in pattern.finditer(text):
                entities.append(
                    PIIEntity(
                        entity_type=entity_type,
                        score=1.0,
                        start=match.start(),
                        end=match.end(),
                        masked_text=mask_sensitive_value(match.group(0)),
                    )
                )
        return entities

    @staticmethod
    def _score_severity(counts: dict[str, int]) -> float:
        weights = {
            "SSN": 0.85,
            "EMAIL": 0.35,
            "PHONE": 0.35,
            "PER": 0.25,
            "PERSON": 0.25,
            "LOC": 0.18,
            "LOCATION": 0.18,
            "ORG": 0.12,
            "MISC": 0.08,
        }
        score = 0.0
        for entity_type, count in counts.items():
            score += weights.get(entity_type.upper(), 0.10) * count
        return clamp01(score)


def redact_text(text: str, entities: list[PIIEntity]) -> str:
    spans = [
        (entity.start, entity.end, entity.masked_text)
        for entity in entities
        if entity.start is not None and entity.end is not None and entity.start < entity.end
    ]
    if not spans:
        return text

    redacted = text
    for start, end, replacement in sorted(spans, reverse=True):
        redacted = redacted[:start] + replacement + redacted[end:]
    return redacted


class LlamaGuardService:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._processor: Any = None
        self._model: Any = None
        self._device: str = "cpu"
        self._lock = threading.Lock()
        self._load_error: str | None = None

    @property
    def model_name(self) -> str:
        return self._config.models.llama_guard_model_name

    def evaluate(
        self,
        *,
        text: str,
        images: list[AttachmentImage],
        code_execution_intent: bool,
    ) -> LlamaGuardEvaluation:
        required = bool(images or code_execution_intent)
        if not required:
            return LlamaGuardEvaluation(
                required=False,
                evaluated=False,
                unsafe=False,
                code_abuse=False,
                categories=[],
                rationale="L4 was not required for this request.",
                raw_output="",
                model_name=self.model_name,
            )

        try:
            self._ensure_initialized()
            raw_output = self._generate(text=text, images=images, code_execution_intent=code_execution_intent)
            unsafe, code_abuse, categories, rationale = self._parse_output(raw_output)
            return LlamaGuardEvaluation(
                required=True,
                evaluated=True,
                unsafe=unsafe,
                code_abuse=code_abuse,
                categories=categories,
                rationale=rationale,
                raw_output=raw_output,
                model_name=self.model_name,
            )
        except Exception as exc:
            error = str(exc)
            fail_closed = self._config.models.llama_guard_fail_closed
            return LlamaGuardEvaluation(
                required=True,
                evaluated=False,
                unsafe=fail_closed,
                code_abuse=False,
                categories=["LG4_UNAVAILABLE"] if fail_closed else [],
                rationale="Llama Guard 4 could not be evaluated locally.",
                raw_output="",
                model_name=self.model_name,
                error=error,
            )

    def _ensure_initialized(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        with self._lock:
            if self._processor is not None and self._model is not None:
                return
            if self._load_error:
                raise RuntimeError(self._load_error)
            try:
                from transformers import AutoProcessor

                model_cls = self._resolve_model_class()
                self._processor = AutoProcessor.from_pretrained(self._config.models.llama_guard_model_name)
                self._device = self._resolve_device()
                model_kwargs: dict[str, Any] = {}
                if self._device == "cuda":
                    model_kwargs["device_map"] = "auto"
                    model_kwargs["torch_dtype"] = torch.bfloat16
                model = model_cls.from_pretrained(self._config.models.llama_guard_model_name, **model_kwargs)
                if "device_map" not in model_kwargs:
                    model.to(self._device)
                model.eval()
                self._model = model
            except Exception as exc:
                self._load_error = str(exc)
                raise

    @staticmethod
    def _resolve_model_class() -> Any:
        try:
            from transformers import Llama4ForConditionalGeneration

            return Llama4ForConditionalGeneration
        except Exception:
            pass
        try:
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText
        except Exception:
            from transformers import AutoModelForCausalLM

            return AutoModelForCausalLM

    def _resolve_device(self) -> str:
        configured = self._config.models.device.lower()
        if configured == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return configured

    def _generate(self, *, text: str, images: list[AttachmentImage], code_execution_intent: bool) -> str:
        if self._processor is None or self._model is None:
            raise RuntimeError("Llama Guard 4 service has not been initialized.")

        content: list[dict[str, Any]] = []
        for image in images:
            pil_image = _attachment_image_to_pil(image)
            if pil_image is not None:
                content.append({"type": "image", "image": pil_image})

        prompt = self._build_guard_prompt(text=text, code_execution_intent=code_execution_intent)
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = inputs.to(self._device if self._device != "auto" else "cuda")

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._config.models.llama_guard_max_new_tokens,
                do_sample=False,
            )
        new_tokens = outputs[:, inputs["input_ids"].shape[-1] :]
        return self._processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

    @staticmethod
    def _build_guard_prompt(*, text: str, code_execution_intent: bool) -> str:
        prefix = (
            "Classify this user request and any provided images for safety. "
            "Return safe or unsafe, and include violated categories when unsafe."
        )
        if code_execution_intent:
            prefix += " Pay special attention to S14 Code Interpreter Abuse and tool misuse."
        return f"{prefix}\n\nRequest/context:\n{text[:6000]}"

    @staticmethod
    def _parse_output(raw_output: str) -> tuple[bool, bool, list[str], str]:
        lowered = raw_output.strip().lower()
        unsafe = lowered.startswith("unsafe") or "\nunsafe" in lowered
        categories = sorted(set(re.findall(r"\bS\d{1,2}\b", raw_output.upper())))
        code_abuse = "S14" in categories or "code interpreter abuse" in lowered
        rationale = raw_output.strip()
        return unsafe, code_abuse, categories, rationale


def _attachment_image_to_pil(image: AttachmentImage) -> Any | None:
    try:
        import io
        from PIL import Image

        return Image.open(io.BytesIO(image.data)).convert("RGB")
    except Exception:
        return None


class FirewallEngine:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self.scope_service = ScopeService(config)
        self.prompt_injection_service = PromptGuardService(config)
        self.pii_service = PIIService(config)
        self.llama_guard_service = LlamaGuardService(config)

    def validate_agent(self, agent_id: str) -> AgentConfig:
        agent = self._config.agents.get(agent_id)
        if agent is None:
            raise KeyError(
                f"Unknown agent_id '{agent_id}'. Available agents: {', '.join(self._config.agents)}"
            )
        return agent

    def initialize_core_services(self) -> None:
        self.scope_service.initialize()
        self.prompt_injection_service.initialize()
        self.pii_service.initialize()

    def evaluate(
        self,
        agent_id: str,
        messages: list[dict[str, Any]],
        request_body: Any | None = None,
    ) -> FirewallDecision:
        self.validate_agent(agent_id)
        latest_user_message = normalize_user_message(extract_latest_user_message(messages))
        request_body = request_body or {}
        attachment_result = extract_attachments_from_body(
            request_body,
            chunk_chars=self._config.models.attachment_chunk_chars,
            chunk_overlap=self._config.models.attachment_chunk_overlap,
            max_pages=self._config.models.attachment_max_pages,
            max_lg4_images=self._config.models.attachment_max_lg4_images,
        )
        code_execution_intent = detect_code_execution_request(request_body, latest_user_message)

        scope_result, injection_result, pii_result = self._evaluate_main_signals(
            agent_id,
            latest_user_message,
            messages,
        )
        attachment_evaluation = self._evaluate_attachments(agent_id, attachment_result)
        llama_guard = self.llama_guard_service.evaluate(
            text=self._build_lg4_context(latest_user_message, attachment_result),
            images=attachment_result.images_for_lg4[: self._config.models.attachment_max_lg4_images],
            code_execution_intent=code_execution_intent,
        )

        risk = self._fuse_decision(
            pg2_main=injection_result.malicious_probability,
            scope_main=scope_result.scope_score,
            pii_main=pii_result.severity,
            attachments=attachment_evaluation,
            llama_guard=llama_guard,
            code_execution_intent=code_execution_intent,
            attachment_result=attachment_result,
        )
        decision, trigger_layer, reason = self._select_decision(risk, pii_result, attachment_evaluation)

        return FirewallDecision(
            agent_id=agent_id,
            latest_user_message=pii_result.redacted_text,
            decision=decision,
            trigger_layer=trigger_layer,
            reason=reason,
            scope=scope_result,
            prompt_injection=injection_result,
            pii=pii_result,
            attachments=attachment_evaluation,
            llama_guard=llama_guard,
            risk=risk,
            model_versions=self._model_versions(),
        )

    def _evaluate_main_signals(
        self,
        agent_id: str,
        latest_user_message: str,
        messages: list[dict[str, Any]],
    ) -> tuple[ScopeEvaluation, PromptInjectionEvaluation, PIIEvaluation]:
        with ThreadPoolExecutor(max_workers=3) as executor:
            scope_future = executor.submit(self.scope_service.score, agent_id, latest_user_message)
            prompt_guard_future = executor.submit(
                self.prompt_injection_service.score,
                latest_user_message,
                messages,
            )
            pii_future = executor.submit(self.pii_service.detect, latest_user_message)
            return scope_future.result(), prompt_guard_future.result(), pii_future.result()

    def _evaluate_attachments(
        self,
        agent_id: str,
        attachment_result: AttachmentExtractionResult,
    ) -> AttachmentEvaluation:
        if not attachment_result.chunks:
            return AttachmentEvaluation(
                summary=attachment_result.summary_dict(),
                doc_pg2_max=0.0,
                doc_scope_min=1.0,
                doc_scope_mean=1.0,
                doc_flagged_chunks=0,
                doc_flagged_ratio=0.0,
                pii=self.pii_service.detect(""),
                chunks=[],
            )

        chunk_texts = [chunk.text for chunk in attachment_result.chunks]
        with ThreadPoolExecutor(max_workers=3) as executor:
            pg_future = executor.submit(self.prompt_injection_service.score_texts, chunk_texts)
            scope_future = executor.submit(self.scope_service.score_many, agent_id, chunk_texts)
            pii_future = executor.submit(self.pii_service.detect_many, chunk_texts)

            pg_scores = pg_future.result()
            scope_scores = scope_future.result()
            pii_scores = pii_future.result()

        chunk_evaluations: list[ChunkEvaluation] = []
        for chunk, pg_score, scope_score, pii_score in zip(
            attachment_result.chunks,
            pg_scores,
            scope_scores,
            pii_scores,
        ):
            reasons: list[str] = []
            if pg_score >= self._config.thresholds.doc_pg2_warn:
                reasons.append("Prompt Guard 2 flagged this attachment chunk.")
            if scope_score.scope_score < self._config.thresholds.scope_warn:
                reasons.append("Attachment chunk is weakly aligned with the agent scope.")
            if pii_score.severity >= self._config.thresholds.pii_high:
                reasons.append("Attachment chunk contains high-severity PII.")
            flagged = any(
                reason
                for reason in reasons
                if not reason.startswith("Attachment chunk contains high-severity PII")
            )
            chunk_evaluations.append(
                ChunkEvaluation(
                    chunk=chunk,
                    prompt_guard_score=pg_score,
                    scope=scope_score,
                    pii=pii_score,
                    flagged=flagged,
                    reasons=reasons,
                )
            )

        doc_pg2_max = max((chunk.prompt_guard_score for chunk in chunk_evaluations), default=0.0)
        scope_values = [chunk.scope.scope_score for chunk in chunk_evaluations]
        doc_scope_min = min(scope_values) if scope_values else 1.0
        doc_scope_mean = float(sum(scope_values) / len(scope_values)) if scope_values else 1.0
        flagged_chunks = sum(1 for chunk in chunk_evaluations if chunk.flagged)
        flagged_ratio = flagged_chunks / len(chunk_evaluations) if chunk_evaluations else 0.0
        attachment_pii = aggregate_pii_evaluations(pii_scores, model_name=self.pii_service.model_name)

        return AttachmentEvaluation(
            summary=attachment_result.summary_dict(),
            doc_pg2_max=doc_pg2_max,
            doc_scope_min=doc_scope_min,
            doc_scope_mean=doc_scope_mean,
            doc_flagged_chunks=flagged_chunks,
            doc_flagged_ratio=flagged_ratio,
            pii=attachment_pii,
            chunks=chunk_evaluations,
        )

    @staticmethod
    def _build_lg4_context(latest_user_message: str, attachment_result: AttachmentExtractionResult) -> str:
        nearby_text: list[str] = []
        for chunk in attachment_result.chunks[:4]:
            nearby_text.append(
                f"Attachment {chunk.attachment_name}, page {chunk.page_number}: {chunk.text[:1200]}"
            )
        if nearby_text:
            return latest_user_message + "\n\nNearby extracted attachment text:\n" + "\n\n".join(nearby_text)
        return latest_user_message

    def _fuse_decision(
        self,
        *,
        pg2_main: float,
        scope_main: float,
        pii_main: float,
        attachments: AttachmentEvaluation,
        llama_guard: LlamaGuardEvaluation,
        code_execution_intent: bool,
        attachment_result: AttachmentExtractionResult,
    ) -> RiskEvaluation:
        scope_risk_main = 1.0 - scope_main
        risk_main = clamp01(0.60 * pg2_main + 0.25 * scope_risk_main + 0.15 * pii_main)

        has_text_attachment = attachment_result.has_text_attachments
        risk_doc: float | None = None
        if has_text_attachment:
            scope_risk_doc = 1.0 - attachments.doc_scope_min
            attachment_pii = attachments.pii.severity if attachments.pii.enabled else pii_main
            risk_doc = clamp01(
                0.50 * attachments.doc_pg2_max
                + 0.25 * scope_risk_doc
                + 0.15 * attachments.doc_flagged_ratio
                + 0.10 * attachment_pii
            )

        needs_l4 = attachment_result.has_multimodal_attachments or code_execution_intent
        risk_multimodal: float | None = None
        if needs_l4:
            risk_multimodal = max(risk_main, risk_doc if risk_doc is not None else 0.0)
            if llama_guard.unsafe:
                risk_multimodal = max(risk_multimodal, 0.92)
            if llama_guard.code_abuse:
                risk_multimodal = max(risk_multimodal, 0.95)

        if needs_l4:
            final_risk = risk_multimodal or risk_main
        elif has_text_attachment:
            final_risk = max(risk_main, risk_doc if risk_doc is not None else 0.0)
        else:
            final_risk = risk_main

        layers = ["L0_SCOPE", "L1_PROMPT_GUARD_2", "L2_PII_NER"]
        if attachment_result.has_attachments:
            layers.append("L3_TEXT_ATTACHMENT")
        if needs_l4:
            layers.append("L4_LLAMA_GUARD_4")

        return RiskEvaluation(
            pg2_main=pg2_main,
            scope_main=scope_main,
            pii_main=pii_main,
            risk_main=risk_main,
            risk_doc=risk_doc,
            risk_multimodal=risk_multimodal,
            final_risk=clamp01(final_risk),
            doc_pg2_max=attachments.doc_pg2_max,
            doc_scope_min=attachments.doc_scope_min,
            doc_flagged_ratio=attachments.doc_flagged_ratio,
            lg4_unsafe=1 if llama_guard.unsafe else 0,
            lg4_code_abuse=1 if llama_guard.code_abuse else 0,
            code_execution_intent=code_execution_intent,
            layers_executed=layers,
            reasons=[],
        )

    def _select_decision(
        self,
        risk: RiskEvaluation,
        main_pii: PIIEvaluation,
        attachments: AttachmentEvaluation,
    ) -> tuple[str, str, str]:
        reasons: list[str] = []
        deny_layer = "RISK_FUSION"

        if risk.pg2_main >= self._config.thresholds.pg2_deny:
            reasons.append("Prompt Guard 2 marked the main user text as high-risk.")
            deny_layer = "L1_PROMPT_GUARD_2"
        if risk.scope_main < self._config.thresholds.scope_deny:
            reasons.append("Main user text is out of scope for the selected agent.")
            deny_layer = "L0_SCOPE"
        if attachments.summary.get("text_attachment_count", 0) and risk.doc_pg2_max >= self._config.thresholds.doc_pg2_deny:
            reasons.append("Prompt Guard 2 marked an attachment chunk as high-risk.")
            deny_layer = "L3_TEXT_ATTACHMENT"
        if attachments.summary.get("text_attachment_count", 0) and risk.doc_scope_min < self._config.thresholds.doc_scope_deny:
            reasons.append("At least one attachment chunk is out of scope for the selected agent.")
            deny_layer = "L3_TEXT_ATTACHMENT"
        if risk.lg4_unsafe:
            reasons.append("Llama Guard 4 marked multimodal or tool-use content as unsafe.")
            deny_layer = "L4_LLAMA_GUARD_4"
        if risk.lg4_code_abuse:
            reasons.append("Llama Guard 4 detected code-interpreter abuse or tool misuse.")
            deny_layer = "L4_LLAMA_GUARD_4"
        if risk.final_risk >= self._config.thresholds.final_deny:
            reasons.append("Final fused risk exceeded the deny threshold.")

        decision = "DENY" if reasons else "ALLOW"
        trigger_layer = deny_layer if reasons else "NONE"

        warn_reasons: list[str] = []
        warn_layer = "RISK_FUSION"
        if decision != "DENY":
            if risk.pg2_main >= self._config.thresholds.pg2_warn:
                warn_reasons.append("Prompt Guard 2 marked the main user text as suspicious.")
                warn_layer = "L1_PROMPT_GUARD_2"
            if risk.scope_main < self._config.thresholds.scope_warn:
                warn_reasons.append("Main user text is weakly aligned with the selected agent.")
                warn_layer = "L0_SCOPE"
            if attachments.summary.get("text_attachment_count", 0) and risk.doc_pg2_max >= self._config.thresholds.doc_pg2_warn:
                warn_reasons.append("Prompt Guard 2 marked an attachment chunk as suspicious.")
                warn_layer = "L3_TEXT_ATTACHMENT"
            if attachments.summary.get("text_attachment_count", 0) and risk.doc_flagged_ratio >= self._config.thresholds.doc_flagged_ratio_warn:
                warn_reasons.append("A meaningful share of attachment chunks were flagged.")
                warn_layer = "L3_TEXT_ATTACHMENT"
            if self._config.thresholds.final_warn <= risk.final_risk < self._config.thresholds.final_deny:
                warn_reasons.append("Final fused risk exceeded the warning threshold.")

            if warn_reasons:
                decision = "WARN"
                trigger_layer = warn_layer
                reasons = warn_reasons

        decision, trigger_layer, reasons = self._apply_pii_escalation(
            decision,
            trigger_layer,
            reasons,
            risk,
            main_pii,
            attachments,
        )
        risk.reasons.extend(reasons)
        reason = " ".join(reasons) if reasons else "The request passed the firewall checks and was forwarded upstream."
        return decision, trigger_layer, reason

    def _apply_pii_escalation(
        self,
        decision: str,
        trigger_layer: str,
        reasons: list[str],
        risk: RiskEvaluation,
        main_pii: PIIEvaluation,
        attachments: AttachmentEvaluation,
    ) -> tuple[str, str, list[str]]:
        high_pii = max(main_pii.severity, attachments.pii.severity) >= self._config.thresholds.pii_high
        guard_suspicious = (
            risk.pg2_main >= self._config.thresholds.pg2_warn
            or risk.doc_pg2_max >= self._config.thresholds.doc_pg2_warn
            or bool(risk.lg4_unsafe)
            or bool(risk.lg4_code_abuse)
        )
        if not (high_pii and guard_suspicious):
            return decision, trigger_layer, reasons

        escalation_reason = "High-severity PII appeared alongside a suspicious guardrail signal."
        if decision == "ALLOW":
            return "WARN", "L2_PII_NER", [*reasons, escalation_reason]
        if decision == "WARN":
            return "DENY", "L2_PII_NER", [*reasons, escalation_reason]
        return decision, trigger_layer, [*reasons, escalation_reason]

    def _model_versions(self) -> dict[str, str]:
        return {
            "L0_scope": self.scope_service.model_name,
            "L1_prompt_guard": self.prompt_injection_service.model_name,
            "L2_pii_ner": self.pii_service.model_name,
            "L4_llama_guard": self.llama_guard_service.model_name,
        }


def aggregate_pii_evaluations(evaluations: list[PIIEvaluation], *, model_name: str) -> PIIEvaluation:
    if not evaluations:
        return PIIEvaluation(
            severity=0.0,
            entity_counts={},
            entities=[],
            redacted_text="",
            model_name=model_name,
            enabled=True,
        )

    counts: dict[str, int] = {}
    entities: list[PIIEntity] = []
    redacted_samples: list[str] = []
    enabled = any(evaluation.enabled for evaluation in evaluations)
    errors = [evaluation.error for evaluation in evaluations if evaluation.error]
    for evaluation in evaluations:
        for entity_type, count in evaluation.entity_counts.items():
            counts[entity_type] = counts.get(entity_type, 0) + count
        entities.extend(evaluation.entities)
        if evaluation.redacted_text:
            redacted_samples.append(evaluation.redacted_text[:300])

    severity = max((evaluation.severity for evaluation in evaluations), default=0.0)
    return PIIEvaluation(
        severity=severity,
        entity_counts=counts,
        entities=entities[:20],
        redacted_text="\n---\n".join(redacted_samples[:5]),
        model_name=model_name,
        enabled=enabled,
        error="; ".join(errors) if errors else None,
    )
