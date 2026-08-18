from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.routing.policies import RoutingPreference


class AegisSchema(BaseModel):
	model_config = ConfigDict(extra="forbid", populate_by_name=True)


class MessageRole(StrEnum):
	SYSTEM = "system"
	USER = "user"
	ASSISTANT = "assistant"
	TOOL = "tool"


class AgentType(StrEnum):
	RESEARCH = "research"
	CODER = "coder"
	ANALYST = "analyst"
	DOCUMENT = "document"
	PLANNER = "planner"


class ChatMessage(AegisSchema):
	role: MessageRole
	content: str = Field(min_length=1)


class ChatRequest(AegisSchema):
	messages: list[ChatMessage] = Field(min_length=1)
	preference: RoutingPreference = "balanced"
	preferred_provider: str | None = None
	agent_hint: AgentType | None = None
	model_hint: str | None = None
	evaluate_with_llm: bool = False
	conversation_id: str | None = None
	orchestrate: bool = False


class AgentRoutingInput(AegisSchema):
	task: str = Field(min_length=1)
	context: str | None = None
	preferred_agent: AgentType | None = None
	preference: RoutingPreference = "balanced"
	preferred_provider: str | None = None


class AgentRoutingDecision(AegisSchema):
	agent: AgentType
	confidence: float = Field(ge=0.0, le=1.0)
	reason: str
	metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderCallRequest(AegisSchema):
	prompt: str = Field(min_length=1)
	model: str = Field(min_length=1)
	system_prompt: str | None = None
	provider: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderCallResult(AegisSchema):
	content: str
	model: str
	provider: str
	input_tokens: int | None = None
	output_tokens: int | None = None
	latency_ms: float | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


class ObservabilityEventType(StrEnum):
	REQUEST_RECEIVED = "request.received"
	AGENT_SELECTED = "agent.selected"
	MODEL_SELECTED = "model.selected"
	PROVIDER_CALLED = "provider.called"
	REQUEST_COMPLETED = "request.completed"
	REQUEST_FAILED = "request.failed"


class ObservabilityEvent(AegisSchema):
	event_type: ObservabilityEventType
	timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	request_id: str | None = None
	agent: AgentType | None = None
	model: str | None = None
	provider: str | None = None
	latency_ms: float | None = None
	input_tokens: int | None = None
	output_tokens: int | None = None
	error: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationRecord(AegisSchema):
	request_id: str
	agent: AgentType
	model: str
	provider: str
	score: float = Field(ge=0.0, le=1.0)
	metric: str
	metadata: dict[str, Any] = Field(default_factory=dict)


class RoutingResult(AegisSchema):
	agent: AgentType
	model: str
	provider: str
	preference: RoutingPreference = "balanced"
	confidence: float = Field(ge=0.0, le=1.0)
	reason: str
	metadata: dict[str, Any] = Field(default_factory=dict)


class RiskLevel(StrEnum):
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"


class MissionBrief(AegisSchema):
	summary: str
	risk_score: int = Field(ge=0, le=100)
	risk_level: RiskLevel
	top_alerts: list[str] = Field(default_factory=list)
	recommended_actions: list[str] = Field(default_factory=list)
	evidence: list[str] = Field(default_factory=list)


class ResponseEvaluation(AegisSchema):
	"""Deterministic, always-free quality signal computed for every response.

	groundedness is None when no RAG context was available to ground against (nothing to
	measure), not 0.0 (which would wrongly imply "context was available but ignored").
	judge_score is only populated when the request opted into the separate, costed
	LLM-as-judge path (ChatRequest.evaluate_with_llm) — see app/eval/evaluator.py.
	"""

	groundedness: float | None = Field(default=None, ge=0.0, le=1.0)
	structure_quality: float = Field(ge=0.0, le=1.0)
	overall_score: float = Field(ge=0.0, le=1.0)
	judge_score: float | None = Field(default=None, ge=0.0, le=1.0)
	notes: list[str] = Field(default_factory=list)


class CostEstimate(AegisSchema):
	"""Estimated USD cost for one LLM call, from app/llm/pricing.py's static price table.

	This is an estimate for demo/telemetry purposes, not billing-accurate accounting — actual
	provider invoices remain the source of truth. Unknown models fall back to a conservative
	default price rather than omitting cost entirely.
	"""

	input_tokens: int
	output_tokens: int
	cost_usd: float = Field(ge=0.0)
	model: str
	provider: str


class ModerationResult(AegisSchema):
	"""Deterministic, free safety screen — see app/moderation/moderator.py.

	blocked=True means the LLM was never called; content is a static refusal message, not a
	generated response. pii_flags lists categories of likely PII detected in the *response*
	text (email/phone/etc.) — flagged, not redacted, since redacting model output correctly
	without breaking its meaning is a much harder problem than flagging it for review.
	"""

	blocked: bool = False
	block_reason: str | None = None
	pii_flags: list[str] = Field(default_factory=list)


class AgentResult(AegisSchema):
	content: str
	routing: RoutingResult
	telemetry: ObservabilityEvent | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)
	tool_results: list[dict[str, Any]] = Field(default_factory=list)
	mission_brief: MissionBrief | None = None
	evaluation: ResponseEvaluation | None = None
	cost: CostEstimate | None = None
	moderation: ModerationResult | None = None
	conversation_id: str | None = None


class ToolType(StrEnum):
	SEARCH = "search"
	PYTHON = "python"
	RAG = "rag"
	FILE = "file"


class ToolRequest(AegisSchema):
	tool_type: ToolType
	input: str = Field(min_length=1)
	metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResult(AegisSchema):
	tool_type: ToolType
	output: str
	success: bool = True
	metadata: dict[str, Any] = Field(default_factory=dict)


class EventType(StrEnum):
	REQUEST_RECEIVED = "request.received"
	AGENT_SELECTED = "agent.selected"
	MODEL_SELECTED = "model.selected"
	PROVIDER_CALLED = "provider.called"
	REQUEST_COMPLETED = "request.completed"
	REQUEST_FAILED = "request.failed"
	TOOL_USED = "tool.used"


class EventRecord(AegisSchema):
	event_type: EventType
	timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	request_id: str | None = None
	agent: AgentType | None = None
	model: str | None = None
	provider: str | None = None
	tool_type: ToolType | None = None
	latency_ms: float | None = None
	input_tokens: int | None = None
	output_tokens: int | None = None
	error: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(AegisSchema):
	content: str
	routing: RoutingResult
	telemetry: ObservabilityEvent | None = None
	mission_brief: MissionBrief | None = None
	evaluation: ResponseEvaluation | None = None
	tool_results: list[dict[str, Any]] = Field(default_factory=list)
	cost: CostEstimate | None = None
	moderation: ModerationResult | None = None
	conversation_id: str | None = None
	sub_results: list["ChatResponse"] = Field(default_factory=list)


class RunRecordOut(AegisSchema):
	id: str
	created_at: datetime
	prompt: str
	status: str
	agent: str | None = None
	model: str | None = None
	provider: str | None = None
	risk_level: str | None = None
	risk_score: float | None = None
	latency_ms: float | None = None
	error: str | None = None


class ErrorResponse(AegisSchema):
	error: str
	detail: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)

