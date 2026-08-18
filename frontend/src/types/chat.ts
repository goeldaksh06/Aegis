export type MessageRole = "system" | "user" | "assistant" | "tool";

export interface ChatMessage {
  role: MessageRole;
  content: string;
}

export interface ChatRequest {
  messages: ChatMessage[];
  preference?: string;
  preferred_provider?: string | null;
  agent_hint?: string | null;
  model_hint?: string | null;
  evaluate_with_llm?: boolean;
  conversation_id?: string | null;
  orchestrate?: boolean;
}

export interface RoutingResult {
  agent: string;
  model: string;
  provider: string;
  preference?: string;
  confidence: number;
  reason: string;
  metadata?: Record<string, unknown>;
}

export interface ObservabilityEvent {
  event_type: string;
  timestamp: string;
  request_id?: string | null;
  agent?: string | null;
  model?: string | null;
  provider?: string | null;
  latency_ms?: number | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
  error?: string | null;
  metadata?: Record<string, unknown>;
}

export type RiskLevel = "low" | "medium" | "high";

export interface MissionBriefPayload {
  summary: string;
  risk_score: number;
  risk_level: RiskLevel;
  top_alerts: string[];
  recommended_actions: string[];
  evidence: string[];
}

export interface ResponseEvaluation {
  groundedness: number | null;
  structure_quality: number;
  overall_score: number;
  judge_score: number | null;
  notes: string[];
}

export interface CostEstimate {
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  model: string;
  provider: string;
}

export interface ModerationResult {
  blocked: boolean;
  block_reason: string | null;
  pii_flags: string[];
}

export interface ChatResponse {
  content: string;
  routing: RoutingResult;
  telemetry?: ObservabilityEvent | null;
  mission_brief?: MissionBriefPayload | null;
  tool_results?: Array<Record<string, unknown>>;
  evaluation?: ResponseEvaluation | null;
  cost?: CostEstimate | null;
  moderation?: ModerationResult | null;
  conversation_id?: string | null;
  sub_results?: ChatResponse[];
  run_id?: string | null;
}

export interface OperatorRunState {
  prompt: string;
  response: ChatResponse | null;
  status: "idle" | "loading" | "success" | "error";
  error: string | null;
}

export interface OperatorRunRecord {
  id: string;
  runId?: string | null;
  createdAt: string;
  prompt: string;
  status: "success" | "error";
  agent?: string;
  model?: string;
  provider?: string;
  latencyMs?: number;
  error?: string;
  riskLevel?: RiskLevel;
  riskScore?: number;
  qualityScore?: number;
  costUsd?: number;
}