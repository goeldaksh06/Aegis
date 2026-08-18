import type { ChatClientConfig } from "./chatClient";

const DEFAULT_RUNS_PATH = "/runs";

export interface RunRecordPayload {
  id: string;
  created_at: string;
  prompt: string;
  status: "success" | "error";
  agent?: string | null;
  model?: string | null;
  provider?: string | null;
  risk_level?: string | null;
  risk_score?: number | null;
  latency_ms?: number | null;
  error?: string | null;
  cost_usd?: number | null;
  moderation_blocked?: boolean | null;
  conversation_id?: string | null;
}

export interface AgentStepPayload {
  agent: string;
  step_index: number;
  model?: string | null;
  provider?: string | null;
  duration_ms?: number | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
  cost_usd?: number | null;
  retrieved_count?: number | null;
  status: string;
  error?: string | null;
}

export interface RunDetailPayload extends RunRecordPayload {
  steps: AgentStepPayload[];
  total_duration_ms?: number | null;
  total_tokens?: number | null;
  total_cost_usd?: number | null;
}

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.trim().replace(/\/+$/, "");
}

function authHeaders(token?: string | null): Record<string, string> {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export async function fetchPersistedRuns(
  config: ChatClientConfig,
  token?: string | null,
  signal?: AbortSignal,
): Promise<RunRecordPayload[]> {
  const response = await fetch(`${normalizeBaseUrl(config.baseUrl)}${DEFAULT_RUNS_PATH}`, {
    method: "GET",
    headers: { Accept: "application/json", ...authHeaders(token) },
    signal,
  });

  if (!response.ok) {
    throw new Error(`Fetching run history failed (${response.status})`);
  }

  return (await response.json()) as RunRecordPayload[];
}

export async function fetchRunDetail(
  config: ChatClientConfig,
  runId: string,
  token?: string | null,
  signal?: AbortSignal,
): Promise<RunDetailPayload> {
  const response = await fetch(`${normalizeBaseUrl(config.baseUrl)}${DEFAULT_RUNS_PATH}/${runId}`, {
    method: "GET",
    headers: { Accept: "application/json", ...authHeaders(token) },
    signal,
  });

  if (!response.ok) {
    throw new Error(`Fetching mission detail failed (${response.status})`);
  }

  return (await response.json()) as RunDetailPayload;
}
