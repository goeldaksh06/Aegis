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
}

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.trim().replace(/\/+$/, "");
}

function buildRunsUrl(baseUrl: string): string {
  return `${normalizeBaseUrl(baseUrl)}${DEFAULT_RUNS_PATH}`;
}

export async function fetchPersistedRuns(
  config: ChatClientConfig,
  signal?: AbortSignal,
): Promise<RunRecordPayload[]> {
  const response = await fetch(buildRunsUrl(config.baseUrl), {
    method: "GET",
    headers: {
      Accept: "application/json",
    },
    signal,
  });

  if (!response.ok) {
    throw new Error(`Fetching run history failed (${response.status})`);
  }

  return (await response.json()) as RunRecordPayload[];
}
