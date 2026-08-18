import type { ChatClientConfig } from "./chatClient";

const DEFAULT_HEALTH_PATH = "/health";

export interface HealthResponse {
  status: string;
}

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.trim().replace(/\/+$/, "");
}

function buildHealthUrl(baseUrl: string): string {
  return `${normalizeBaseUrl(baseUrl)}${DEFAULT_HEALTH_PATH}`;
}

export async function checkBackendHealth(
  config: ChatClientConfig,
  signal?: AbortSignal,
): Promise<HealthResponse> {
  const response = await fetch(buildHealthUrl(config.baseUrl), {
    method: "GET",
    headers: {
      Accept: "application/json",
    },
    signal,
  });

  if (!response.ok) {
    throw new Error(`Health check failed (${response.status})`);
  }

  return (await response.json()) as HealthResponse;
}