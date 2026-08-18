import type { ChatRequest, ChatResponse } from "../types/chat";

export interface ChatClientConfig {
  baseUrl: string;
}

export interface ChatClientErrorShape {
  error?: string;
  detail?: string;
  message?: string;
}

const DEFAULT_CHAT_PATH = "/chat";

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.trim().replace(/\/+$/, "");
}

function buildChatUrl(baseUrl: string): string {
  return `${normalizeBaseUrl(baseUrl)}${DEFAULT_CHAT_PATH}`;
}

async function parseErrorMessage(response: Response): Promise<string> {
  const contentType = response.headers.get("content-type") ?? "";

  if (contentType.includes("application/json")) {
    try {
      const payload = (await response.json()) as ChatClientErrorShape;
      return payload.detail ?? payload.error ?? payload.message ?? response.statusText;
    } catch {
      return response.statusText;
    }
  }

  try {
    const text = await response.text();
    return text.trim() || response.statusText;
  } catch {
    return response.statusText;
  }
}

export async function runChatRequest(
  config: ChatClientConfig,
  request: ChatRequest,
): Promise<ChatResponse> {
  const response = await fetch(buildChatUrl(config.baseUrl), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }

  return (await response.json()) as ChatResponse;
}