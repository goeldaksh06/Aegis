import type { ChatRequest, ChatResponse } from "../types/chat";

export interface ChatClientConfig {
  baseUrl: string;
}

export type TraceStageName =
  | "received"
  | "routed"
  | "moderation"
  | "memory"
  | "retrieval"
  | "generation"
  | "cost"
  | "brief"
  | "evaluated"
  | "substep_start"
  | "substep_done"
  | "persisted"
  | "done"
  | "error";

export interface TraceStageEvent {
  stage: TraceStageName;
  data: Record<string, unknown>;
}

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.trim().replace(/\/+$/, "");
}

/**
 * Streams POST /chat/stream (Server-Sent Events) and invokes onStage for each pipeline
 * checkpoint as it arrives. Native EventSource can't send a POST body, so this parses the
 * "event: name\ndata: json\n\n" wire format manually off a fetch() ReadableStream.
 * Resolves with the final ChatResponse once the "done" event arrives.
 */
export async function runChatStream(
  config: ChatClientConfig,
  request: ChatRequest,
  onStage: (event: TraceStageEvent) => void,
  token?: string | null,
  signal?: AbortSignal,
): Promise<ChatResponse> {
  const response = await fetch(`${normalizeBaseUrl(config.baseUrl)}/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(request),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(`Stream request failed: ${response.status} ${response.statusText}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalResponse: ChatResponse | null = null;
  let streamError: string | null = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";

    for (const chunk of chunks) {
      const eventLine = chunk.split("\n").find((line) => line.startsWith("event: "));
      const dataLine = chunk.split("\n").find((line) => line.startsWith("data: "));
      if (!eventLine || !dataLine) {
        continue;
      }

      const stage = eventLine.slice("event: ".length).trim() as TraceStageName;
      const data = JSON.parse(dataLine.slice("data: ".length));

      if (stage === "done") {
        finalResponse = data as ChatResponse;
      } else if (stage === "error") {
        streamError = (data as { message?: string }).message ?? "Stream failed.";
      }

      onStage({ stage, data });
    }
  }

  if (streamError) {
    throw new Error(streamError);
  }
  if (!finalResponse) {
    throw new Error("Stream ended without a final response.");
  }

  return finalResponse;
}
