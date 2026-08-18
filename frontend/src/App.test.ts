import { describe, expect, it } from "vitest";

import {
  buildHistoryRecord,
  extractRagChunks,
  formatConfidence,
  friendlyModerationMessage,
} from "./App";
import type { ChatResponse } from "./types/chat";

describe("friendlyModerationMessage", () => {
  it("never includes implementation details like a raw regex pattern", () => {
    const message = friendlyModerationMessage();
    const combined = `${message.reason} ${message.whatHappened} ${message.whatToDo}`;

    // Telltale signs of a leaked regex source (alternation groups, character classes) —
    // the backend's raw block_reason includes exactly this kind of text, which is why the
    // frontend deliberately never renders it directly.
    expect(combined).not.toMatch(/\([^)]*\|[^)]*\)/);
    expect(combined).not.toMatch(/\[[a-zA-Z-]+\]/);
    expect(combined.toLowerCase()).not.toContain("regex");
    expect(combined.toLowerCase()).not.toContain("matched pattern");
  });

  it("explains what happened and what the user can do", () => {
    const message = friendlyModerationMessage();
    expect(message.whatHappened.length).toBeGreaterThan(0);
    expect(message.whatToDo.length).toBeGreaterThan(0);
  });
});

describe("extractRagChunks", () => {
  it("returns an empty array when no RAG tool result is present", () => {
    expect(extractRagChunks([])).toEqual([]);
    expect(extractRagChunks([{ tool_type: "python", metadata: {} }])).toEqual([]);
  });

  it("extracts chunk id/text/score/source from a real tool_results shape", () => {
    const toolResults = [
      {
        tool_type: "rag",
        metadata: {
          chunks: [
            {
              id: "c1",
              text: "Some retrieved evidence text.",
              score: 0.87,
              metadata: { source: "supply-chain-contingency-playbook" },
            },
          ],
        },
      },
    ];

    expect(extractRagChunks(toolResults)).toEqual([
      {
        id: "c1",
        text: "Some retrieved evidence text.",
        score: 0.87,
        source: "supply-chain-contingency-playbook",
      },
    ]);
  });
});

describe("formatConfidence", () => {
  it("formats a 0-1 confidence as a rounded percentage", () => {
    expect(formatConfidence(0.82)).toBe("82%");
    expect(formatConfidence(1)).toBe("100%");
  });

  it("falls back to n/a for non-finite values", () => {
    expect(formatConfidence(Number.NaN)).toBe("n/a");
  });
});

describe("buildHistoryRecord", () => {
  function chatResponse(overrides: Partial<ChatResponse> = {}): ChatResponse {
    return {
      content: "answer",
      routing: { agent: "research", model: "m", provider: "p", confidence: 1, reason: "r" },
      ...overrides,
    };
  }

  it("captures risk, quality, and cost from a successful response", () => {
    const response = chatResponse({
      mission_brief: {
        summary: "s",
        risk_score: 72,
        risk_level: "high",
        top_alerts: [],
        recommended_actions: [],
        evidence: [],
      },
      evaluation: { groundedness: 0.9, structure_quality: 1, overall_score: 0.95, judge_score: null, notes: [] },
      cost: { input_tokens: 10, output_tokens: 5, cost_usd: 0.001, model: "m", provider: "p" },
    });

    const record = buildHistoryRecord("prompt", response, null);

    expect(record.status).toBe("success");
    expect(record.riskLevel).toBe("high");
    expect(record.qualityScore).toBe(0.95);
    expect(record.costUsd).toBe(0.001);
  });

  it("builds an error record with no risk/quality/cost when the request failed", () => {
    const record = buildHistoryRecord("prompt", null, "Request failed.");

    expect(record.status).toBe("error");
    expect(record.error).toBe("Request failed.");
    expect(record.riskLevel).toBeUndefined();
  });
});
