import { afterEach, describe, expect, it, vi } from "vitest";

import { runChatRequest } from "./chatClient";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("runChatRequest", () => {
  it("posts the prompt to the backend chat endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          content: "ok",
          routing: {
            agent: "research",
            model: "stub-model",
            provider: "stub-provider",
            confidence: 1,
            reason: "stub",
          },
        }),
        {
          status: 200,
          headers: { "content-type": "application/json" },
        },
      ),
    );

    vi.stubGlobal("fetch", fetchMock);

    const result = await runChatRequest(
      { baseUrl: "http://localhost:8000/" },
      {
        messages: [{ role: "user", content: "Research the market" }],
      },
    );

    expect(result.content).toBe("ok");
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/chat",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({
          "Content-Type": "application/json",
        }),
      }),
    );
  });

  it("raises the backend error message when the request fails", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          detail: "backend unavailable",
        }),
        {
          status: 503,
          headers: { "content-type": "application/json" },
        },
      ),
    );

    vi.stubGlobal("fetch", fetchMock);

    await expect(
      runChatRequest(
        { baseUrl: "http://localhost:8000" },
        { messages: [{ role: "user", content: "Test" }] },
      ),
    ).rejects.toThrowError("backend unavailable");
  });
});