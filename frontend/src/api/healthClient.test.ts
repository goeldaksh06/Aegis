import { afterEach, describe, expect, it, vi } from "vitest";

import { checkBackendHealth } from "./healthClient";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("checkBackendHealth", () => {
  it("calls the backend health endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ status: "healthy" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );

    vi.stubGlobal("fetch", fetchMock);

    const result = await checkBackendHealth({ baseUrl: "http://localhost:8000/" });

    expect(result.status).toBe("healthy");
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/health",
      expect.objectContaining({
        method: "GET",
      }),
    );
  });

  it("raises an error when health endpoint fails", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response("oops", {
        status: 503,
      }),
    );

    vi.stubGlobal("fetch", fetchMock);

    await expect(checkBackendHealth({ baseUrl: "http://localhost:8000" })).rejects.toThrowError(
      "Health check failed (503)",
    );
  });
});