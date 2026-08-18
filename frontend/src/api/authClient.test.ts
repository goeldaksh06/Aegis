import { afterEach, describe, expect, it, vi } from "vitest";

import { login, register } from "./authClient";

afterEach(() => {
  vi.restoreAllMocks();
});

const TOKEN_PAYLOAD = {
  access_token: "token-123",
  token_type: "bearer",
  user: { id: "u1", email: "a@example.com", created_at: "2026-01-01T00:00:00Z" },
};

describe("register", () => {
  it("posts email/password to /auth/register", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(TOKEN_PAYLOAD), {
        status: 201,
        headers: { "content-type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await register({ baseUrl: "http://localhost:8000" }, "a@example.com", "pw123456");

    expect(result.access_token).toBe("token-123");
    const call = fetchMock.mock.calls[0];
    expect(call).toBeDefined();
    const [url, init] = call!;
    expect(url).toBe("http://localhost:8000/auth/register");
    expect(JSON.parse(init.body)).toEqual({ email: "a@example.com", password: "pw123456" });
  });

  it("throws the server's error detail on failure", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: "An account with this email already exists." }), {
        status: 409,
        headers: { "content-type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(register({ baseUrl: "http://localhost:8000" }, "a@example.com", "pw123456")).rejects.toThrowError(
      "An account with this email already exists.",
    );
  });
});

describe("login", () => {
  it("posts email/password to /auth/login", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(TOKEN_PAYLOAD), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await login({ baseUrl: "http://localhost:8000" }, "a@example.com", "pw123456");

    expect(result.user.email).toBe("a@example.com");
    const call = fetchMock.mock.calls[0];
    expect(call).toBeDefined();
    const [url] = call!;
    expect(url).toBe("http://localhost:8000/auth/login");
  });
});
