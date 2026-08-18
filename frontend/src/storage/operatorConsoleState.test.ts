import { afterEach, describe, expect, it, vi } from "vitest";

import {
  loadOperatorConsoleState,
  saveOperatorConsoleState,
} from "./operatorConsoleState";

afterEach(() => {
  vi.restoreAllMocks();
  window.localStorage.clear();
});

describe("operatorConsoleState storage", () => {
  it("returns fallback values when storage is empty", () => {
    const fallback = {
      backendUrl: "http://localhost:8000",
      prompt: "hello",
    };

    expect(loadOperatorConsoleState(fallback)).toEqual(fallback);
  });

  it("loads stored backend URL and prompt", () => {
    window.localStorage.setItem(
      "aegis.operatorConsoleState",
      JSON.stringify({
        backendUrl: "http://localhost:9000/",
        prompt: "Research the market",
      }),
    );

    expect(
      loadOperatorConsoleState({
        backendUrl: "http://localhost:8000",
        prompt: "fallback",
      }),
    ).toEqual({
      backendUrl: "http://localhost:9000",
      prompt: "Research the market",
    });
  });

  it("saves backend URL and prompt to localStorage", () => {
    saveOperatorConsoleState({
      backendUrl: "http://localhost:8000/",
      prompt: "Compare options",
    });

    expect(window.localStorage.getItem("aegis.operatorConsoleState")).toBe(
      JSON.stringify({
        backendUrl: "http://localhost:8000",
        prompt: "Compare options",
      }),
    );
  });
});