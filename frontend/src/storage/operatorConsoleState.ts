export interface OperatorConsoleStateSnapshot {
  backendUrl: string;
  prompt: string;
}

const STORAGE_KEY = "aegis.operatorConsoleState";

function hasStorage(): boolean {
  return typeof window !== "undefined" && typeof window.localStorage !== "undefined";
}

function normalizeValue(value: unknown, fallback: string): string {
  if (typeof value !== "string") {
    return fallback;
  }

  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed.replace(/\/+$/, "") : fallback;
}

export function loadOperatorConsoleState(
  fallback: OperatorConsoleStateSnapshot,
): OperatorConsoleStateSnapshot {
  if (!hasStorage()) {
    return fallback;
  }

  try {
    const rawValue = window.localStorage.getItem(STORAGE_KEY);

    if (!rawValue) {
      return fallback;
    }

    const parsed = JSON.parse(rawValue) as Partial<OperatorConsoleStateSnapshot>;

    return {
      backendUrl: normalizeValue(parsed.backendUrl, fallback.backendUrl),
      prompt: normalizeValue(parsed.prompt, fallback.prompt),
    };
  } catch {
    return fallback;
  }
}

export function saveOperatorConsoleState(state: OperatorConsoleStateSnapshot): void {
  if (!hasStorage()) {
    return;
  }

  try {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        backendUrl: state.backendUrl.trim().replace(/\/+$/, ""),
        prompt: state.prompt,
      }),
    );
  } catch {
    // Ignore storage failures in constrained browser environments.
  }
}