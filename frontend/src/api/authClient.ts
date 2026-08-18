export interface AuthClientConfig {
  baseUrl: string;
}

export interface UserPayload {
  id: string;
  email: string;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  user: UserPayload;
}

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.trim().replace(/\/+$/, "");
}

async function parseErrorMessage(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: string };
    return payload.detail ?? response.statusText;
  } catch {
    return response.statusText;
  }
}

async function postJson(baseUrl: string, path: string, body: unknown): Promise<TokenResponse> {
  const response = await fetch(`${normalizeBaseUrl(baseUrl)}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }

  return (await response.json()) as TokenResponse;
}

export async function register(
  config: AuthClientConfig,
  email: string,
  password: string,
): Promise<TokenResponse> {
  return postJson(config.baseUrl, "/auth/register", { email, password });
}

export async function login(
  config: AuthClientConfig,
  email: string,
  password: string,
): Promise<TokenResponse> {
  return postJson(config.baseUrl, "/auth/login", { email, password });
}
