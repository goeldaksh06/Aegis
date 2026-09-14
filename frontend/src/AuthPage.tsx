import { FormEvent, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";

import { login, register } from "./api/authClient";
import { loadOperatorConsoleState } from "./storage/operatorConsoleState";

const DEFAULT_BASE_URL = import.meta.env.VITE_AEGIS_API_BASE_URL ?? "http://127.0.0.1:8000";

export function AuthPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const initialMode = location.pathname.includes("register") ? "register" : "login";

  const [mode, setMode] = useState<"login" | "register">(initialMode);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const backendUrl = loadOperatorConsoleState({ backendUrl: DEFAULT_BASE_URL, prompt: "" }).backendUrl;

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setBusy(true);

    try {
      const result =
        mode === "login"
          ? await login({ baseUrl: backendUrl }, email, password)
          : await register({ baseUrl: backendUrl }, email, password);

      localStorage.setItem("aegis_auth_token", result.access_token);
      localStorage.setItem("aegis_auth_user", JSON.stringify(result.user));
      navigate("/console");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong. Please try again.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="auth-page">
      <div className="auth-page__card">
        <Link to="/" className="auth-page__brand">
          Aegis
        </Link>

        <div className="auth-page__tabs" role="tablist">
          <button
            type="button"
            role="tab"
            aria-selected={mode === "login"}
            className={`auth-page__tab${mode === "login" ? " auth-page__tab--active" : ""}`}
            onClick={() => {
              setMode("login");
              setError(null);
            }}
          >
            Sign in
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === "register"}
            className={`auth-page__tab${mode === "register" ? " auth-page__tab--active" : ""}`}
            onClick={() => {
              setMode("register");
              setError(null);
            }}
          >
            Create account
          </button>
        </div>

        <h1 className="auth-page__title">
          {mode === "login" ? "Welcome back" : "Create your account"}
        </h1>
        <p className="auth-page__subtitle">
          {mode === "login"
            ? "Sign in to see your saved mission history and per-agent cost/token observability."
            : "Takes a few seconds. No credit card, no verification email."}
        </p>

        <form onSubmit={handleSubmit} className="auth-page__form">
          <label className="auth-page__label" htmlFor="auth-email">
            Email
          </label>
          <input
            id="auth-email"
            type="email"
            required
            autoFocus
            autoComplete="email"
            className="endpoint-input"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
          />

          <label className="auth-page__label" htmlFor="auth-password">
            Password
          </label>
          <input
            id="auth-password"
            type="password"
            required
            minLength={8}
            autoComplete={mode === "login" ? "current-password" : "new-password"}
            placeholder={mode === "register" ? "At least 8 characters" : undefined}
            className="endpoint-input"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />

          {error ? (
            <p className="auth-page__error" role="alert">
              {error}
            </p>
          ) : null}

          <button type="submit" className="run-button auth-page__submit" disabled={busy}>
            {busy ? "One moment…" : mode === "login" ? "Sign in" : "Create account"}
          </button>
        </form>

        <p className="auth-page__switch">
          {mode === "login" ? (
            <>
              Don't have an account?{" "}
              <button type="button" onClick={() => setMode("register")}>
                Create one
              </button>
            </>
          ) : (
            <>
              Already have an account?{" "}
              <button type="button" onClick={() => setMode("login")}>
                Sign in
              </button>
            </>
          )}
        </p>

        <Link to="/console" className="auth-page__skip">
          Skip — try the anonymous demo instead →
        </Link>
      </div>
    </div>
  );
}
