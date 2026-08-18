import { KeyboardEvent, FormEvent, useEffect, useState } from "react";

import { runChatStream, type TraceStageEvent } from "./api/chatStreamClient";
import { checkBackendHealth } from "./api/healthClient";
import { fetchPersistedRuns } from "./api/runsClient";
import { Panel } from "./components/Panel";
import {
  loadOperatorConsoleState,
  saveOperatorConsoleState,
} from "./storage/operatorConsoleState";
import type { ChatResponse, OperatorRunRecord, OperatorRunState } from "./types/chat";

const DEFAULT_PROMPT =
  "A key supplier just went dark after a port disruption cut off our main shipping lane. " +
  "Analyze the risk to our fulfillment commitments this quarter and recommend an immediate response.";

interface ScenarioChip {
  label: string;
  prompt: string;
}

const SCENARIO_CHIPS: ScenarioChip[] = [
  { label: "Supply chain", prompt: DEFAULT_PROMPT },
  {
    label: "Cyber incident",
    prompt:
      "Our customer database was exposed in a suspected breach discovered 2 hours ago. " +
      "Analyze the risk and recommend an immediate response.",
  },
  {
    label: "Market shock",
    prompt:
      "A major competitor just filed for bankruptcy, triggering a 15% drop in our sector's stock prices overnight. " +
      "Research the market implications and recommend how we should respond this week.",
  },
  {
    label: "Natural disaster",
    prompt:
      "A category 4 hurricane is projected to make landfall near our primary distribution center in 48 hours. " +
      "Analyze the operational risk and recommend an immediate response.",
  },
];

const DEFAULT_BASE_URL = import.meta.env.VITE_AEGIS_API_BASE_URL ?? "http://127.0.0.1:8000";
const RUN_HISTORY_LIMIT = 6;
const INITIAL_OPERATOR_STATE = loadOperatorConsoleState({
  backendUrl: DEFAULT_BASE_URL,
  prompt: DEFAULT_PROMPT,
});

type RiskLevel = "low" | "medium" | "high";

interface MissionBrief {
  summary: string;
  riskScore: number;
  riskLevel: RiskLevel;
  topAlerts: string[];
  recommendedActions: string[];
  evidence: string[];
}

function formatJson(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function formatConfidence(value: number): string {
  return Number.isFinite(value) ? `${Math.round(value * 100)}%` : "n/a";
}

function getTelemetrySummary(response: ChatResponse | null): string {
  if (!response?.telemetry) {
    return "No telemetry returned.";
  }

  const { telemetry } = response;
  const summaryParts = [telemetry.event_type];

  if (telemetry.request_id) {
    summaryParts.push(`request ${telemetry.request_id}`);
  }

  if (telemetry.latency_ms != null) {
    summaryParts.push(`${Math.round(telemetry.latency_ms)} ms`);
  }

  return summaryParts.join(" · ");
}

function sanitizeBullet(line: string): string {
  return line.replace(/^[-*\d.)\s]+/, "").trim();
}

function toSentenceList(content: string): string[] {
  return content
    .replace(/\s+/g, " ")
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter((sentence) => sentence.length > 0);
}

function dedupeAndLimit(items: string[], limit: number): string[] {
  const uniqueItems = new Set<string>();

  for (const item of items) {
    if (!uniqueItems.has(item)) {
      uniqueItems.add(item);
    }

    if (uniqueItems.size >= limit) {
      break;
    }
  }

  return Array.from(uniqueItems);
}

function scoreRisk(content: string): { score: number; level: RiskLevel } {
  const text = content.toLowerCase();
  let score = 28;

  const weightedSignals: Array<[RegExp, number]> = [
    [/(critical|severe|catastrophic|collapse)/g, 20],
    [/(immediate|urgent|escalat|emergency)/g, 12],
    [/(disruption|outage|shortage|bottleneck|delay)/g, 9],
    [/(volatility|exposure|liquidity|compliance|regulatory)/g, 7],
    [/(mitigat|contingenc|monitor|fallback)/g, -6],
  ];

  for (const [pattern, weight] of weightedSignals) {
    const matches = text.match(pattern);
    if (matches) {
      score += matches.length * weight;
    }
  }

  score = Math.max(10, Math.min(95, score));

  if (score >= 70) {
    return { score, level: "high" };
  }

  if (score >= 40) {
    return { score, level: "medium" };
  }

  return { score, level: "low" };
}

function missionBriefFromBackend(response: ChatResponse | null): MissionBrief | null {
  const brief = response?.mission_brief;
  if (!brief) {
    return null;
  }

  return {
    summary: brief.summary,
    riskScore: brief.risk_score,
    riskLevel: brief.risk_level,
    topAlerts: brief.top_alerts,
    recommendedActions: brief.recommended_actions,
    evidence: brief.evidence,
  };
}

// Fallback only: used when the backend didn't return a mission_brief (e.g. an older
// backend, or an agent that hasn't adopted structured output yet) — the backend-owned
// brief in mission_brief.py is the source of truth when present.
function buildMissionBriefFallback(response: ChatResponse | null): MissionBrief | null {
  if (!response?.content?.trim()) {
    return null;
  }

  const content = response.content.trim();
  const lines = content
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
  const sentences = toSentenceList(content);
  const summary = sentences.slice(0, 2).join(" ").slice(0, 320) || content.slice(0, 320);
  const { score, level } = scoreRisk(content);

  const alertSignals = /(risk|alert|threat|impact|loss|delay|shortage|outage|vulnerab|exposure)/i;
  const actionSignals = /(recommend|action|mitigat|prioriti|deploy|activate|coordinate|communicat|escalat|hedge|review)/i;
  const evidenceSignals = /(because|due to|according|signal|indicator|data|metric|report|trend|evidence|observed)/i;

  const sectionHeader = /^[A-Za-z][\w /-]{0,40}:$/;
  const bulletLines = lines.filter((line) => !sectionHeader.test(line)).map(sanitizeBullet);
  const topAlerts = dedupeAndLimit(
    bulletLines.filter((line) => alertSignals.test(line)).concat(sentences.filter((line) => alertSignals.test(line))),
    3,
  );

  const recommendedActions = dedupeAndLimit(
    bulletLines
      .filter((line) => actionSignals.test(line))
      .concat(sentences.filter((line) => /^(use|shift|activate|review|reduce|increase|secure|prepare|monitor)/i.test(line))),
    4,
  );

  const evidence = dedupeAndLimit(
    bulletLines.filter((line) => evidenceSignals.test(line)).concat(sentences.filter((line) => evidenceSignals.test(line))),
    3,
  );

  return {
    summary,
    riskScore: score,
    riskLevel: level,
    topAlerts: topAlerts.length > 0 ? topAlerts : ["No explicit alerts extracted from this response."],
    recommendedActions:
      recommendedActions.length > 0
        ? recommendedActions
        : ["No concrete actions were extracted. Ask for action-oriented recommendations."],
    evidence:
      evidence.length > 0
        ? evidence
        : ["No explicit evidence lines found. Ask the model to cite data or sources."],
  };
}

type BackendHealthState =
  | { status: "checking"; label: string }
  | { status: "healthy"; label: string }
  | { status: "unhealthy"; label: string };

function buildHistoryRecord(
  prompt: string,
  response: ChatResponse | null,
  error: string | null,
): OperatorRunRecord {
  const now = new Date();

  if (!response || error) {
    return {
      id: `${now.getTime()}-error`,
      createdAt: now.toISOString(),
      prompt,
      status: "error",
      error: error ?? "Request failed",
    };
  }

  return {
    id: `${now.getTime()}-success`,
    createdAt: now.toISOString(),
    prompt,
    status: "success",
    agent: response.routing.agent,
    model: response.routing.model,
    provider: response.routing.provider,
    latencyMs: response.telemetry?.latency_ms ?? undefined,
  };
}

const RISK_GAUGE_COLORS: Record<RiskLevel, string> = {
  low: "#34d399",
  medium: "#f6ad55",
  high: "#f87171",
};

function RiskGauge({ score, level }: { score: number; level: RiskLevel }) {
  const radius = 52;
  const circumference = 2 * Math.PI * radius;
  const clamped = Math.max(0, Math.min(100, score));
  const offset = circumference * (1 - clamped / 100);
  const color = RISK_GAUGE_COLORS[level];

  return (
    <svg
      className="risk-gauge"
      viewBox="0 0 120 120"
      role="img"
      aria-label={`Risk score ${clamped} out of 100, ${level} risk`}
    >
      <circle className="risk-gauge__track" cx="60" cy="60" r={radius} />
      <circle
        className="risk-gauge__fill"
        cx="60"
        cy="60"
        r={radius}
        stroke={color}
        strokeDasharray={circumference}
        strokeDashoffset={offset}
      />
      <text x="60" y="56" className="risk-gauge__value" fill={color}>
        {clamped}
      </text>
      <text x="60" y="76" className="risk-gauge__unit">
        / 100
      </text>
    </svg>
  );
}

const TRACE_STAGE_ORDER: TraceStageEvent["stage"][] = [
  "received",
  "routed",
  "moderation",
  "memory",
  "retrieval",
  "generation",
  "cost",
  "brief",
  "evaluated",
  "persisted",
];

// Stages that legitimately never fire for a given request (e.g. "memory" only fires when a
// conversation_id was supplied) — excluded from the "did every prior stage happen" check so
// their absence doesn't permanently freeze the pulsing "active" indicator on later steps.
const OPTIONAL_STAGES = new Set<TraceStageEvent["stage"]>(["memory"]);

const TRACE_STAGE_LABELS: Record<string, string> = {
  received: "Request received",
  routed: "Agent routing",
  moderation: "Safety check",
  memory: "Conversation memory",
  retrieval: "RAG retrieval",
  generation: "LLM generation",
  cost: "Cost estimated",
  brief: "Mission brief built",
  evaluated: "Quality evaluated",
  persisted: "Persisted to database",
  error: "Error",
};

function summarizeStage(event: TraceStageEvent): string {
  const d = event.data as Record<string, any>;
  switch (event.stage) {
    case "received":
      return `${d.message_count} message(s) received by the backend`;
    case "routed":
      return `Routed to "${d.agent}" agent · ${Math.round(d.confidence * 100)}% confidence — ${d.reason}`;
    case "moderation":
      return d.blocked ? `Blocked — ${d.reason}` : "Prompt passed the safety screen";
    case "memory":
      return `${d.prior_message_count} prior message(s) loaded into context`;
    case "retrieval":
      return d.retrieved_count > 0
        ? `${d.retrieved_count} chunk(s) retrieved from [${d.sources.join(", ")}] in ${d.elapsed_ms}ms`
        : `No relevant context found in ${d.elapsed_ms}ms`;
    case "generation":
      return `${d.provider}/${d.model} responded in ${d.elapsed_ms}ms${d.output_tokens ? ` · ${d.output_tokens} tokens` : ""}`;
    case "cost":
      return `~$${Number(d.cost_usd).toFixed(6)} (${d.model})`;
    case "brief":
      return d.risk_score != null ? `Risk score ${d.risk_score}/100 (${d.risk_level})` : "No structured brief extracted";
    case "evaluated":
      return `Quality ${Math.round(d.overall_score * 100)}%${d.groundedness != null ? ` · grounded ${Math.round(d.groundedness * 100)}%` : ""}`;
    case "persisted":
      return `Full pipeline completed in ${d.total_elapsed_ms}ms`;
    case "error":
      return String(d.message ?? "Something went wrong.");
    default:
      return JSON.stringify(d);
  }
}

function ExecutionTrace({ events, isLive }: { events: TraceStageEvent[]; isLive: boolean }) {
  const seen = new Map(events.map((event) => [event.stage, event]));
  const stages = events.some((event) => event.stage === "error")
    ? [...TRACE_STAGE_ORDER.filter((stage) => seen.has(stage)), "error" as const]
    : TRACE_STAGE_ORDER;

  const substepEvents = events.filter((e) => e.stage === "substep_start" || e.stage === "substep_done");
  const substepsByIndex = new Map<number, { step: string; agent: string; started: boolean; done: boolean }>();
  for (const event of substepEvents) {
    const d = event.data as Record<string, any>;
    const existing = substepsByIndex.get(d.index) ?? { step: d.step ?? "", agent: d.agent, started: false, done: false };
    if (event.stage === "substep_start") existing.started = true;
    if (event.stage === "substep_done") existing.done = true;
    if (d.agent) existing.agent = d.agent;
    substepsByIndex.set(d.index, existing);
  }

  return (
    <>
      <ol className="execution-trace">
        {stages.map((stage, index) => {
          const event = seen.get(stage);
          const isDone = Boolean(event);
          const isActive =
            isLive &&
            !isDone &&
            stages.slice(0, index).every((s) => seen.has(s) || OPTIONAL_STAGES.has(s));

          return (
            <li
              key={stage}
              className={`execution-trace__step${isDone ? " execution-trace__step--done" : ""}${
                isActive ? " execution-trace__step--active" : ""
              }${stage === "error" ? " execution-trace__step--error" : ""}`}
            >
              <span className="execution-trace__marker">{isDone ? (stage === "error" ? "!" : "✓") : index + 1}</span>
              <div className="execution-trace__body">
                <span className="execution-trace__label">{TRACE_STAGE_LABELS[stage]}</span>
                {event ? <span className="execution-trace__detail">{summarizeStage(event)}</span> : null}
              </div>
            </li>
          );
        })}
      </ol>
      {substepsByIndex.size > 0 ? (
        <div className="orchestration-trace">
          <p className="orchestration-trace__label">Multi-agent handoff</p>
          <ol className="execution-trace">
            {[...substepsByIndex.entries()]
              .sort(([a], [b]) => a - b)
              .map(([index, sub]) => (
                <li
                  key={index}
                  className={`execution-trace__step${sub.done ? " execution-trace__step--done" : sub.started ? " execution-trace__step--active" : ""}`}
                >
                  <span className="execution-trace__marker">{sub.done ? "✓" : index}</span>
                  <div className="execution-trace__body">
                    <span className="execution-trace__label">
                      Sub-task {index}: {sub.agent} agent
                    </span>
                    <span className="execution-trace__detail">{sub.step}</span>
                  </div>
                </li>
              ))}
          </ol>
        </div>
      ) : null}
    </>
  );
}

export function App() {
  const [backendUrl, setBackendUrl] = useState(INITIAL_OPERATOR_STATE.backendUrl);
  const [backendHealth, setBackendHealth] = useState<BackendHealthState>({
    status: "checking",
    label: "Checking backend",
  });
  const [runHistory, setRunHistory] = useState<OperatorRunRecord[]>([]);
  const [traceEvents, setTraceEvents] = useState<TraceStageEvent[]>([]);
  const [agentOverride, setAgentOverride] = useState<
    "auto" | "research" | "analyst" | "coder" | "document" | "planner"
  >("auto");
  const [orchestrate, setOrchestrate] = useState(false);
  const [conversationMode, setConversationMode] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [turnCount, setTurnCount] = useState(0);
  const [transcript, setTranscript] = useState<
    Array<{ role: "user" | "assistant"; content: string; agent?: string }>
  >([]);
  const [sessionStats, setSessionStats] = useState({ requests: 0, totalCostUsd: 0, blockedCount: 0 });
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const [state, setState] = useState<OperatorRunState>({
    prompt: INITIAL_OPERATOR_STATE.prompt,
    response: null,
    status: "idle",
    error: null,
  });

  const canRun = state.status !== "loading" && state.prompt.trim().length > 0;
  const hasFailedHistory = runHistory.some((record) => record.status === "error");
  const toolResults = state.response?.tool_results ?? [];
  const missionBrief =
    missionBriefFromBackend(state.response) ?? buildMissionBriefFallback(state.response);

  async function refreshBackendHealth(signal?: AbortSignal) {
    setBackendHealth({ status: "checking", label: "Checking backend" });

    try {
      const health = await checkBackendHealth({ baseUrl: backendUrl }, signal);
      setBackendHealth({
        status: health.status.toLowerCase() === "healthy" ? "healthy" : "unhealthy",
        label: health.status,
      });
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        return;
      }

      setBackendHealth({ status: "unhealthy", label: "Unavailable" });
    }
  }

  useEffect(() => {
    const controller = new AbortController();
    void refreshBackendHealth(controller.signal);

    return () => {
      controller.abort();
    };
  }, [backendUrl]);

  useEffect(() => {
    const controller = new AbortController();

    fetchPersistedRuns({ baseUrl: backendUrl }, controller.signal)
      .then((records) => {
        setRunHistory(
          records.slice(0, RUN_HISTORY_LIMIT).map((record) => ({
            id: record.id,
            createdAt: record.created_at,
            prompt: record.prompt,
            status: record.status,
            agent: record.agent ?? undefined,
            model: record.model ?? undefined,
            provider: record.provider ?? undefined,
            latencyMs: record.latency_ms ?? undefined,
            error: record.error ?? undefined,
          })),
        );
      })
      .catch(() => {
        // Backend history is a convenience seed, not required — local history still
        // accumulates via buildHistoryRecord() on each run regardless.
      });

    return () => {
      controller.abort();
    };
  }, [backendUrl]);

  useEffect(() => {
    saveOperatorConsoleState({
      backendUrl,
      prompt: state.prompt,
    });
  }, [backendUrl, state.prompt]);

  async function runPrompt(prompt: string) {
    setState((current) => ({ ...current, status: "loading", error: null }));
    setTraceEvents([]);

    try {
      const effectiveConversationId = conversationMode ? conversationId ?? crypto.randomUUID() : null;

      const response = await runChatStream(
        { baseUrl: backendUrl },
        {
          messages: [{ role: "user", content: prompt }],
          preference: "balanced",
          agent_hint: agentOverride === "auto" ? null : agentOverride,
          conversation_id: effectiveConversationId,
          orchestrate,
        },
        (event) => {
          if (event.stage !== "done") {
            setTraceEvents((current) => [...current, event]);
          }
        },
      );

      if (conversationMode && response.conversation_id) {
        setConversationId(response.conversation_id);
        setTurnCount((current) => current + 1);
        setTranscript((current) => [
          ...current,
          { role: "user", content: prompt },
          { role: "assistant", content: response.content, agent: response.routing.agent },
        ]);
      }

      setSessionStats((current) => ({
        requests: current.requests + 1,
        totalCostUsd: current.totalCostUsd + (response.cost?.cost_usd ?? 0),
        blockedCount: current.blockedCount + (response.moderation?.blocked ? 1 : 0),
      }));

      setState((current) => ({
        ...current,
        status: "success",
        response,
        error: null,
      }));
      setRunHistory((current) => [
        buildHistoryRecord(prompt, response, null),
        ...current,
      ].slice(0, RUN_HISTORY_LIMIT));
    } catch (error) {
      const message = error instanceof Error ? error.message : "Request failed.";
      setState((current) => ({
        ...current,
        status: "error",
        error: message,
      }));
      setRunHistory((current) => [
        buildHistoryRecord(prompt, null, message),
        ...current,
      ].slice(0, RUN_HISTORY_LIMIT));
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const prompt = state.prompt.trim();
    if (!prompt) {
      setState((current) => ({ ...current, status: "error", error: "Enter a prompt before running." }));
      return;
    }

    await runPrompt(prompt);
  }

  // Auto-run the default scenario once on first load, so the mission dashboard is
  // populated the instant the page opens instead of showing an empty composer —
  // the first three seconds are what a recruiter clicking a live link actually sees.
  useEffect(() => {
    void runPrompt(INITIAL_OPERATOR_STATE.prompt);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function handlePromptKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key !== "Enter" || !event.ctrlKey) {
      return;
    }

    event.preventDefault();
    event.currentTarget.form?.requestSubmit();
  }

  function clearFailedHistory() {
    setRunHistory((current) => current.filter((record) => record.status !== "error"));
  }

  async function copyToClipboard(text: string, key: string) {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedKey(key);
      setTimeout(() => setCopiedKey((current) => (current === key ? null : current)), 1500);
    } catch {
      // Clipboard access can be denied by the browser; failing silently is fine here — the
      // button simply won't show the "Copied" confirmation, nothing else breaks.
    }
  }

  const response = state.response;

  return (
    <div className="app-shell">
      <div className="app-shell__backdrop" aria-hidden="true" />
      <main className="console">
        <header className="hero">
          <div>
            <p className="hero__kicker">Aegis crisis decision copilot</p>
            <h1 className="hero__title">Turn disruption signals into decisions.</h1>
            <p className="hero__subtitle">
              Generate a structured mission brief with risk score, top alerts, recommended actions,
              and supporting evidence while preserving full routing and telemetry visibility.
            </p>
            {sessionStats.requests > 0 ? (
              <div className="session-stats">
                <span>
                  <strong>{sessionStats.requests}</strong> request{sessionStats.requests === 1 ? "" : "s"}
                </span>
                <span>
                  <strong>${sessionStats.totalCostUsd.toFixed(6)}</strong> session cost
                </span>
                {sessionStats.blockedCount > 0 ? (
                  <span className="session-stats__flag">
                    <strong>{sessionStats.blockedCount}</strong> blocked by safety policy
                  </span>
                ) : null}
              </div>
            ) : null}
          </div>

          <div className="hero__meta">
            <span className="badge">Backend target</span>
            <input
              className="endpoint-input"
              value={backendUrl}
              onChange={(event) => setBackendUrl(event.target.value)}
              spellCheck={false}
              aria-label="Backend base URL"
            />
            <div className="health-row">
              <span className={`health-pill health-pill--${backendHealth.status}`}>
                {backendHealth.label}
              </span>
              <button
                type="button"
                className="health-button"
                onClick={() => {
                  void refreshBackendHealth();
                }}
              >
                Recheck
              </button>
            </div>
          </div>
        </header>

        <div className="scenario-chips">
          {SCENARIO_CHIPS.map((chip) => (
            <button
              key={chip.label}
              type="button"
              className="scenario-chip"
              disabled={state.status === "loading"}
              onClick={() => {
                setState((current) => ({ ...current, prompt: chip.prompt }));
                void runPrompt(chip.prompt);
              }}
            >
              {chip.label}
            </button>
          ))}
        </div>

        <form className="composer" onSubmit={handleSubmit}>
          <label className="composer__label" htmlFor="prompt">
            Task prompt
          </label>
          <textarea
            id="prompt"
            className="composer__input"
            value={state.prompt}
            onChange={(event) => setState((current) => ({ ...current, prompt: event.target.value }))}
            onKeyDown={handlePromptKeyDown}
            rows={5}
            placeholder="Ask the system to research, summarize, compare, or route a task."
          />

          <div className="agent-picker">
            <span className="agent-picker__label">Agent</span>
            <div className="agent-picker__options" role="radiogroup" aria-label="Agent selection">
              {(
                [
                  { value: "auto", label: "Auto-route" },
                  { value: "research", label: "Research" },
                  { value: "analyst", label: "Analyst" },
                  { value: "coder", label: "Coder" },
                  { value: "document", label: "Document" },
                  { value: "planner", label: "Planner" },
                ] as const
              ).map((option) => (
                <button
                  key={option.value}
                  type="button"
                  role="radio"
                  aria-checked={agentOverride === option.value}
                  className={`agent-picker__option${
                    agentOverride === option.value ? " agent-picker__option--active" : ""
                  }`}
                  onClick={() => setAgentOverride(option.value)}
                >
                  {option.label}
                </button>
              ))}
            </div>
            <p className="agent-picker__hint">
              {agentOverride === "auto"
                ? "The backend's agent_router picks based on task keywords."
                : `Forces routing.agent = "${agentOverride}" for this request (ChatRequest.agent_hint).`}
            </p>
          </div>

          <div className="feature-toggles">
            <label className="feature-toggle">
              <input
                type="checkbox"
                checked={orchestrate}
                onChange={(event) => setOrchestrate(event.target.checked)}
              />
              <span>
                Orchestrate <span className="feature-toggle__hint">Planner decomposes the task and dispatches sub-steps to other agents</span>
              </span>
            </label>

            <label className="feature-toggle">
              <input
                type="checkbox"
                checked={conversationMode}
                onChange={(event) => {
                  const checked = event.target.checked;
                  setConversationMode(checked);
                  if (!checked) {
                    setConversationId(null);
                    setTurnCount(0);
                    setTranscript([]);
                  }
                }}
              />
              <span>
                Continue conversation{" "}
                <span className="feature-toggle__hint">
                  {conversationMode
                    ? `Turn ${turnCount + 1} — the backend recalls prior turns via conversation_id`
                    : "Off — each request is single-shot with no memory"}
                </span>
              </span>
            </label>

            {conversationMode && conversationId ? (
              <button
                type="button"
                className="scenario-chip"
                onClick={() => {
                  setConversationId(null);
                  setTurnCount(0);
                  setTranscript([]);
                }}
              >
                New conversation
              </button>
            ) : null}
          </div>

          <div className="composer__footer">
            <p className="composer__hint">Press Enter to submit from the button, or use Ctrl+Enter for a quick run.</p>
            <button className="run-button" type="submit" disabled={!canRun}>
              {state.status === "loading" ? "Running…" : "Run request"}
            </button>
          </div>
        </form>

        {state.status === "error" && state.error ? (
          <section className="state-banner state-banner--error" role="alert">
            <strong>Request failed</strong>
            <p>{state.error}</p>
          </section>
        ) : null}

        {traceEvents.length > 0 ? (
          <Panel
            eyebrow="Live execution trace"
            title={state.status === "loading" ? "Pipeline running…" : "Pipeline trace"}
            description="Real backend checkpoints streamed live via Server-Sent Events, not simulated."
          >
            <ExecutionTrace events={traceEvents} isLive={state.status === "loading"} />
          </Panel>
        ) : null}

        {conversationMode && transcript.length > 0 ? (
          <Panel
            eyebrow="Conversation"
            title={`Thread — turn ${turnCount}`}
            description="The backend actually recalls these prior turns, not just the UI showing them."
          >
            <div className="chat-thread">
              {transcript.map((message, index) => (
                <div key={index} className={`chat-bubble chat-bubble--${message.role}`}>
                  <span className="chat-bubble__role">
                    {message.role === "user" ? "You" : message.agent ?? "assistant"}
                  </span>
                  <p>{message.content}</p>
                </div>
              ))}
            </div>
          </Panel>
        ) : null}

        <section className="grid">
          <div className="grid__main">
            <Panel
              eyebrow="Mission Brief"
              title="Crisis decision dashboard"
              description="Structured analysis extracted from the latest backend response."
            >
              {missionBrief && response ? (
                <div className="decision-stack">
                  <div className="mission-score">
                    <RiskGauge score={missionBrief.riskScore} level={missionBrief.riskLevel} />
                    <div className="mission-score__body">
                      <p className="mission-score__label">Risk score</p>
                      <p className="mission-score__summary">{missionBrief.summary}</p>
                      <div className="mission-score__badges">
                        <span className={`risk-pill risk-pill--${missionBrief.riskLevel}`}>
                          {missionBrief.riskLevel} risk
                        </span>
                        {response?.evaluation ? (
                          <span
                            className="quality-pill"
                            title="Deterministic quality score: how well-formed the brief is, and whether the response actually used retrieved context."
                          >
                            {Math.round(response.evaluation.overall_score * 100)}% quality
                          </span>
                        ) : null}
                        {response?.cost ? (
                          <span
                            className="cost-pill"
                            title={`${response.cost.input_tokens} input + ${response.cost.output_tokens} output tokens on ${response.cost.model}`}
                          >
                            ~${response.cost.cost_usd.toFixed(6)}
                          </span>
                        ) : null}
                        {response?.moderation?.pii_flags && response.moderation.pii_flags.length > 0 ? (
                          <span className="pii-pill" title="Response text matched a PII-shaped pattern (e.g. email/phone) — flagged for review, not redacted.">
                            ⚠ PII flagged: {response.moderation.pii_flags.join(", ")}
                          </span>
                        ) : null}
                      </div>
                    </div>
                  </div>

                  <div className="decision-grid">
                    <article className="decision-card">
                      <div className="decision-card__header">
                        <h3>Top alerts</h3>
                        <button
                          type="button"
                          className="copy-btn"
                          onClick={() => copyToClipboard(missionBrief.topAlerts.join("\n"), "alerts")}
                        >
                          {copiedKey === "alerts" ? "Copied" : "Copy"}
                        </button>
                      </div>
                      <ul className="decision-list">
                        {missionBrief.topAlerts.map((alert, index) => (
                          <li key={`alert-${index}`}>{alert}</li>
                        ))}
                      </ul>
                    </article>

                    <article className="decision-card">
                      <div className="decision-card__header">
                        <h3>Recommended actions</h3>
                        <button
                          type="button"
                          className="copy-btn"
                          onClick={() => copyToClipboard(missionBrief.recommendedActions.join("\n"), "actions")}
                        >
                          {copiedKey === "actions" ? "Copied" : "Copy"}
                        </button>
                      </div>
                      <ol className="decision-list decision-list--ordered">
                        {missionBrief.recommendedActions.map((action, index) => (
                          <li key={`action-${index}`}>{action}</li>
                        ))}
                      </ol>
                    </article>

                    <article className="decision-card">
                      <div className="decision-card__header">
                        <h3>Evidence</h3>
                        <button
                          type="button"
                          className="copy-btn"
                          onClick={() => copyToClipboard(missionBrief.evidence.join("\n"), "evidence")}
                        >
                          {copiedKey === "evidence" ? "Copied" : "Copy"}
                        </button>
                      </div>
                      <ul className="decision-list">
                        {missionBrief.evidence.map((item, index) => (
                          <li key={`evidence-${index}`}>{item}</li>
                        ))}
                      </ul>
                    </article>
                  </div>

                  <details className="raw-output">
                    <summary>Full model output</summary>
                    <button
                      type="button"
                      className="copy-btn copy-btn--floating"
                      onClick={(event) => {
                        event.preventDefault();
                        copyToClipboard(response.content, "raw-output");
                      }}
                    >
                      {copiedKey === "raw-output" ? "Copied" : "Copy"}
                    </button>
                    <pre className="response-copy">{response.content}</pre>
                  </details>

                  {response.sub_results && response.sub_results.length > 0 ? (
                    <div className="sub-results">
                      <h3>Sub-agent results ({response.sub_results.length})</h3>
                      {response.sub_results.map((sub, index) => (
                        <details key={index} className="sub-results__item">
                          <summary>
                            Sub-task {index + 1} — handled by <strong>{sub.routing.agent}</strong>
                            {sub.mission_brief ? ` · ${sub.mission_brief.risk_level} risk` : ""}
                          </summary>
                          <pre className="response-copy">{sub.content}</pre>
                        </details>
                      ))}
                    </div>
                  ) : null}
                </div>
              ) : response?.moderation?.blocked ? (
                <div className="moderation-blocked">
                  <strong>Request blocked by safety policy</strong>
                  <p>{response.moderation.block_reason}</p>
                  <p className="muted-copy">No LLM call was made — this was screened before reaching the model.</p>
                </div>
              ) : (
                <div className="empty-state">
                  <p>No mission brief yet.</p>
                  <span>Run a prompt to generate risk, alerts, actions, and evidence cards.</span>
                </div>
              )}
            </Panel>
          </div>

          <div className="grid__aside">
            <Panel
              eyebrow="Routing"
              title="Agent and model metadata"
              description="The backend decides this; the UI only renders it."
            >
              {response ? (
                <dl className="detail-list">
                  <div>
                    <dt>Selected agent</dt>
                    <dd>{response.routing.agent}</dd>
                  </div>
                  <div>
                    <dt>Model</dt>
                    <dd>{response.routing.model}</dd>
                  </div>
                  <div>
                    <dt>Provider</dt>
                    <dd>{response.routing.provider}</dd>
                  </div>
                  <div>
                    <dt>Confidence</dt>
                    <dd>{formatConfidence(response.routing.confidence)}</dd>
                  </div>
                  <div>
                    <dt>Preference</dt>
                    <dd>{response.routing.preference ?? "balanced"}</dd>
                  </div>
                  <div>
                    <dt>Reason</dt>
                    <dd>{response.routing.reason}</dd>
                  </div>
                </dl>
              ) : (
                <div className="empty-state">
                  <p>Routing metadata will appear here.</p>
                </div>
              )}
            </Panel>
          </div>

          <div className="grid__full">
            <Panel
              eyebrow="Runs"
              title="Recent request history"
              description="A lightweight local log for quick operator comparisons."
              actions={
                <button
                  type="button"
                  className="panel-action-button"
                  onClick={clearFailedHistory}
                  disabled={!hasFailedHistory}
                >
                  Clear failed entries
                </button>
              }
            >
              {runHistory.length > 0 ? (
                <ul className="history-list">
                  {runHistory.map((record) => (
                    <li key={record.id} className="history-list__item">
                      <div className="history-list__top">
                        <span className={`history-status history-status--${record.status}`}>{record.status}</span>
                        <span className="history-time">{new Date(record.createdAt).toLocaleTimeString()}</span>
                      </div>
                      <p className="history-prompt">{record.prompt}</p>
                      {record.status === "success" ? (
                        <p className="history-meta">
                          {record.agent} · {record.model} · {record.provider}
                          {record.latencyMs != null ? ` · ${Math.round(record.latencyMs)} ms` : ""}
                        </p>
                      ) : (
                        <p className="history-error">{record.error}</p>
                      )}
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="empty-state">
                  <p>No runs recorded yet.</p>
                  <span>Each request is listed here after execution.</span>
                </div>
              )}
            </Panel>
          </div>

          <div className="grid__full">
            <Panel
              eyebrow="Telemetry"
              title="Event and tool details"
              description="Shows request telemetry and any tool payloads if the backend includes them."
            >
              <div className="telemetry-stack">
                <div className="telemetry-summary">{getTelemetrySummary(response)}</div>
                <div className="telemetry-grid">
                  <div>
                    <span className="telemetry-label">Request ID</span>
                    <strong>{response?.telemetry?.request_id ?? "n/a"}</strong>
                  </div>
                  <div>
                    <span className="telemetry-label">Input tokens</span>
                    <strong>{response?.telemetry?.input_tokens ?? "n/a"}</strong>
                  </div>
                  <div>
                    <span className="telemetry-label">Output tokens</span>
                    <strong>{response?.telemetry?.output_tokens ?? "n/a"}</strong>
                  </div>
                  <div>
                    <span className="telemetry-label">Latency</span>
                    <strong>{response?.telemetry?.latency_ms != null ? `${Math.round(response.telemetry.latency_ms)} ms` : "n/a"}</strong>
                  </div>
                </div>

                <div className="tool-results">
                  <div className="tool-results__header">
                    <h3>Response evaluation</h3>
                    <span>{response?.evaluation ? "deterministic + optional judge" : "none returned"}</span>
                  </div>
                  {response?.evaluation ? (
                    <div className="telemetry-grid">
                      <div>
                        <span className="telemetry-label">Overall score</span>
                        <strong>{Math.round(response.evaluation.overall_score * 100)}%</strong>
                      </div>
                      <div>
                        <span className="telemetry-label">Structure quality</span>
                        <strong>{Math.round(response.evaluation.structure_quality * 100)}%</strong>
                      </div>
                      <div>
                        <span className="telemetry-label">Groundedness</span>
                        <strong>
                          {response.evaluation.groundedness != null
                            ? `${Math.round(response.evaluation.groundedness * 100)}%`
                            : "n/a (no RAG context)"}
                        </strong>
                      </div>
                      <div>
                        <span className="telemetry-label">LLM judge score</span>
                        <strong>
                          {response.evaluation.judge_score != null
                            ? `${Math.round(response.evaluation.judge_score * 100)}%`
                            : "not requested"}
                        </strong>
                      </div>
                      {response.evaluation.notes.length > 0 ? (
                        <div className="telemetry-notes">
                          {response.evaluation.notes.map((note, index) => (
                            <p key={index} className="muted-copy">{note}</p>
                          ))}
                        </div>
                      ) : null}
                    </div>
                  ) : (
                    <p className="muted-copy">No evaluation returned for this response.</p>
                  )}
                </div>

                <div className="tool-results">
                  <div className="tool-results__header">
                    <h3>Tool results</h3>
                    <span>{toolResults.length > 0 ? `${toolResults.length} item(s)` : "none returned"}</span>
                  </div>
                  {toolResults.length > 0 ? (
                    <div className="tool-results__list">
                      {toolResults.map((result, index) => (
                        <pre key={index} className="json-block">{formatJson(result)}</pre>
                      ))}
                    </div>
                  ) : (
                    <p className="muted-copy">
                      The current chat endpoint does not return tool results, but this panel is ready if
                      the backend starts including them.
                    </p>
                  )}
                </div>
              </div>
            </Panel>
          </div>
        </section>
      </main>
    </div>
  );
}