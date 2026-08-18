import { KeyboardEvent, FormEvent, useEffect, useRef, useState } from "react";

import { login as apiLogin, register as apiRegister, type UserPayload } from "./api/authClient";
import { runChatStream, type TraceStageEvent } from "./api/chatStreamClient";
import { checkBackendHealth } from "./api/healthClient";
import { fetchPersistedRuns, fetchRunDetail, type RunDetailPayload } from "./api/runsClient";
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
  icon: string;
  description: string;
  exampleQuestion: string;
  prompt: string;
}

const SCENARIO_CHIPS: ScenarioChip[] = [
  {
    label: "Supply Chain Risk",
    icon: "📦",
    description: "Assess operational disruption risk and identify major contributing factors.",
    exampleQuestion: "A key supplier just went dark after a port disruption — what's our exposure?",
    prompt: DEFAULT_PROMPT,
  },
  {
    label: "Cybersecurity",
    icon: "🛡️",
    description: "Analyze a potential security incident and recommend immediate actions.",
    exampleQuestion: "Our customer database may have been breached — how bad is it?",
    prompt:
      "Our customer database was exposed in a suspected breach discovered 2 hours ago. " +
      "Analyze the risk and recommend an immediate response.",
  },
  {
    label: "Market Intelligence",
    icon: "📈",
    description: "Investigate market signals and identify key risks and opportunities.",
    exampleQuestion: "A competitor just went bankrupt and our sector dropped 15% — do we need to react?",
    prompt:
      "A major competitor just filed for bankruptcy, triggering a 15% drop in our sector's stock prices overnight. " +
      "Research the market implications and recommend how we should respond this week.",
  },
  {
    label: "Disaster Response",
    icon: "🌀",
    description: "Analyze a developing situation and prioritize response actions.",
    exampleQuestion: "A hurricane is 48 hours from our main distribution center — what should we do first?",
    prompt:
      "A category 4 hurricane is projected to make landfall near our primary distribution center in 48 hours. " +
      "Analyze the operational risk and recommend an immediate response.",
  },
];

type AgentKey = "research" | "analyst" | "coder" | "document" | "planner";

interface AgentInfo {
  key: AgentKey;
  label: string;
  tagline: string;
  description: string;
  usefulWhen: string;
}

const AGENT_INFO: AgentInfo[] = [
  {
    key: "research",
    label: "Research",
    tagline: "Finds relevant information",
    description: "Retrieves evidence from the knowledge base and available research sources.",
    usefulWhen: "You need to gather facts, context, or background before deciding anything.",
  },
  {
    key: "analyst",
    label: "Analyst",
    tagline: "Turns evidence into analysis",
    description: "Identifies patterns, risks, findings, and implications, with quantified reasoning.",
    usefulWhen: "You need risk scoring, comparisons, or a quantified impact assessment.",
  },
  {
    key: "planner",
    label: "Planner",
    tagline: "Coordinates complex missions",
    description: "Breaks a mission into sub-tasks and delegates each one to the specialized agent that fits it.",
    usefulWhen: "The mission has multiple distinct parts that benefit from different agents.",
  },
  {
    key: "coder",
    label: "Coder",
    tagline: "Performs computational analysis",
    description: "Used when calculations, debugging, or code-based analysis are useful to the mission.",
    usefulWhen: "The mission involves a technical implementation, bug, or computational question.",
  },
  {
    key: "document",
    label: "Document",
    tagline: "Extracts and structures findings",
    description: "Interprets documents, contracts, or policies precisely, flagging ambiguity instead of guessing.",
    usefulWhen: "The mission involves interpreting a document, contract, or policy.",
  },
];

const ARCHITECTURE_STEPS = [
  { title: "Understand", detail: "Aegis parses the mission and screens it for safety before anything else runs." },
  { title: "Plan", detail: "The router picks the right agent — or the Planner decomposes a multi-part mission." },
  { title: "Research", detail: "Relevant evidence is retrieved from a grounded knowledge base, when applicable." },
  { title: "Analyze", detail: "The selected agent reasons over the evidence and drafts findings." },
  { title: "Verify", detail: "The response is evaluated for groundedness and structure before being returned." },
  { title: "Deliver", detail: "A structured intelligence brief is built, costed, and saved to your mission history." },
];

const TRUST_ITEMS = [
  "Evidence-backed analysis, grounded in a real retrieval index",
  "Safety screening before any request reaches the model",
  "Deterministic response quality evaluation on every mission",
  "Full agent execution visibility, not a black box",
  "Source transparency — evidence links back to real documents",
  "Real cost transparency, per mission and per agent",
];

const ENGINEERING_HIGHLIGHTS = [
  "5 specialized agents with real routing",
  "Genuine multi-agent orchestration (Planner → sub-agents)",
  "Grounded RAG with a relevance threshold",
  "Live SSE execution streaming",
  "Per-agent duration/token/cost observability",
  "Real cost tracking, not estimated after the fact",
  "Multi-turn conversation memory",
  "Prompt-injection protection",
  "PII detection",
  "Deterministic + optional LLM-judge evaluation",
  "JWT authentication with cross-user isolation",
  "101 backend tests, 10+ frontend tests, CI on every push",
];

const MISSION_TIP_EXAMPLES = {
  bad: "Supply chain?",
  good:
    "Assess the risk of semiconductor supply disruption over the next 30 days and identify the " +
    "three biggest contributing factors.",
};

/** Friendly, non-technical explanation shown when a request is blocked by the safety layer.
 * Deliberately does not surface the backend's raw block_reason (which currently includes the
 * matched regex pattern) — that's an implementation detail, not something a user needs to see. */
export function friendlyModerationMessage(): { reason: string; whatHappened: string; whatToDo: string } {
  return {
    reason: "Prompt injection detected.",
    whatHappened:
      "The request appeared to attempt to override Aegis's internal instructions or behavior, " +
      "so it was stopped before reaching any model.",
    whatToDo:
      "Try asking Aegis to perform the task directly, without language intended to override its " +
      "instructions or reveal internal behavior.",
  };
}

interface RagChunk {
  id: string;
  text: string;
  score: number;
  source: string;
}

/** Extracts real retrieved RAG chunks from tool_results, if the backend included any — used to
 * render evidence as source-attributed cards instead of only plain extracted sentences. */
export function extractRagChunks(toolResults: Array<Record<string, unknown>>): RagChunk[] {
  const ragResult = toolResults.find((result) => result.tool_type === "rag");
  const metadata = ragResult?.metadata as Record<string, unknown> | undefined;
  const chunks = metadata?.chunks as Array<Record<string, unknown>> | undefined;
  if (!chunks) {
    return [];
  }

  return chunks.map((chunk) => {
    const chunkMetadata = chunk.metadata as Record<string, unknown> | undefined;
    return {
      id: String(chunk.id ?? ""),
      text: String(chunk.text ?? ""),
      score: Number(chunk.score ?? 0),
      source: String(chunkMetadata?.source ?? "unknown source"),
    };
  });
}

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

export function formatConfidence(value: number): string {
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

export function buildHistoryRecord(
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
    runId: response.run_id ?? null,
    createdAt: now.toISOString(),
    prompt,
    status: "success",
    agent: response.routing.agent,
    model: response.routing.model,
    provider: response.routing.provider,
    latencyMs: response.telemetry?.latency_ms ?? undefined,
    riskLevel: response.mission_brief?.risk_level,
    riskScore: response.mission_brief?.risk_score,
    qualityScore: response.evaluation?.overall_score,
    costUsd: response.cost?.cost_usd,
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
      <ol className="execution-trace" aria-live="polite" aria-label="Mission execution progress">
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

const ONBOARDING_STEPS = [
  {
    title: "1. Describe your mission",
    detail: "Tell Aegis what you want investigated — a situation, a risk, a decision you need evidence for.",
  },
  {
    title: "2. Aegis plans the work",
    detail: "The Planner determines which agents and steps are needed for the mission.",
  },
  {
    title: "3. Agents investigate",
    detail: "Research, Analyst, Coder, and Document agents perform specialized work.",
  },
  { title: "4. Evidence is retrieved", detail: "Aegis searches relevant knowledge sources for supporting evidence." },
  {
    title: "5. The result is evaluated",
    detail: "Safety and quality checks run before the final brief is returned.",
  },
  {
    title: "6. You receive the intelligence brief",
    detail: "With risk, evidence, recommendations, metrics, and full execution details.",
  },
];

const ONBOARDING_STORAGE_KEY = "aegis_onboarding_seen";

function OnboardingModal({ onTryDemo, onExplore, onSkip }: { onTryDemo: () => void; onExplore: () => void; onSkip: () => void }) {
  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent | globalThis.KeyboardEvent) {
      if (event.key === "Escape") {
        onSkip();
      }
    }
    window.addEventListener("keydown", handleKeyDown as EventListener);
    return () => window.removeEventListener("keydown", handleKeyDown as EventListener);
  }, [onSkip]);

  return (
    <div className="onboarding-overlay" role="presentation" onClick={onSkip}>
      <div
        className="onboarding-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="onboarding-title"
        onClick={(event) => event.stopPropagation()}
      >
        <h2 id="onboarding-title">Welcome to Aegis</h2>
        <p className="onboarding-modal__intro">
          Aegis turns complex questions into structured intelligence by coordinating specialized AI
          agents, retrieving relevant evidence, and evaluating the result.
        </p>
        <ol className="onboarding-steps">
          {ONBOARDING_STEPS.map((step) => (
            <li key={step.title}>
              <strong>{step.title}</strong>
              <span>{step.detail}</span>
            </li>
          ))}
        </ol>
        <div className="onboarding-modal__actions">
          <button type="button" className="run-button" onClick={onTryDemo}>
            Try a Demo
          </button>
          <button type="button" className="scenario-chip" onClick={onExplore}>
            Explore Aegis
          </button>
          <button type="button" className="onboarding-skip" onClick={onSkip}>
            Skip
          </button>
        </div>
      </div>
    </div>
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
  const [authToken, setAuthToken] = useState<string | null>(() =>
    localStorage.getItem("aegis_auth_token"),
  );
  const [authUser, setAuthUser] = useState<UserPayload | null>(() => {
    const raw = localStorage.getItem("aegis_auth_user");
    return raw ? (JSON.parse(raw) as UserPayload) : null;
  });
  const [authMode, setAuthMode] = useState<"login" | "register">("login");
  const [authEmail, setAuthEmail] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authError, setAuthError] = useState<string | null>(null);
  const [authBusy, setAuthBusy] = useState(false);
  const [missionDetail, setMissionDetail] = useState<RunDetailPayload | null>(null);
  const [missionDetailLoading, setMissionDetailLoading] = useState<string | null>(null);
  const [showOnboarding, setShowOnboarding] = useState(
    () => typeof window !== "undefined" && !localStorage.getItem(ONBOARDING_STORAGE_KEY),
  );
  const [viewMode, setViewMode] = useState<"simple" | "advanced">("simple");
  const [howItWorksOpen, setHowItWorksOpen] = useState(false);
  const [missionTipsOpen, setMissionTipsOpen] = useState(false);
  const composerRef = useRef<HTMLFormElement | null>(null);
  const howItWorksRef = useRef<HTMLDivElement | null>(null);
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

    if (authToken) {
      fetchPersistedRuns({ baseUrl: backendUrl }, authToken, controller.signal)
        .then((records) => {
          setRunHistory(
            records.slice(0, RUN_HISTORY_LIMIT).map((record) => ({
              id: record.id,
              runId: record.id,
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
    }

    return () => {
      controller.abort();
    };
  }, [backendUrl, authToken]);

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
        authToken,
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

  function persistAuth(token: string, user: UserPayload) {
    localStorage.setItem("aegis_auth_token", token);
    localStorage.setItem("aegis_auth_user", JSON.stringify(user));
    setAuthToken(token);
    setAuthUser(user);
  }

  async function handleAuthSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setAuthError(null);
    setAuthBusy(true);

    try {
      const result =
        authMode === "login"
          ? await apiLogin({ baseUrl: backendUrl }, authEmail, authPassword)
          : await apiRegister({ baseUrl: backendUrl }, authEmail, authPassword);
      persistAuth(result.access_token, result.user);
      setAuthPassword("");
    } catch (error) {
      setAuthError(error instanceof Error ? error.message : "Authentication failed.");
    } finally {
      setAuthBusy(false);
    }
  }

  function handleLogout() {
    localStorage.removeItem("aegis_auth_token");
    localStorage.removeItem("aegis_auth_user");
    setAuthToken(null);
    setAuthUser(null);
    setMissionDetail(null);
  }

  async function loadMissionDetail(runId: string) {
    if (!authToken) {
      return;
    }
    setMissionDetailLoading(runId);
    try {
      const detail = await fetchRunDetail({ baseUrl: backendUrl }, runId, authToken);
      setMissionDetail(detail);
    } catch {
      setMissionDetail(null);
    } finally {
      setMissionDetailLoading(null);
    }
  }

  function dismissOnboarding() {
    localStorage.setItem(ONBOARDING_STORAGE_KEY, "1");
    setShowOnboarding(false);
  }

  function handleOnboardingTryDemo() {
    dismissOnboarding();
    composerRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function handleOnboardingExplore() {
    dismissOnboarding();
    setHowItWorksOpen(true);
    howItWorksRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  const response = state.response;
  const ragChunks = extractRagChunks(response?.tool_results ?? []);

  return (
    <div className="app-shell">
      <div className="app-shell__backdrop" aria-hidden="true" />
      {showOnboarding ? (
        <OnboardingModal
          onTryDemo={handleOnboardingTryDemo}
          onExplore={handleOnboardingExplore}
          onSkip={dismissOnboarding}
        />
      ) : null}
      <main className="console">
        <header className="hero">
          <div>
            <p className="hero__kicker">Aegis</p>
            <h1 className="hero__title">Multi-Agent Intelligence for Complex Decisions</h1>
            <p className="hero__subtitle">
              Aegis coordinates specialized AI agents to investigate complex questions, ground findings
              in evidence, evaluate results, and produce structured intelligence briefs.
            </p>
            <div className="hero__cta">
              <button
                type="button"
                className="run-button"
                onClick={() => composerRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })}
              >
                Run a Mission
              </button>
              <button
                type="button"
                className="scenario-chip"
                onClick={() => {
                  setHowItWorksOpen(true);
                  howItWorksRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
                }}
              >
                How Aegis Works
              </button>
            </div>
            {sessionStats.requests > 0 ? (
              <div className="session-stats">
                <span title="Missions processed during this session.">
                  <strong>{sessionStats.requests}</strong> request{sessionStats.requests === 1 ? "" : "s"}
                </span>
                <span title="Cumulative estimated model/API cost for this session.">
                  <strong>${sessionStats.totalCostUsd.toFixed(6)}</strong> session cost
                </span>
                {sessionStats.blockedCount > 0 ? (
                  <span className="session-stats__flag" title="Requests prevented by the safety layer.">
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

          <div className="auth-widget">
            {authUser ? (
              <div className="auth-widget__signed-in">
                <span>
                  Signed in as <strong>{authUser.email}</strong>
                </span>
                <button type="button" className="scenario-chip" onClick={handleLogout}>
                  Log out
                </button>
              </div>
            ) : (
              <form className="auth-widget__form" onSubmit={handleAuthSubmit}>
                <div className="auth-widget__tabs">
                  <button
                    type="button"
                    className={`auth-widget__tab${authMode === "login" ? " auth-widget__tab--active" : ""}`}
                    onClick={() => setAuthMode("login")}
                  >
                    Sign in
                  </button>
                  <button
                    type="button"
                    className={`auth-widget__tab${authMode === "register" ? " auth-widget__tab--active" : ""}`}
                    onClick={() => setAuthMode("register")}
                  >
                    Register
                  </button>
                </div>
                <input
                  type="email"
                  required
                  placeholder="email"
                  className="endpoint-input"
                  value={authEmail}
                  onChange={(event) => setAuthEmail(event.target.value)}
                />
                <input
                  type="password"
                  required
                  minLength={8}
                  placeholder="password (min 8 chars)"
                  className="endpoint-input"
                  value={authPassword}
                  onChange={(event) => setAuthPassword(event.target.value)}
                />
                <button type="submit" className="scenario-chip" disabled={authBusy}>
                  {authBusy ? "…" : authMode === "login" ? "Sign in" : "Create account"}
                </button>
                {authError ? <span className="auth-widget__error">{authError}</span> : null}
              </form>
            )}
            <p className="auth-widget__hint">
              Anonymous demo works fully without an account — sign in to get a personal, isolated
              mission history with per-agent cost/token observability.
            </p>
          </div>
        </header>

        <div ref={howItWorksRef}>
          <Panel
            eyebrow="How Aegis works"
            title="Mission → Agents → Evidence → Analysis → Evaluation → Intelligence"
            description="The same six stages run on every mission — you can watch them happen live in the execution trace below."
            actions={
              <button type="button" className="panel-action-button" onClick={() => setHowItWorksOpen((v) => !v)}>
                {howItWorksOpen ? "Hide" : "Show"}
              </button>
            }
          >
            {howItWorksOpen ? (
              <ol className="how-it-works">
                {ARCHITECTURE_STEPS.map((step, index) => (
                  <li key={step.title}>
                    <span className="how-it-works__index">{index + 1}</span>
                    <div>
                      <strong>{step.title}</strong>
                      <p>{step.detail}</p>
                    </div>
                  </li>
                ))}
              </ol>
            ) : (
              <p className="muted-copy">Click "Show" to see how a mission moves through Aegis, step by step.</p>
            )}
          </Panel>
        </div>

        <div className="scenario-cards">
          {SCENARIO_CHIPS.map((chip) => (
            <article key={chip.label} className="scenario-card">
              <div className="scenario-card__icon" aria-hidden="true">
                {chip.icon}
              </div>
              <h3>{chip.label}</h3>
              <p className="scenario-card__description">{chip.description}</p>
              <p className="scenario-card__example">"{chip.exampleQuestion}"</p>
              <button
                type="button"
                className="scenario-card__try"
                disabled={state.status === "loading"}
                onClick={() => {
                  setState((current) => ({ ...current, prompt: chip.prompt }));
                  void runPrompt(chip.prompt);
                }}
              >
                Try this
              </button>
            </article>
          ))}
        </div>

        <form className="composer" onSubmit={handleSubmit} ref={composerRef}>
          <div className="composer__label-row">
            <label className="composer__label" htmlFor="prompt">
              Describe your mission
            </label>
            <button
              type="button"
              className="composer__tips-toggle"
              onClick={() => setMissionTipsOpen((v) => !v)}
              aria-expanded={missionTipsOpen}
            >
              {missionTipsOpen ? "Hide tips" : "What should I ask?"}
            </button>
          </div>

          {missionTipsOpen ? (
            <div className="mission-tips">
              <p>Describe what you want investigated, the relevant timeframe, and the outcome you want.</p>
              <div className="mission-tips__examples">
                <p>
                  <span className="mission-tips__bad">Bad:</span> "{MISSION_TIP_EXAMPLES.bad}"
                </p>
                <p>
                  <span className="mission-tips__good">Good:</span> "{MISSION_TIP_EXAMPLES.good}"
                </p>
              </div>
              <ul>
                <li>Be specific</li>
                <li>Include a timeframe when relevant</li>
                <li>Mention the company, event, market, or domain</li>
                <li>Ask for a specific outcome</li>
              </ul>
            </div>
          ) : null}

          <textarea
            id="prompt"
            className="composer__input"
            value={state.prompt}
            onChange={(event) => setState((current) => ({ ...current, prompt: event.target.value }))}
            onKeyDown={handlePromptKeyDown}
            rows={5}
            placeholder="Describe the situation you want Aegis to investigate."
          />

          <div className="agent-picker">
            <span className="agent-picker__label">Agent</span>
            <div className="agent-picker__options" role="radiogroup" aria-label="Agent selection">
              <button
                type="button"
                role="radio"
                aria-checked={agentOverride === "auto"}
                title="The backend's agent_router picks based on task keywords."
                className={`agent-picker__option${agentOverride === "auto" ? " agent-picker__option--active" : ""}`}
                onClick={() => setAgentOverride("auto")}
              >
                Auto-route
              </button>
              {AGENT_INFO.map((agent) => (
                <button
                  key={agent.key}
                  type="button"
                  role="radio"
                  aria-checked={agentOverride === agent.key}
                  title={`${agent.tagline}. ${agent.description}`}
                  className={`agent-picker__option${
                    agentOverride === agent.key ? " agent-picker__option--active" : ""
                  }`}
                  onClick={() => setAgentOverride(agent.key)}
                >
                  {agent.label}
                </button>
              ))}
            </div>
            <p className="agent-picker__hint">
              {agentOverride === "auto"
                ? "Aegis picks the right agent automatically based on your mission."
                : (() => {
                    const info = AGENT_INFO.find((a) => a.key === agentOverride);
                    return info ? `${info.tagline} — ${info.usefulWhen}` : "";
                  })()}
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
                    ? "Aegis can use relevant context from earlier messages in this conversation."
                    : "Start a fresh independent mission."}
                </span>
              </span>
            </label>

            {conversationMode && conversationId ? (
              <span className="conversation-active-pill" title={`Turn ${turnCount + 1} — context is being recalled from this conversation.`}>
                ● Conversation context active
              </span>
            ) : null}

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
            <strong>Aegis couldn't complete this mission</strong>
            <p>Something went wrong while processing your request.</p>
            <div className="state-banner__actions">
              <button type="button" className="scenario-chip" onClick={() => void runPrompt(state.prompt)}>
                Try again
              </button>
              <button
                type="button"
                className="scenario-chip"
                onClick={() => setState((current) => ({ ...current, status: "idle", error: null }))}
              >
                Start a new mission
              </button>
            </div>
            <details className="error-technical">
              <summary>Technical details</summary>
              <pre className="response-copy">{state.error}</pre>
            </details>
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
                    {message.role === "user"
                      ? "You"
                      : message.agent
                      ? `Aegis · ${message.agent.charAt(0).toUpperCase()}${message.agent.slice(1)} Agent`
                      : "Aegis"}
                  </span>
                  <p>{message.content}</p>
                </div>
              ))}
            </div>
          </Panel>
        ) : null}

        {response ? (
          <div className="view-mode-toggle" role="radiogroup" aria-label="Detail level">
            <button
              type="button"
              role="radio"
              aria-checked={viewMode === "simple"}
              className={`view-mode-toggle__option${viewMode === "simple" ? " view-mode-toggle__option--active" : ""}`}
              onClick={() => setViewMode("simple")}
            >
              Simple View
            </button>
            <button
              type="button"
              role="radio"
              aria-checked={viewMode === "advanced"}
              className={`view-mode-toggle__option${viewMode === "advanced" ? " view-mode-toggle__option--active" : ""}`}
              onClick={() => setViewMode("advanced")}
            >
              Advanced View
            </button>
          </div>
        ) : null}

        <section className="grid">
          <div className="grid__main">
            <Panel
              eyebrow="Mission Brief"
              title="Intelligence Brief"
              description="Structured analysis extracted from the latest backend response."
            >
              {missionBrief && response ? (
                <div className="decision-stack">
                  <div className="mission-score">
                    <RiskGauge score={missionBrief.riskScore} level={missionBrief.riskLevel} />
                    <div className="mission-score__body">
                      <p className="mission-score__label" title="Estimated severity or likelihood of the risk identified by the analysis.">
                        Risk
                      </p>
                      <p className="mission-score__summary">{missionBrief.summary}</p>
                      <div className="mission-score__badges">
                        <span
                          className={`risk-pill risk-pill--${missionBrief.riskLevel}`}
                          title="Estimated severity or likelihood of the risk identified by the analysis."
                        >
                          {missionBrief.riskLevel} risk
                        </span>
                        <span
                          className="confidence-pill"
                          title="How strongly the agent's routing confidence and retrieved evidence support this assessment."
                        >
                          {formatConfidence(response.routing.confidence)} confidence
                        </span>
                        {response?.evaluation ? (
                          <span
                            className="quality-pill"
                            title="A structured evaluation of the response: how well-formed the brief is, and whether it actually used retrieved evidence."
                          >
                            {Math.round(response.evaluation.overall_score * 100)}% quality
                          </span>
                        ) : null}
                        {response?.cost ? (
                          <span
                            className="cost-pill"
                            title={`Estimated model/API cost for this mission — ${response.cost.input_tokens} input + ${response.cost.output_tokens} output tokens on ${response.cost.model}.`}
                          >
                            ~${response.cost.cost_usd.toFixed(6)}
                          </span>
                        ) : null}
                        {response?.moderation?.pii_flags && response.moderation.pii_flags.length > 0 ? (
                          <span className="pii-pill" title="Indicates whether potentially sensitive personal information (e.g. email/phone) was detected — flagged for review, not redacted.">
                            ⚠ PII flagged: {response.moderation.pii_flags.join(", ")}
                          </span>
                        ) : null}
                      </div>
                    </div>
                  </div>

                  <div className="decision-grid">
                    <article className="decision-card">
                      <div className="decision-card__header">
                        <h3>Key findings</h3>
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
                      {ragChunks.length > 0 ? (
                        <div className="evidence-cards">
                          {ragChunks.map((chunk) => (
                            <details key={chunk.id} className="evidence-card">
                              <summary>
                                <span className="evidence-card__source">{chunk.source.replace(/-/g, " ")}</span>
                                <span
                                  className="evidence-card__relevance"
                                  title="How closely this retrieved document matched the mission."
                                >
                                  {Math.round(chunk.score * 100)}% relevant
                                </span>
                              </summary>
                              <p className="evidence-card__excerpt">{chunk.text.trim().slice(0, 320)}</p>
                            </details>
                          ))}
                        </div>
                      ) : (
                        <ul className="decision-list">
                          {missionBrief.evidence.map((item, index) => (
                            <li key={`evidence-${index}`}>{item}</li>
                          ))}
                        </ul>
                      )}
                    </article>
                  </div>

                  {viewMode === "advanced" ? (
                    <>
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
                          <h3>Agent execution ({response.sub_results.length} sub-agents)</h3>
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
                    </>
                  ) : null}
                </div>
              ) : response?.moderation?.blocked ? (
                (() => {
                  const explanation = friendlyModerationMessage();
                  return (
                    <div className="moderation-blocked">
                      <strong>Request blocked</strong>
                      <p>
                        Aegis detected a request that could interfere with the safe operation of the
                        system.
                      </p>
                      <div className="moderation-blocked__section">
                        <h4>Reason</h4>
                        <p>{explanation.reason}</p>
                      </div>
                      <div className="moderation-blocked__section">
                        <h4>What happened?</h4>
                        <p>{explanation.whatHappened}</p>
                      </div>
                      <div className="moderation-blocked__section">
                        <h4>What can I do?</h4>
                        <p>{explanation.whatToDo}</p>
                      </div>
                    </div>
                  );
                })()
              ) : (
                <div className="empty-state">
                  <p>No mission brief yet.</p>
                  <span>Run a prompt to generate risk, alerts, actions, and evidence cards.</span>
                </div>
              )}
            </Panel>
          </div>

          {viewMode === "advanced" ? (
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
                      <dt title="How strongly the agent router's rule matched this task.">Confidence</dt>
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
          ) : null}

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
                        <>
                          <p className="history-meta">
                            {record.agent} · {record.model} · {record.provider}
                            {record.latencyMs != null ? ` · ${Math.round(record.latencyMs)} ms` : ""}
                          </p>
                          <div className="history-badges">
                            {record.riskLevel ? (
                              <span className={`risk-pill risk-pill--${record.riskLevel}`}>{record.riskLevel} risk</span>
                            ) : null}
                            {record.qualityScore != null ? (
                              <span className="quality-pill">{Math.round(record.qualityScore * 100)}% quality</span>
                            ) : null}
                            {record.costUsd != null ? (
                              <span className="cost-pill">~${record.costUsd.toFixed(6)}</span>
                            ) : null}
                          </div>
                        </>
                      ) : (
                        <p className="history-error">{record.error}</p>
                      )}
                      <div className="history-actions">
                        <button
                          type="button"
                          className="copy-btn"
                          onClick={() => copyToClipboard(record.prompt, `history-${record.id}`)}
                        >
                          {copiedKey === `history-${record.id}` ? "Copied" : "Copy prompt"}
                        </button>
                        {record.status === "success" ? (
                          <button
                            type="button"
                            className="copy-btn"
                            onClick={() => {
                              setState((current) => ({ ...current, prompt: record.prompt }));
                              composerRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
                            }}
                          >
                            Open
                          </button>
                        ) : null}
                        {authToken && record.runId ? (
                          <button
                            type="button"
                            className="copy-btn"
                            onClick={() => void loadMissionDetail(record.runId!)}
                          >
                            {missionDetailLoading === record.runId ? "Loading…" : "View agent steps"}
                          </button>
                        ) : null}
                      </div>
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

          {missionDetail ? (
            <div className="grid__full">
              <Panel
                eyebrow="Mission Observability"
                title={`Mission ${missionDetail.id.slice(0, 8)}`}
                description="Real per-agent timing, tokens, and cost — persisted server-side, not fabricated."
                actions={
                  <button type="button" className="panel-action-button" onClick={() => setMissionDetail(null)}>
                    Close
                  </button>
                }
              >
                <ol className="mission-steps">
                  {missionDetail.steps.map((step, index) => (
                    <li key={index} className="mission-steps__item">
                      <span className="mission-steps__agent">{step.agent}</span>
                      <span className="mission-steps__stat">
                        {step.status === "ok" ? "✓" : "✗"}{" "}
                        {step.duration_ms != null ? `${(step.duration_ms / 1000).toFixed(2)}s` : "—"}
                      </span>
                      <span className="mission-steps__stat">
                        {(step.input_tokens ?? 0) + (step.output_tokens ?? 0)} tokens
                      </span>
                      <span className="mission-steps__stat">
                        ${step.cost_usd != null ? step.cost_usd.toFixed(6) : "0.000000"}
                      </span>
                      <span className="mission-steps__model">
                        {step.provider}/{step.model}
                      </span>
                    </li>
                  ))}
                </ol>
                <div className="mission-steps__total">
                  <strong>Total</strong>
                  <span>
                    {missionDetail.total_duration_ms != null
                      ? `${(missionDetail.total_duration_ms / 1000).toFixed(2)}s`
                      : "—"}
                  </span>
                  <span>{missionDetail.total_tokens ?? 0} tokens</span>
                  <span>${missionDetail.total_cost_usd != null ? missionDetail.total_cost_usd.toFixed(6) : "0.000000"}</span>
                </div>
              </Panel>
            </div>
          ) : null}

          {viewMode === "advanced" ? (
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
          ) : null}

          <div className="grid__full">
            <Panel
              eyebrow="Why trust this result?"
              title="Transparency checklist"
              description="Only capabilities Aegis actually implements — nothing aspirational."
            >
              <ul className="trust-list">
                {TRUST_ITEMS.map((item) => (
                  <li key={item}>
                    <span aria-hidden="true">✓</span> {item}
                  </li>
                ))}
              </ul>
            </Panel>
          </div>

          <div className="grid__full">
            <Panel
              eyebrow="About Aegis"
              title="Architecture"
              description="For recruiters and technical reviewers — how a mission actually moves through the system."
            >
              <div className="architecture-flow">
                {["Frontend", "FastAPI", "Orchestrator", "Specialized Agents", "RAG + Evidence", "Safety", "Evaluation", "Mission Brief"].map(
                  (stage, index, all) => (
                    <span key={stage} className="architecture-flow__stage">
                      {stage}
                      {index < all.length - 1 ? <span aria-hidden="true"> → </span> : null}
                    </span>
                  ),
                )}
              </div>
              <h4 className="architecture-highlights__title">Engineering highlights</h4>
              <ul className="architecture-highlights">
                {ENGINEERING_HIGHLIGHTS.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </Panel>
          </div>
        </section>
      </main>
    </div>
  );
}