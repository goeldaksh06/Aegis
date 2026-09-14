import type { ReactElement } from "react";
import { Link } from "react-router-dom";

function IconRoute() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="6" cy="6" r="2.5" />
      <circle cx="18" cy="18" r="2.5" />
      <path d="M8.5 6H14a4 4 0 0 1 4 4v2" />
      <path d="m15.5 9.5 2.5 2.5 2.5-2.5" />
    </svg>
  );
}

function IconLibrary() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 19.5V5a1 1 0 0 1 1-1h5v16" />
      <path d="M10 20h9a1 1 0 0 0 1-1V6.5" />
      <path d="M14 4.5 20 6v12.5" />
      <path d="M7 8h1" />
      <path d="M7 11h1" />
    </svg>
  );
}

function IconPulse() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 12h4l2 7 4-14 2 7h6" />
    </svg>
  );
}

function IconLock() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <rect x="5" y="10.5" width="14" height="9" rx="1.5" />
      <path d="M8 10.5V7a4 4 0 0 1 8 0v3.5" />
      <path d="M12 14.5v2" />
    </svg>
  );
}

interface CapabilityItem {
  Icon: () => ReactElement;
  title: string;
  detail: string;
}

const CAPABILITIES: CapabilityItem[] = [
  {
    Icon: IconRoute,
    title: "Five agents, real routing",
    detail:
      "Research, Analyst, Coder, Document, and Planner. A rule-based router picks the right one, " +
      "or a Planner agent decomposes a mission and dispatches sub-steps to the others — genuine " +
      "agent-to-agent handoff, not a single hardcoded prompt chain.",
  },
  {
    Icon: IconLibrary,
    title: "Grounded, not just generated",
    detail:
      "Retrieval runs against an indexed knowledge base across four domains, with a relevance " +
      "threshold — off-topic questions get the model's own knowledge, not force-fed irrelevant " +
      "documents padded in to look grounded.",
  },
  {
    Icon: IconPulse,
    title: "Watch it think",
    detail:
      "Every mission streams live over Server-Sent Events — routing, retrieval, generation, " +
      "evaluation, persistence — and every agent step's real duration, tokens, and cost is " +
      "stored and queryable afterward, not estimated after the fact.",
  },
  {
    Icon: IconLock,
    title: "Built like it matters",
    detail:
      "Prompt-injection screening before any model call, PII flagging on responses, JWT auth with " +
      "per-user data isolation enforced server-side, and 100+ automated tests running in CI on " +
      "every push.",
  },
];

const STEPS = [
  { n: "01", title: "Describe the situation", detail: "A disruption, an incident, a decision you need evidence for." },
  { n: "02", title: "Aegis routes it", detail: "The right agent handles it — or a Planner splits it across several." },
  { n: "03", title: "Evidence gets pulled in", detail: "Retrieval runs against the indexed knowledge base, when relevant." },
  { n: "04", title: "You get a brief, not a wall of text", detail: "Risk score, findings, actions, evidence, cost, and a quality score." },
];

export function LandingPage() {
  return (
    <div className="landing">
      <nav className="landing-nav">
        <span className="landing-nav__mark">Aegis</span>
        <div className="landing-nav__links">
          <a href="https://github.com/goeldaksh06/Aegis" target="_blank" rel="noreferrer">
            Source
          </a>
          <Link to="/login">Sign in</Link>
          <Link to="/console" className="landing-nav__cta">
            Launch Console
          </Link>
        </div>
      </nav>

      <header className="landing-hero">
        <p className="landing-hero__kicker">Multi-agent decision intelligence</p>
        <h1 className="landing-hero__title">
          A question isn't a decision.
          <br />
          Aegis closes that gap.
        </h1>
        <p className="landing-hero__subtitle">
          Describe a situation and Aegis routes it to the right specialized agent, grounds the
          answer in real retrieved evidence, evaluates its own output, and hands you a
          risk-scored brief — with the full execution trace behind it, not a black box.
        </p>
        <div className="landing-hero__actions">
          <Link to="/console" className="run-button">
            Try it live
          </Link>
          <a className="landing-hero__secondary" href="#how-it-works">
            See how it works ↓
          </a>
        </div>
      </header>

      <section className="landing-section">
        <p className="landing-section__eyebrow">Why not just ask a chatbot</p>
        <h2>Because a chat window doesn't ground, verify, or remember why it said something.</h2>
        <p className="landing-section__body">
          A general-purpose chat model gives you its best guess from training data, with no way
          to check what it actually used. Aegis is different in three concrete ways: it retrieves
          from a real, indexed knowledge base before answering; it scores its own response for
          groundedness and structure before returning it; and it persists the full record — every
          agent step, every token, every dollar — so an answer from last week is still auditable
          today.
        </p>
      </section>

      <section className="landing-section" id="how-it-works">
        <p className="landing-section__eyebrow">How it works</p>
        <h2>Four steps, every time.</h2>
        <div className="landing-steps">
          {STEPS.map((step) => (
            <div key={step.n} className="landing-step">
              <span className="landing-step__n">{step.n}</span>
              <h3>{step.title}</h3>
              <p>{step.detail}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="landing-section">
        <p className="landing-section__eyebrow">What's actually under the hood</p>
        <h2>Not a wrapper around one API call.</h2>
        <div className="landing-capabilities">
          {CAPABILITIES.map((cap) => (
            <div key={cap.title} className="landing-capability">
              <div className="landing-capability__icon">
                <cap.Icon />
              </div>
              <h3>{cap.title}</h3>
              <p>{cap.detail}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="landing-cta">
        <h2>Run a real mission — it takes about ten seconds.</h2>
        <p>The demo works instantly, no account needed. Sign in later if you want a saved history.</p>
        <Link to="/console" className="run-button">
          Launch Console
        </Link>
      </section>

      <footer className="landing-footer">
        <span>Aegis — built as a working system, not a mockup.</span>
        <a href="https://github.com/goeldaksh06/Aegis" target="_blank" rel="noreferrer">
          View source
        </a>
      </footer>
    </div>
  );
}
