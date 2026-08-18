from __future__ import annotations

import re

from app.models.schemas import MissionBrief, RiskLevel

# Exported so app/eval/evaluator.py can detect "empty" mission briefs without duplicating
# these exact strings — a brief that's entirely these placeholders extracted nothing real.
NO_ALERTS_PLACEHOLDER = "No explicit alerts extracted from this response."
NO_ACTIONS_PLACEHOLDER = "No concrete actions were extracted. Ask for action-oriented recommendations."
NO_EVIDENCE_PLACEHOLDER = "No explicit evidence lines found. Ask the model to cite data or sources."

_WEIGHTED_SIGNALS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"(critical|severe|catastrophic|collapse)"), 20),
    (re.compile(r"(immediate|urgent|escalat|emergency)"), 12),
    (re.compile(r"(disruption|outage|shortage|bottleneck|delay)"), 9),
    (re.compile(r"(volatility|exposure|liquidity|compliance|regulatory)"), 7),
    (re.compile(r"(mitigat|contingenc|monitor|fallback)"), -6),
]

_ALERT_SIGNALS = re.compile(
    r"(risk|alert|threat|impact|loss|delay|shortage|outage|vulnerab|exposure)", re.I
)
_ACTION_SIGNALS = re.compile(
    r"(recommend|action|mitigat|prioriti|deploy|activate|coordinate|communicat|escalat|hedge|review)",
    re.I,
)
_ACTION_LEAD_WORDS = re.compile(
    r"^(use|shift|activate|review|reduce|increase|secure|prepare|monitor)", re.I
)
_EVIDENCE_SIGNALS = re.compile(
    r"(because|due to|according|signal|indicator|data|metric|report|trend|evidence|observed)",
    re.I,
)
_BULLET_PREFIX = re.compile(r"^[-*\d.)\s]+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_SECTION_HEADER = re.compile(r"^[A-Za-z][\w /-]{0,40}:$")
_MD_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_MD_TABLE_SEP = re.compile(r"^\s*[|:\s-]+$")
_MD_HEADER_PREFIX = re.compile(r"^#{1,6}\s*")
_MD_EMPHASIS = re.compile(r"\*{1,3}|_{2,3}")


def _strip_markdown_noise(lines: list[str]) -> list[str]:
    """Drop markdown table rows/separators and strip heading/emphasis markup.

    Real LLM output (unlike the mock provider's plain prose) is often formatted with
    markdown tables, "### " headers, and "**bold**" emphasis — none of which the
    keyword heuristic below is meant to parse. Without this, table pipes and heading
    hashes leak into the alerts/actions/evidence lists as if they were content.
    """
    cleaned: list[str] = []
    for line in lines:
        if _MD_TABLE_ROW.match(line) or _MD_TABLE_SEP.match(line):
            continue
        line = _MD_HEADER_PREFIX.sub("", line)
        line = _MD_EMPHASIS.sub("", line)
        line = line.strip()
        if line:
            cleaned.append(line)
    return cleaned


def _score_risk(content: str) -> tuple[int, RiskLevel]:
    text = content.lower()
    score = 28

    for pattern, weight in _WEIGHTED_SIGNALS:
        score += len(pattern.findall(text)) * weight

    score = max(10, min(95, score))

    if score >= 70:
        return score, RiskLevel.HIGH
    if score >= 40:
        return score, RiskLevel.MEDIUM
    return score, RiskLevel.LOW


def _sentences(content: str) -> list[str]:
    collapsed = re.sub(r"\s+", " ", content)
    return [s.strip() for s in _SENTENCE_SPLIT.split(collapsed) if s.strip()]


def _dedupe_and_limit(items: list[str], limit: int) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        if item not in seen:
            seen[item] = None
        if len(seen) >= limit:
            break
    return list(seen.keys())


def build_mission_brief(content: str) -> MissionBrief | None:
    """Derive a structured decision brief from an agent's raw text output.

    This is a deterministic heuristic (keyword/regex scoring), not a model-guaranteed
    structured output. Chosen over provider-side JSON mode to avoid per-provider prompting/
    parsing differences and to keep the mock provider's behavior identical to a real one.
    """
    stripped = content.strip()
    if not stripped:
        return None

    lines = _strip_markdown_noise(
        [line.strip() for line in stripped.splitlines() if line.strip()]
    )
    score, level = _score_risk(stripped)

    bullet_lines = [
        _BULLET_PREFIX.sub("", line).strip()
        for line in lines
        if not _SECTION_HEADER.match(line)
    ]
    sentences = _sentences(" ".join(bullet_lines))
    summary = " ".join(sentences[:2])[:320] or stripped[:320]

    top_alerts = _dedupe_and_limit(
        [line for line in bullet_lines if _ALERT_SIGNALS.search(line)]
        + [s for s in sentences if _ALERT_SIGNALS.search(s)],
        3,
    )
    recommended_actions = _dedupe_and_limit(
        [line for line in bullet_lines if _ACTION_SIGNALS.search(line)]
        + [s for s in sentences if _ACTION_LEAD_WORDS.match(s)],
        4,
    )
    evidence = _dedupe_and_limit(
        [line for line in bullet_lines if _EVIDENCE_SIGNALS.search(line)]
        + [s for s in sentences if _EVIDENCE_SIGNALS.search(s)],
        3,
    )

    return MissionBrief(
        summary=summary,
        risk_score=score,
        risk_level=level,
        top_alerts=top_alerts or [NO_ALERTS_PLACEHOLDER],
        recommended_actions=recommended_actions or [NO_ACTIONS_PLACEHOLDER],
        evidence=evidence or [NO_EVIDENCE_PLACEHOLDER],
    )
