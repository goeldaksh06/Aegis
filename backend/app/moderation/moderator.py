from __future__ import annotations

import re

from app.models.schemas import ModerationResult

# Deterministic, free, zero-latency safety screen — no LLM call, runs on every request.
# Two independent checks: (1) does the incoming prompt look like a prompt-injection /
# jailbreak attempt, in which case the LLM is never called at all; (2) does the *response*
# text contain patterns that look like leaked PII, in which case it's flagged (not blocked —
# reliably redacting model output without corrupting its meaning is a much harder problem
# than a keyword filter can solve, so this is deliberately a review flag, not a hard block).

_INJECTION_PATTERNS = [
    re.compile(r"ignore (all|any|the) (previous|prior|above) instructions", re.I),
    re.compile(r"disregard (all|any|the) (previous|prior|above) (instructions|rules)", re.I),
    re.compile(r"reveal (your|the) system prompt", re.I),
    re.compile(r"you are now (DAN|in developer mode|unrestricted)", re.I),
    re.compile(r"pretend (you have no|there are no) (restrictions|rules|guidelines)", re.I),
    re.compile(r"act as if you have no (content policy|safety|restrictions)", re.I),
]

_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(r"\b(?:\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"),
    "ssn_like": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card_like": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
}


def check_prompt_safety(prompt: str) -> ModerationResult:
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(prompt):
            return ModerationResult(
                blocked=True,
                block_reason=(
                    "This request matches a known prompt-injection pattern "
                    f"({pattern.pattern!r}) and was not sent to the model."
                ),
            )
    return ModerationResult(blocked=False)


def screen_response_for_pii(content: str) -> list[str]:
    return sorted(name for name, pattern in _PII_PATTERNS.items() if pattern.search(content))
