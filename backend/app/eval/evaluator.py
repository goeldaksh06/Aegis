from __future__ import annotations

import re

from app.agents.mission_brief import (
    NO_ACTIONS_PLACEHOLDER,
    NO_ALERTS_PLACEHOLDER,
    NO_EVIDENCE_PLACEHOLDER,
)
from app.models.schemas import MissionBrief, ResponseEvaluation

_SIGNIFICANT_WORD = re.compile(r"[a-z]{4,}")
_MIN_SHARED_WORDS_FOR_A_HIT = 5
_PLACEHOLDERS = {NO_ALERTS_PLACEHOLDER, NO_ACTIONS_PLACEHOLDER, NO_EVIDENCE_PLACEHOLDER}


def _structure_quality(mission_brief: MissionBrief | None) -> float:
    """Fraction of the brief's three sections that hold real content, not a fallback placeholder."""
    if mission_brief is None:
        return 0.0

    sections = [
        mission_brief.top_alerts,
        mission_brief.recommended_actions,
        mission_brief.evidence,
    ]
    real_sections = sum(
        1 for section in sections if section and section[0] not in _PLACEHOLDERS
    )
    return real_sections / len(sections)


def _significant_words(text: str) -> set[str]:
    return set(_SIGNIFICANT_WORD.findall(text.lower()))


def _groundedness(content: str, tool_results: list[dict]) -> float | None:
    """Fraction of retrieved RAG chunks whose vocabulary actually shows up in the response.

    Returns None (not 0.0) when no RAG context was retrieved — there's nothing to be
    grounded against, so "ungrounded" would be a misleading label. A crude word-overlap
    heuristic, not semantic similarity, but cheap enough to run on every request for free.
    """
    rag_results = [t for t in tool_results if t.get("tool_type") == "rag"]
    if not rag_results:
        return None

    chunks = rag_results[0].get("metadata", {}).get("chunks", [])
    if not chunks:
        return None

    content_words = _significant_words(content)
    hits = sum(
        1
        for chunk in chunks
        if len(content_words & _significant_words(chunk.get("text", ""))) >= _MIN_SHARED_WORDS_FOR_A_HIT
    )
    return hits / len(chunks)


def evaluate_response(
    content: str,
    mission_brief: MissionBrief | None,
    tool_results: list[dict],
) -> ResponseEvaluation:
    """Deterministic, zero-cost quality signal computed for every response.

    Two independent, cheap checks — no LLM call, no network, safe to run unconditionally
    (including in CI on the mock provider). See app/eval/judge.py for the separate,
    opt-in, costed LLM-as-judge path.
    """
    structure = _structure_quality(mission_brief)
    grounded = _groundedness(content, tool_results)

    scored_parts = [structure] + ([grounded] if grounded is not None else [])
    overall = sum(scored_parts) / len(scored_parts)

    notes: list[str] = []
    if grounded is not None and grounded < 0.34:
        notes.append("Retrieved context was available but the response barely reflects it.")
    if structure < 0.67:
        notes.append("Mission brief is missing real content in one or more sections.")

    return ResponseEvaluation(
        groundedness=grounded,
        structure_quality=structure,
        overall_score=overall,
        notes=notes,
    )
