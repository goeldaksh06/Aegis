from app.agents.mission_brief import (
    NO_ACTIONS_PLACEHOLDER,
    NO_ALERTS_PLACEHOLDER,
    NO_EVIDENCE_PLACEHOLDER,
)
from app.eval.evaluator import evaluate_response
from app.models.schemas import MissionBrief, RiskLevel


def _brief(**overrides) -> MissionBrief:
    defaults = dict(
        summary="A disruption occurred.",
        risk_score=80,
        risk_level=RiskLevel.HIGH,
        top_alerts=["Critical delay at the hub."],
        recommended_actions=["Activate the contingency plan."],
        evidence=["Transit times increased 30%."],
    )
    defaults.update(overrides)
    return MissionBrief(**defaults)


def _rag_tool_result(chunks: list[dict]) -> dict:
    return {
        "tool_type": "rag",
        "output": "...",
        "success": True,
        "metadata": {"chunks": chunks},
    }


def test_structure_quality_is_full_when_all_sections_have_real_content():
    result = evaluate_response("some content", _brief(), tool_results=[])
    assert result.structure_quality == 1.0
    assert result.groundedness is None
    assert result.overall_score == 1.0


def test_structure_quality_drops_when_sections_are_placeholders():
    brief = _brief(
        top_alerts=[NO_ALERTS_PLACEHOLDER],
        recommended_actions=[NO_ACTIONS_PLACEHOLDER],
        evidence=["Transit times increased 30%."],
    )
    result = evaluate_response("some content", brief, tool_results=[])
    assert round(result.structure_quality, 2) == round(1 / 3, 2)


def test_structure_quality_is_zero_when_mission_brief_is_none():
    result = evaluate_response("some content", None, tool_results=[])
    assert result.structure_quality == 0.0


def test_groundedness_is_none_when_no_rag_context_was_retrieved():
    result = evaluate_response("some content", _brief(), tool_results=[])
    assert result.groundedness is None


def test_groundedness_is_none_when_rag_ran_but_found_nothing():
    result = evaluate_response(
        "some content", _brief(), tool_results=[_rag_tool_result([])]
    )
    assert result.groundedness is None


def test_groundedness_is_high_when_response_reuses_retrieved_vocabulary():
    chunk_text = "contingency logistics hub secondary reroute shipments backup vendors"
    response = (
        "We should activate the contingency logistics hub and reroute shipments "
        "through backup vendors immediately."
    )
    result = evaluate_response(
        response,
        _brief(),
        tool_results=[_rag_tool_result([{"text": chunk_text}])],
    )
    assert result.groundedness == 1.0
    assert result.overall_score == (result.structure_quality + 1.0) / 2


def test_groundedness_is_low_when_response_ignores_retrieved_context():
    chunk_text = "contingency logistics hub secondary reroute shipments backup vendors"
    response = "The weather today is sunny with a light breeze."
    result = evaluate_response(
        response,
        _brief(),
        tool_results=[_rag_tool_result([{"text": chunk_text}])],
    )
    assert result.groundedness == 0.0
    assert "barely reflects" in " ".join(result.notes)


def test_judge_score_is_not_set_by_the_deterministic_evaluator():
    result = evaluate_response("some content", _brief(), tool_results=[])
    assert result.judge_score is None
