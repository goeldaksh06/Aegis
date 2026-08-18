import pytest

from app.llm.registry import MODEL_REGISTRY, ModelConfig, register_model
from app.routing.model_router import model_router
from app.routing.policies import RoutingPolicy


@pytest.fixture(autouse=True)
def setup_registry():
    MODEL_REGISTRY.clear()

    register_model(
        ModelConfig(
            name="fast-model",
            provider="provider-a",
            tier="fast",
            supports_tools=False,
            quality_score=0.60,
            speed_score=0.95,
            cost_score=0.95,
        )
    )

    register_model(
        ModelConfig(
            name="balanced-model",
            provider="provider-b",
            tier="balanced",
            supports_tools=True,
            quality_score=0.80,
            speed_score=0.75,
            cost_score=0.70,
        )
    )

    register_model(
        ModelConfig(
            name="powerful-model",
            provider="provider-c",
            tier="powerful",
            supports_tools=True,
            quality_score=0.98,
            speed_score=0.40,
            cost_score=0.30,
        )
    )

    yield

    MODEL_REGISTRY.clear()


def test_speed_routing():
    selected = model_router.route(
        RoutingPolicy(preference="speed")
    )

    assert selected.name == "fast-model"


def test_cost_routing():
    selected = model_router.route(
        RoutingPolicy(preference="cost")
    )

    assert selected.name == "fast-model"


def test_quality_routing():
    selected = model_router.route(
        RoutingPolicy(preference="quality")
    )

    assert selected.name == "powerful-model"


def test_tool_requirement():
    selected = model_router.route(
        RoutingPolicy(
            preference="speed",
            require_tools=True,
        )
    )

    assert selected.name == "balanced-model"


def test_no_matching_model():
    MODEL_REGISTRY.clear()

    register_model(
        ModelConfig(
            name="text-only",
            provider="provider-a",
            tier="fast",
            supports_vision=False,
        )
    )

    with pytest.raises(RuntimeError):
        model_router.route(
            RoutingPolicy(require_vision=True)
        )