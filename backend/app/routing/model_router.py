from app.llm.registry import MODEL_REGISTRY, ModelConfig
from app.routing.policies import RoutingPolicy


class ModelRouter:

    def route(self, policy: RoutingPolicy) -> ModelConfig:
        candidates = list(MODEL_REGISTRY.values())

        if policy.require_tools:
            candidates = [
                model for model in candidates
                if model.supports_tools
            ]

        if policy.require_vision:
            candidates = [
                model for model in candidates
                if model.supports_vision
            ]

        if policy.preferred_provider:
            provider_models = [
                model for model in candidates
                if model.provider == policy.preferred_provider
            ]

            # Treat provider as a preference rather than a hard requirement.
            if provider_models:
                candidates = provider_models

        if not candidates:
            raise RuntimeError(
                "No registered model satisfies the routing requirements."
            )

        return max(
            candidates,
            key=lambda model: self._score(model, policy),
        )

    @staticmethod
    def _score(
        model: ModelConfig,
        policy: RoutingPolicy,
    ) -> float:

        if policy.preference == "quality":
            return model.quality_score

        if policy.preference == "speed":
            return model.speed_score

        if policy.preference == "cost":
            return model.cost_score

        # Balanced routing
        return (
            model.quality_score * 0.5
            + model.speed_score * 0.25
            + model.cost_score * 0.25
        )


model_router = ModelRouter()