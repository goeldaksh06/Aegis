from app.llm.pricing import estimate_cost_usd


def test_mock_model_is_free():
    assert estimate_cost_usd("mock-default", 1000, 1000) == 0.0


def test_known_model_computes_nonzero_cost():
    cost = estimate_cost_usd("deepseek-chat", 1000, 1000)
    assert cost > 0.0


def test_unknown_model_falls_back_to_default_price_instead_of_raising():
    cost = estimate_cost_usd("some-future-model-not-in-registry", 1000, 1000)
    assert cost > 0.0


def test_zero_tokens_costs_nothing():
    assert estimate_cost_usd("deepseek-chat", None, None) == 0.0
