"""Pure router (PR3): softmax, entropy, P_top and RouteLLM's R_α rule."""

import math

import pytest

from src.engine.router import (
    CascadeRouter,
    shannon_entropy_bits,
    softmax_renormalize,
    top_confidence,
)


def test_softmax_renormalization_sums_to_one() -> None:
    logprobs = {"MERCEARIA": -0.9, "HORTIFRUTI": -2.1, "PADARIA E CONFEITARIA": -4.7}

    probs = softmax_renormalize(logprobs)

    assert math.isclose(sum(probs.values()), 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert max(probs, key=probs.__getitem__) == "MERCEARIA"
    direct = {label: math.exp(value) for label, value in logprobs.items()}
    total = sum(direct.values())
    for label, value in direct.items():
        assert math.isclose(probs[label], value / total, rel_tol=1e-12)


def test_softmax_uniform_logprobs_gives_uniform_probs() -> None:
    probs = softmax_renormalize({"A": -3.0, "B": -3.0, "C": -3.0})

    assert all(math.isclose(value, 1.0 / 3.0, rel_tol=1e-12) for value in probs.values())


def test_shannon_entropy_uniform_is_log2_of_support_size() -> None:
    uniform = {label: 0.25 for label in "ABCD"}

    assert math.isclose(shannon_entropy_bits(uniform), 2.0, rel_tol=1e-12)


def test_shannon_entropy_one_hot_is_zero() -> None:
    one_hot = {"A": 1.0, "B": 0.0, "C": 0.0}

    assert shannon_entropy_bits(one_hot) == 0.0


def test_top_confidence_returns_max_probability() -> None:
    assert top_confidence({"A": 0.2, "B": 0.5, "C": 0.3}) == 0.5


def test_empty_inputs_raise() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        softmax_renormalize({})
    with pytest.raises(ValueError, match="must not be empty"):
        top_confidence({})


def test_decide_accepts_when_p_top_equals_alpha_boundary() -> None:
    assert CascadeRouter.decide(0.85, 0.85) is True


def test_decide_routes_when_p_top_below_alpha() -> None:
    assert CascadeRouter.decide(0.8499, 0.85) is False


def test_decide_alpha_zero_accepts_everything() -> None:
    assert CascadeRouter.decide(0.0, 0.0) is True
    assert CascadeRouter.decide(0.001, 0.0) is True
    assert CascadeRouter.decide(1.0, 0.0) is True


def test_decide_alpha_one_routes_anything_below_exact_one() -> None:
    assert CascadeRouter.decide(0.999999, 1.0) is False
    assert CascadeRouter.decide(1.0, 1.0) is True


def test_decide_rejects_out_of_range_inputs() -> None:
    with pytest.raises(ValueError, match="p_top"):
        CascadeRouter.decide(1.5, 0.85)
    with pytest.raises(ValueError, match="alpha"):
        CascadeRouter.decide(0.85, -0.1)
