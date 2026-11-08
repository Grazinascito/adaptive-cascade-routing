"""Testes dos Componentes D e E: métricas de incerteza e decisão de rota da cascata.

Cada teste corresponde a um cenário de aceite das Stories 3–4 de
docs/engine-router-spec.md. A aritmética é verificada contra valores
analíticos conhecidos (softmax de dois rótulos e entropia de Shannon),
sem depender de vLLM nem de rede.
"""

import math

import pytest

from src.data.loader import Label
from src.engine.router import (
    RoutingQuery,
    System2Handler,
    compute_metrics,
    route,
    softmax,
)

_CHOICES = ["Entailment", "Contradiction"]

_QUERY: RoutingQuery = {
    "premise": "O gato dorme no sofá.",
    "hypothesis": "O gato está dormindo.",
}


def _spy_system2(calls: list[RoutingQuery]) -> System2Handler:
    """Handler de System 2 que registra as queries recebidas e devolve um rótulo fixo."""

    def _handle(query: RoutingQuery) -> Label:
        calls.append(query)
        return "Entailment"

    return _handle


def test_softmax_renormalizes_logprobs_into_probabilities_that_sum_to_one() -> None:
    probabilities = softmax({"Entailment": math.log(0.6), "Contradiction": math.log(0.4)}, _CHOICES)

    assert probabilities == pytest.approx({"Entailment": 0.6, "Contradiction": 0.4})
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_compute_metrics_returns_confidence_and_entropy_for_90_10_split() -> None:
    metrics = compute_metrics(
        {"Entailment": math.log(0.9), "Contradiction": math.log(0.1)}, _CHOICES
    )

    assert metrics["top_choice"] == "Entailment"
    assert metrics["confidence"] == pytest.approx(0.9, abs=1e-9)
    assert metrics["entropy"] == pytest.approx(0.468996, abs=1e-6)


def test_compute_metrics_returns_max_entropy_for_uniform_distribution() -> None:
    metrics = compute_metrics(
        {"Entailment": math.log(0.5), "Contradiction": math.log(0.5)}, _CHOICES
    )

    assert metrics["confidence"] == pytest.approx(0.5)
    assert metrics["entropy"] == pytest.approx(1.0)


def test_compute_metrics_returns_zero_entropy_for_deterministic_distribution() -> None:
    metrics = compute_metrics({"Entailment": 0.0, "Contradiction": -100.0}, _CHOICES)

    assert metrics["top_choice"] == "Entailment"
    assert metrics["confidence"] == pytest.approx(1.0)
    assert metrics["entropy"] == pytest.approx(0.0, abs=1e-12)


def test_compute_metrics_fills_missing_choice_with_logprob_floor() -> None:
    metrics = compute_metrics({"Entailment": -0.2}, _CHOICES)

    assert metrics["top_choice"] == "Entailment"
    assert metrics["confidence"] == pytest.approx(1.0)
    assert metrics["probabilities"]["Contradiction"] == pytest.approx(0.0, abs=1e-12)


def test_compute_metrics_ignores_logprob_keys_outside_choices() -> None:
    metrics = compute_metrics(
        {
            "Entailment": math.log(0.6),
            "Contradiction": math.log(0.4),
            "NeutralToken": math.log(0.99),
        },
        _CHOICES,
    )

    assert set(metrics["probabilities"]) == {"Entailment", "Contradiction"}
    assert metrics["confidence"] == pytest.approx(0.6)


def test_compute_metrics_breaks_probability_ties_by_first_choice_order() -> None:
    metrics = compute_metrics(
        {"Entailment": math.log(0.5), "Contradiction": math.log(0.5)}, _CHOICES
    )

    assert metrics["top_choice"] == "Entailment"


def test_compute_metrics_rejects_fewer_than_two_choices() -> None:
    with pytest.raises(ValueError, match="two"):
        compute_metrics({"Entailment": -0.1}, ["Entailment"])


def test_compute_metrics_rejects_non_finite_logprobs() -> None:
    for invalid in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="finite"):
            compute_metrics({"Entailment": invalid, "Contradiction": -0.1}, _CHOICES)


def test_compute_metrics_rejects_input_without_any_choice_signal() -> None:
    for empty_signal in ({}, {"Unrelated": -0.1}):
        with pytest.raises(ValueError, match="signal"):
            compute_metrics(empty_signal, _CHOICES)


def test_route_returns_system_1_without_invoking_system_2_when_confidence_meets_alpha() -> None:
    calls: list[RoutingQuery] = []

    outcome = route(_QUERY, confidence=0.92, alpha=0.85, system2=_spy_system2(calls))

    assert outcome == {
        "route": "SYSTEM_1",
        "routed_to_system_2": False,
        "system2_prediction": None,
    }
    assert calls == []


def test_route_dispatches_to_system_2_when_confidence_falls_below_alpha() -> None:
    calls: list[RoutingQuery] = []

    outcome = route(_QUERY, confidence=0.61, alpha=0.85, system2=_spy_system2(calls))

    assert outcome == {
        "route": "SYSTEM_2",
        "routed_to_system_2": True,
        "system2_prediction": "Entailment",
    }
    assert calls == [_QUERY]


def test_route_routes_to_system_1_when_confidence_equals_alpha_exactly() -> None:
    calls: list[RoutingQuery] = []

    outcome = route(_QUERY, confidence=0.85, alpha=0.85, system2=_spy_system2(calls))

    assert outcome["route"] == "SYSTEM_1"
    assert outcome["system2_prediction"] is None
    assert calls == []


def test_route_sends_everything_to_system_1_at_alpha_zero() -> None:
    calls: list[RoutingQuery] = []

    outcome = route(_QUERY, confidence=0.3, alpha=0.0, system2=_spy_system2(calls))

    assert outcome["route"] == "SYSTEM_1"
    assert calls == []


def test_route_sends_everything_except_perfect_confidence_to_system_2_at_alpha_one() -> None:
    calls: list[RoutingQuery] = []
    spy = _spy_system2(calls)

    low = route(_QUERY, confidence=0.999, alpha=1.0, system2=spy)
    perfect = route(_QUERY, confidence=1.0, alpha=1.0, system2=spy)

    assert low["route"] == "SYSTEM_2"
    assert low["system2_prediction"] == "Entailment"
    assert perfect["route"] == "SYSTEM_1"
    assert calls == [_QUERY]


def test_route_rejects_confidence_outside_unit_interval() -> None:
    for invalid in (-0.01, 1.01, math.nan):
        with pytest.raises(ValueError, match="confidence"):
            route(_QUERY, confidence=invalid, alpha=0.85, system2=_spy_system2([]))


def test_route_rejects_alpha_outside_unit_interval() -> None:
    for invalid in (-0.1, 1.1):
        with pytest.raises(ValueError, match="alpha"):
            route(_QUERY, confidence=0.9, alpha=invalid, system2=_spy_system2([]))
