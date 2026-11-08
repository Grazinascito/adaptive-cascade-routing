"""Testes dos Componentes D e E: métricas de incerteza e decisão de rota da cascata.

Cada teste corresponde a um cenário de aceite das Stories 3–4 de
docs/engine-router-spec.md. A aritmética é verificada contra valores
analíticos conhecidos (softmax de dois rótulos e entropia de Shannon),
sem depender de vLLM nem de rede.
"""

import math

import pytest

from src.engine.router import compute_metrics, softmax

_CHOICES = ["Entailment", "Contradiction"]


def test_softmax_renormalizes_logprobs_into_probabilities_that_sum_to_one() -> None:
    probabilities = softmax(
        {"Entailment": math.log(0.6), "Contradiction": math.log(0.4)}, _CHOICES
    )

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
