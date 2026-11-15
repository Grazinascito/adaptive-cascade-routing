"""Testes dos Componentes D e E: entropia de Shannon e decisão de rota da cascata.

Cada teste cobre um cenário de aceite, sem depender de vLLM nem de rede. Pós-pivot
NCM (2026-11-14) o sinal de incerteza vem dos logprobs da geração curta;
`softmax`/`compute_metrics` da era /v1/systemone foram removidos com o
endpoint (histórico no git, commit 79a8654 e anteriores).
"""

import math

import pytest

from src.engine.router import (
    RoutingQuery,
    System2Handler,
    route,
    shannon_entropy,
)

_QUERY: RoutingQuery = {"item_name": "CACHACA BOAZINHA BALSAMO 670ML"}


def _spy_system2(calls: list[RoutingQuery]) -> System2Handler:
    """Handler de System 2 que registra as queries recebidas e devolve um código fixo."""

    def _handle(query: RoutingQuery) -> str:
        calls.append(query)
        return "22084000"

    return _handle


def test_shannon_entropy_returns_known_values_for_reference_distributions() -> None:
    # Distribuição uniforme sobre 2 tokens: 1 bit; determinística: 0 bit.
    assert shannon_entropy({"a": 0.5, "b": 0.5}) == pytest.approx(1.0)
    assert shannon_entropy({"a": 1.0, "b": 0.0}) == pytest.approx(0.0, abs=1e-12)
    # 90/10 (o exemplo clássico da spec): ~0.469 bits.
    assert shannon_entropy({"a": 0.9, "b": 0.1}) == pytest.approx(0.468996, abs=1e-6)


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
        "system2_prediction": "22084000",
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
    assert low["system2_prediction"] == "22084000"
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
