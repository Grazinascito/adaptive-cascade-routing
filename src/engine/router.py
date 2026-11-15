"""EntropyEvaluator + CascadeRouter — entropia de Shannon e decisão de rota (α, RouteLLM).

Componentes D e E (tabela de componentes no README);
stories correspondentes em docs/engine-router-spec.md. Pós-pivot NCM
(2026-11-14) não há mais pergunta `choice` binária: o sinal de incerteza do
fast-path é a média geométrica dos logprobs dos tokens gerados (calculada em
src/engine/system1.py) e a entropia de Shannon sobre os top-logprobs do
primeiro token (função deste módulo). `softmax`/`compute_metrics` da era
ASSIN foram removidos com o endpoint `/v1/systemone` — o histórico está no
git (commit 79a8654 e anteriores).
"""

import math
from collections.abc import Callable, Mapping
from typing import Literal, TypedDict


def shannon_entropy(probabilities: Mapping[str, float]) -> float:
    """Calcula a entropia de Shannon em bits; termos com p = 0 valem 0 por convenção."""
    return -sum(p * math.log2(p) for p in probabilities.values() if p > 0.0)


Route = Literal["SYSTEM_1", "SYSTEM_2"]


class RoutingQuery(TypedDict):
    item_name: str


System2Handler = Callable[[RoutingQuery], str]


class RouteOutcome(TypedDict):
    route: Route

    routed_to_system_2: bool

    system2_prediction: str | None


def route(
    query: RoutingQuery,
    confidence: float,
    alpha: float,
    system2: System2Handler,
) -> RouteOutcome:
    """Componente E: decisão R_α com repasse da query ao System 2 injetado.

    Fast-path é `confidence >= alpha` (FR-005): retorna System 1 sem invocar o
    handler. Levanta `ValueError` se `confidence` ou `alpha` não forem valores
    finitos em [0, 1] (as pontas 0.0 e 1.0 são permitidas).
    """
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be a finite value in [0, 1], got {confidence!r}")
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be a finite value in [0, 1], got {alpha!r}")

    if confidence >= alpha:
        return RouteOutcome(
            route="SYSTEM_1",
            routed_to_system_2=False,
            system2_prediction=None,
        )

    prediction = system2(query)
    return RouteOutcome(
        route="SYSTEM_2",
        routed_to_system_2=True,
        system2_prediction=prediction,
    )
