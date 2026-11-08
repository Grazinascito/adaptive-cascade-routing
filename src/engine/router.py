"""EntropyEvaluator + CascadeRouter — softmax, entropia de Shannon e decisão de rota (α, RouteLLM).

Componentes D e E do design doc (docs/adaptive-cascade-design-doc.md §5.4 e §5.5);
stories correspondentes em docs/engine-router-spec.md (Stories 3 e 4).
"""

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Final, Literal, TypedDict

from src.data.loader import Label

LOGPROB_FLOOR: Final = -100.0


def softmax(logprobs: Mapping[str, float], choices: Sequence[Label]) -> dict[Label, float]:
    """Converte os logprobs dos rótulos de `choices` em probabilidades normalizadas.

    Rótulos de `choices` ausentes em `logprobs` recebem `LOGPROB_FLOOR`; chaves de
    `logprobs` fora de `choices` são ignoradas. A subtração do máximo antes de
    `exp` evita overflow para logprobs grandes.
    """
    candidates = {choice: logprobs.get(choice, LOGPROB_FLOOR) for choice in choices}
    top = max(candidates.values())
    exps = {choice: math.exp(value - top) for choice, value in candidates.items()}
    total = sum(exps.values())
    return {choice: value / total for choice, value in exps.items()}


def shannon_entropy(probabilities: Mapping[Label, float]) -> float:
    """Calcula a entropia de Shannon em bits; termos com p = 0 valem 0 por convenção."""
    return -sum(p * math.log2(p) for p in probabilities.values() if p > 0.0)


class DecisionMetrics(TypedDict):
    top_choice: Label

    confidence: float

    entropy: float

    probabilities: dict[Label, float]


def compute_metrics(logprobs: Mapping[str, float], choices: Sequence[Label]) -> DecisionMetrics:
    """Componente D: métricas de incerteza sobre os logprobs dos rótulos candidatos.

    Levanta `ValueError` se `choices` tiver menos de dois rótulos ou duplicatas,
    se nenhum rótulo de `choices` aparecer em `logprobs` (nenhum sinal) ou se
    algum logprob presente não for finito. Empates de probabilidade são
    desfeitos pelo primeiro rótulo na ordem de `choices`.
    """
    if len(choices) < 2:
        raise ValueError(f"choices must contain at least two labels, got {choices!r}")
    if len(set(choices)) != len(choices):
        raise ValueError(f"choices must not contain duplicate labels, got {choices!r}")

    present = [choice for choice in choices if choice in logprobs]
    if not present:
        raise ValueError(f"no logprob signal for any of the choices {choices!r} in logprobs")
    for choice in present:
        if not math.isfinite(logprobs[choice]):
            raise ValueError(f"logprob for {choice!r} must be finite, got {logprobs[choice]!r}")

    probabilities = softmax(logprobs, choices)
    confidence = max(probabilities.values())
    top_choice = next(choice for choice in choices if probabilities[choice] == confidence)
    return DecisionMetrics(
        top_choice=top_choice,
        confidence=confidence,
        entropy=shannon_entropy(probabilities),
        probabilities=probabilities,
    )


Route = Literal["SYSTEM_1", "SYSTEM_2"]


class RoutingQuery(TypedDict):
    premise: str

    hypothesis: str


System2Handler = Callable[[RoutingQuery], Label]


class RouteOutcome(TypedDict):
    route: Route

    routed_to_system_2: bool

    system2_prediction: Label | None


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
