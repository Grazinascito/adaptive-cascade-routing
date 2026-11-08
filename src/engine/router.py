"""EntropyEvaluator + CascadeRouter — softmax, entropia de Shannon e decisão de rota (α, RouteLLM).

Componentes D e E do design doc (docs/adaptive-cascade-design-doc.md §5.4 e §5.5);
stories correspondentes em docs/engine-router-spec.md (Stories 3 e 4).
"""

import math
from collections.abc import Mapping, Sequence
from typing import Final, TypedDict

LOGPROB_FLOOR: Final = -100.0


def softmax(logprobs: Mapping[str, float], choices: Sequence[str]) -> dict[str, float]:
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


def shannon_entropy(probabilities: Mapping[str, float]) -> float:
    """Calcula a entropia de Shannon em bits; termos com p = 0 valem 0 por convenção."""
    return -sum(p * math.log2(p) for p in probabilities.values() if p > 0.0)


class DecisionMetrics(TypedDict):
    top_choice: str

    confidence: float

    entropy: float

    probabilities: dict[str, float]


def compute_metrics(logprobs: Mapping[str, float], choices: Sequence[str]) -> DecisionMetrics:
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
