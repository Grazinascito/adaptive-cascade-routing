"""Matemática pura do roteamento da cascata (Componentes D e E).

Sem I/O: `softmax_renormalize`/`shannon_entropy_bits`/`top_confidence`
transformam as logprobs do System 1 nos sinais de confiança, e
`CascadeRouter.decide` aplica a regra R_α do RouteLLM.
"""

import math
from collections.abc import Mapping


def softmax_renormalize(logprobs: Mapping[str, float]) -> dict[str, float]:
    """Softmax numéricamente estável (subtrai o máximo) sobre logprobs."""
    if not logprobs:
        raise ValueError("logprobs must not be empty")
    maximum = max(logprobs.values())
    exps = {label: math.exp(value - maximum) for label, value in logprobs.items()}
    total = sum(exps.values())
    return {label: value / total for label, value in exps.items()}


def shannon_entropy_bits(probs: Mapping[str, float]) -> float:
    """Entropia de Shannon em bits; termos com p = 0 valem 0 por convenção."""
    return -sum(p * math.log2(p) for p in probs.values() if p > 0.0)


def top_confidence(probs: Mapping[str, float]) -> float:
    """P_top: a maior probabilidade da distribuição."""
    if not probs:
        raise ValueError("probs must not be empty")
    return max(probs.values())


class CascadeRouter:
    """Regra R_α do RouteLLM: aceita o System 1 quando P_top >= α."""

    @staticmethod
    def decide(p_top: float, alpha: float) -> bool:
        """True = aceitar o System 1; False = escalar para o System 2."""
        if not math.isfinite(p_top) or not 0.0 <= p_top <= 1.0:
            raise ValueError(f"p_top must be a finite value in [0, 1], got {p_top!r}")
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be a finite value in [0, 1], got {alpha!r}")
        return p_top >= alpha
