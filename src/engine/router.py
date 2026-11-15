"""Pure routing math for the cascade (Components D and E).

No I/O: `softmax_renormalize`/`shannon_entropy_bits`/`top_confidence`
turn the System 1 logprobs into confidence signals, and
`CascadeRouter.decide` applies RouteLLM's R_α rule.
"""

import math
from collections.abc import Mapping


def softmax_renormalize(logprobs: Mapping[str, float]) -> dict[str, float]:
    """Numerically stable softmax (shift by the max) over logprobs."""
    if not logprobs:
        raise ValueError("logprobs must not be empty")
    maximum = max(logprobs.values())
    exps = {label: math.exp(value - maximum) for label, value in logprobs.items()}
    total = sum(exps.values())
    return {label: value / total for label, value in exps.items()}


def shannon_entropy_bits(probs: Mapping[str, float]) -> float:
    """Shannon entropy in bits; terms with p = 0 count as 0 by convention."""
    return -sum(p * math.log2(p) for p in probs.values() if p > 0.0)


def top_confidence(probs: Mapping[str, float]) -> float:
    """P_top: the largest probability in the distribution."""
    if not probs:
        raise ValueError("probs must not be empty")
    return max(probs.values())


class CascadeRouter:
    """RouteLLM's R_α rule: accept System 1 when P_top >= α."""

    @staticmethod
    def decide(p_top: float, alpha: float) -> bool:
        """True = accept System 1; False = escalate to System 2."""
        if not math.isfinite(p_top) or not 0.0 <= p_top <= 1.0:
            raise ValueError(f"p_top must be a finite value in [0, 1], got {p_top!r}")
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be a finite value in [0, 1], got {alpha!r}")
        return p_top >= alpha
