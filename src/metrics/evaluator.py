"""Pure metrics for the DARU benchmark (PR4) and the raw per-item record.

No HTTP and no router: they receive ready `BenchmarkRecord`s (or numbers)
so they can be tested in isolation. `summary.py` composes these functions
per scenario.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    """One item's measurements across the three cascade roles (DARU spec §5)."""

    item_id: str
    raw_text: str
    canonical_category: str
    is_long_tail: bool
    s1_prediction: str
    s1_p_top: float
    s1_entropy: float
    s1_latency_ms: float
    routed_to_s2: bool
    s2_prediction: str | None
    s2_reasoning: str | None
    s2_revisor_insight: str | None
    s2_decode_tokens: int | None
    s2_latency_ms: float | None
    final_prediction: str
    is_correct: bool


def accuracy(records: Sequence[BenchmarkRecord]) -> float:
    """Fraction of correct records (`is_correct`) over all records."""
    if not records:
        raise ValueError("records must not be empty")
    return sum(1 for record in records if record.is_correct) / len(records)


def slice_accuracy(
    records: Sequence[BenchmarkRecord], *, is_long_tail: bool
) -> float | None:
    """Accuracy restricted to one slice (tail = description with <= 20 characters).

    None when the slice has no records (head-only/tail-only scenario).
    """
    selected = [record for record in records if record.is_long_tail == is_long_tail]
    if not selected:
        return None
    return sum(1 for record in selected if record.is_correct) / len(selected)


def apgr(acc_s1: float, acc_s2: float, acc_cascade: float) -> float | None:
    """Average Performance Gap Recovered (RouteLLM, DARU spec §5).

    None when the denominator (gap S2 − S1) is <= 0: with no gap there is
    nothing to recover, and a negative gap (S1 >= S2) makes the ratio
    meaningless.
    """
    gap = acc_s2 - acc_s1
    if gap <= 0.0:
        return None
    return (acc_cascade - acc_s1) / gap


def percentile(values: Sequence[float], q: float) -> float:
    """q-th percentile (0–100) with linear interpolation, numpy-style.

    rank = q/100 * (n − 1); between two neighbors, weight by the rank
    fraction.
    """
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be in [0, 100], got {q!r}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (q / 100.0) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def latency_percentiles(values: Sequence[float]) -> dict[str, float]:
    """p50/p95/p99 required by DARU spec §5 (linear interpolation)."""
    return {
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def total_decode_tokens(records: Sequence[BenchmarkRecord]) -> int:
    """Sum of System 2 decode tokens (unrouted items count as 0)."""
    return sum(record.s2_decode_tokens or 0 for record in records)
