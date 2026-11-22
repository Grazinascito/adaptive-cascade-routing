"""Pure per-scenario aggregation (PR4): composes the evaluator metrics.

No HTTP: `summarize` takes a scenario's `BenchmarkRecord`s and returns the
`ScenarioSummary` with overall/slice accuracy, latency percentiles and the
decode-token total (DARU spec §5).
"""

from collections.abc import Sequence
from dataclasses import dataclass

from src.metrics.evaluator import (
    BenchmarkRecord,
    accuracy,
    latency_percentiles,
    slice_accuracy,
    total_decode_tokens,
)


@dataclass(frozen=True, slots=True)
class ScenarioSummary:
    n_items: int
    accuracy: float
    accuracy_head: float | None
    accuracy_tail: float | None
    routed_to_s2_count: int
    s2_call_rate: float
    decode_tokens_total: int
    s1_latency_percentiles_ms: dict[str, float]
    s2_latency_percentiles_ms: dict[str, float] | None = None
    s2_parse_failures: int = 0
    s2_fuzzy_category_matches: int = 0


def summarize(records: Sequence[BenchmarkRecord]) -> ScenarioSummary:
    """Aggregates the records of ONE scenario; S2 latencies from routed items only."""
    if not records:
        raise ValueError("records must not be empty")
    s2_latencies = [record.s2_latency_ms for record in records if record.s2_latency_ms is not None]
    return ScenarioSummary(
        n_items=len(records),
        accuracy=accuracy(records),
        accuracy_head=slice_accuracy(records, is_long_tail=False),
        accuracy_tail=slice_accuracy(records, is_long_tail=True),
        routed_to_s2_count=sum(1 for record in records if record.routed_to_s2),
        s2_call_rate=sum(1 for record in records if record.routed_to_s2) / len(records),
        decode_tokens_total=total_decode_tokens(records),
        s1_latency_percentiles_ms=latency_percentiles([record.s1_latency_ms for record in records]),
        s2_latency_percentiles_ms=(latency_percentiles(s2_latencies) if s2_latencies else None),
        s2_parse_failures=sum(1 for record in records if not record.s2_parse_ok),
        s2_fuzzy_category_matches=sum(1 for record in records if record.s2_category_fuzzy),
    )
