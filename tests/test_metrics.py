"""Pure metrics (PR4): per-slice accuracy, APGR, percentiles and token totals."""

from dataclasses import fields

import pytest

from src.metrics.evaluator import (
    BenchmarkRecord,
    accuracy,
    apgr,
    latency_percentiles,
    percentile,
    slice_accuracy,
    total_decode_tokens,
)
from src.metrics.summary import summarize


def _record(
    *,
    item_id: str,
    is_long_tail: bool,
    is_correct: bool,
    routed: bool = False,
    decode_tokens: int | None = None,
    s2_latency_ms: float | None = None,
    s1_latency_ms: float = 10.0,
) -> BenchmarkRecord:
    return BenchmarkRecord(
        item_id=item_id,
        raw_text=f"item text {item_id}",
        canonical_category="MERCEARIA",
        is_long_tail=is_long_tail,
        s1_prediction="MERCEARIA",
        s1_p_top=0.9,
        s1_entropy=0.5,
        s1_latency_ms=s1_latency_ms,
        routed_to_s2=routed,
        s2_prediction="MERCEARIA" if routed else None,
        s2_reasoning="short rule" if routed else None,
        s2_revisor_insight="insight" if routed else None,
        s2_decode_tokens=decode_tokens,
        s2_latency_ms=s2_latency_ms,
        final_prediction="MERCEARIA",
        is_correct=is_correct,
    )


@pytest.fixture
def records() -> list[BenchmarkRecord]:
    return [
        _record(item_id="1", is_long_tail=False, is_correct=True),
        _record(item_id="2", is_long_tail=False, is_correct=True),
        _record(item_id="3", is_long_tail=False, is_correct=False),
        _record(item_id="4", is_long_tail=False, is_correct=False),
        _record(item_id="5", is_long_tail=True, is_correct=True),
        _record(item_id="6", is_long_tail=True, is_correct=False),
    ]


def test_benchmark_record_has_exactly_the_spec_fields() -> None:
    expected = [
        "item_id",
        "raw_text",
        "canonical_category",
        "is_long_tail",
        "s1_prediction",
        "s1_p_top",
        "s1_entropy",
        "s1_latency_ms",
        "routed_to_s2",
        "s2_prediction",
        "s2_reasoning",
        "s2_revisor_insight",
        "s2_decode_tokens",
        "s2_latency_ms",
        "final_prediction",
        "is_correct",
    ]
    assert [field.name for field in fields(BenchmarkRecord)] == expected


def test_accuracy_overall_and_per_slice(records: list[BenchmarkRecord]) -> None:
    assert accuracy(records) == pytest.approx(3 / 6)
    assert slice_accuracy(records, is_long_tail=False) == pytest.approx(2 / 4)
    assert slice_accuracy(records, is_long_tail=True) == pytest.approx(1 / 2)


def test_slice_accuracy_none_when_slice_is_empty(records: list[BenchmarkRecord]) -> None:
    head_only = [record for record in records if not record.is_long_tail]

    assert slice_accuracy(head_only, is_long_tail=True) is None


def test_accuracy_on_empty_records_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        accuracy([])


def test_apgr_known_value() -> None:
    assert apgr(0.60, 0.80, 0.75) == pytest.approx(0.75)


def test_apgr_zero_denominator_returns_none() -> None:
    assert apgr(0.70, 0.70, 0.90) is None


def test_apgr_negative_gap_returns_none() -> None:
    assert apgr(0.80, 0.70, 0.85) is None


def test_percentiles_on_one_to_ten() -> None:
    values = [float(value) for value in range(1, 11)]

    result = latency_percentiles(values)

    assert result["p50"] == pytest.approx(5.5)
    assert result["p95"] == pytest.approx(9.55)
    assert result["p99"] == pytest.approx(9.91)


def test_percentile_linear_interpolation_matches_manual_formula() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 25) == pytest.approx(1.75)
    assert percentile([42.0], 95) == 42.0
    assert percentile([3.0, 1.0, 2.0], 50) == pytest.approx(2.0)


def test_percentile_rejects_empty_and_out_of_range_q() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        percentile([], 50)
    with pytest.raises(ValueError, match="q must be in"):
        percentile([1.0, 2.0], 101.0)


def test_total_decode_tokens_sums_and_treats_none_as_zero(
    records: list[BenchmarkRecord],
) -> None:
    routed = [_record(item_id="7", is_long_tail=True, is_correct=True, decode_tokens=120)]
    mixed = records + routed

    assert total_decode_tokens(mixed) == 120


def test_summarize_aggregates_counts_accuracies_tokens_and_percentiles(
    records: list[BenchmarkRecord],
) -> None:
    routed = [
        _record(
            item_id="7",
            is_long_tail=True,
            is_correct=True,
            routed=True,
            decode_tokens=100,
            s2_latency_ms=200.0,
        ),
        _record(
            item_id="8",
            is_long_tail=True,
            is_correct=True,
            routed=True,
            decode_tokens=100,
            s2_latency_ms=400.0,
        ),
    ]
    scenario = records + routed

    summary = summarize(scenario)

    assert summary.n_items == 8
    assert summary.accuracy == pytest.approx(5 / 8)
    assert summary.accuracy_head == pytest.approx(2 / 4)
    assert summary.accuracy_tail == pytest.approx(3 / 4)
    assert summary.routed_to_s2_count == 2
    assert summary.s2_call_rate == pytest.approx(2 / 8)
    assert summary.decode_tokens_total == 200
    assert summary.s2_latency_percentiles_ms is not None
    assert summary.s2_latency_percentiles_ms["p50"] == pytest.approx(300.0)


def test_summarize_without_routed_items_has_no_s2_percentiles(
    records: list[BenchmarkRecord],
) -> None:
    summary = summarize(records)

    assert summary.routed_to_s2_count == 0
    assert summary.decode_tokens_total == 0
    assert summary.s2_latency_percentiles_ms is None


def test_summarize_on_empty_records_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        summarize([])


def test_summarize_counts_parse_failures_and_fuzzy_matches() -> None:
    from src.metrics.evaluator import BenchmarkRecord
    from src.metrics.summary import summarize

    def record(**overrides: object) -> BenchmarkRecord:
        base: dict[str, object] = dict(
            item_id="1",
            raw_text="X",
            canonical_category="MERCEARIA",
            is_long_tail=False,
            s1_prediction="MERCEARIA",
            s1_p_top=0.9,
            s1_entropy=0.5,
            s1_latency_ms=1.0,
            routed_to_s2=False,
            s2_prediction="",
            s2_reasoning="",
            s2_revisor_insight="",
            s2_decode_tokens=0,
            s2_latency_ms=None,
            final_prediction="MERCEARIA",
            is_correct=True,
            s2_parse_ok=True,
            s2_category_fuzzy=False,
        )
        base.update(overrides)
        return BenchmarkRecord(**base)  # type: ignore[arg-type]

    summary = summarize(
        [
            record(),
            record(s2_parse_ok=False),
            record(s2_category_fuzzy=True),
        ]
    )
    assert summary.s2_parse_failures == 1
    assert summary.s2_fuzzy_category_matches == 1
