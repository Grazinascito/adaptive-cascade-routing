"""Testes das métricas puras do Componente G (src/metrics/summary.py).

Funções puras: entradas e saídas são valores — nenhum mock é necessário.
Corresponde às Stories 2 e 3 de docs/system2-evaluator-spec.md.
"""

import statistics
from typing import cast

import pytest

from src.data.loader import Label
from src.metrics.summary import (
    BenchmarkResult,
    Scenario,
    SummaryReport,
    accuracy,
    apgr,
    compute_summary_metrics,
    evaluate_kpis,
    latency_percentiles_ms,
    macro_f1,
)


def _record(
    query_id: int,
    *,
    scenario: Scenario = "system1_pure",
    alpha: float | None = None,
    total_latency_ms: float = 100.0,
    routed_to_system_2: bool = False,
    tokens_prefill: int = 0,
    tokens_decode: int = 0,
    prediction: Label = "Entailment",
    ground_truth: Label = "Entailment",
    system2_fallback_used: bool | None = None,
    json_valid: bool | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        query_id=query_id,
        scenario=scenario,
        alpha=alpha,
        replayed=False,
        system_1_latency_ms=total_latency_ms,
        system_2_latency_ms=None,
        total_latency_ms=total_latency_ms,
        confidence=0.9,
        entropy=0.1,
        routed_to_system_2=routed_to_system_2,
        tokens_prefill=tokens_prefill,
        tokens_decode=tokens_decode,
        prediction=prediction,
        ground_truth=ground_truth,
        is_correct=prediction == ground_truth,
        system2_fallback_used=system2_fallback_used,
        system2_error_fallback=False,
        json_valid=json_valid,
    )


def _summary(
    *,
    p95: float = 1000.0,
    tokens_decode_total: int = 1000,
    accuracy_value: float = 0.9,
    apgr_value: float | None = None,
) -> SummaryReport:
    return SummaryReport(
        scenario="cascade",
        alpha=0.85,
        n_requests=10,
        n_latency_samples=5,
        latency_percentiles_ms={"p50": p95 / 2.0, "p95": p95, "p99": p95},
        tokens_prefill_total=1000,
        tokens_decode_total=tokens_decode_total,
        accuracy=accuracy_value,
        f1_macro=accuracy_value,
        system2_call_rate=0.5,
        apgr=apgr_value,
        json_success_rate=1.0,
    )


def test_latency_percentiles_ms_returns_configured_percentiles() -> None:
    latencies = [float(value) for value in range(1, 101)]

    result = latency_percentiles_ms(latencies, (50, 95, 99))

    cutpoints = statistics.quantiles(latencies, n=100)
    assert result == {
        "p50": pytest.approx(cutpoints[49]),
        "p95": pytest.approx(cutpoints[94]),
        "p99": pytest.approx(cutpoints[98]),
    }


def test_latency_percentiles_ms_rejects_short_samples_and_out_of_range_percentiles() -> None:
    with pytest.raises(ValueError, match="at least two"):
        latency_percentiles_ms([10.0], (50,))
    with pytest.raises(ValueError, match=r"\[1, 99\]"):
        latency_percentiles_ms([10.0, 20.0], (0, 100))


def test_accuracy_and_macro_f1_handle_confusion_and_unpredicted_class() -> None:
    predictions: list[Label] = ["Entailment", "Entailment", "Contradiction", "Entailment"]
    ground_truths: list[Label] = ["Entailment", "Contradiction", "Contradiction", "Entailment"]

    assert accuracy(predictions, ground_truths) == pytest.approx(0.75)
    # Entailment: tp=2 fp=1 fn=0 → F1 0.8; Contradiction: tp=1 fp=0 fn=1 → F1 2/3.
    assert macro_f1(predictions, ground_truths) == pytest.approx((0.8 + 2.0 / 3.0) / 2.0)

    # Classe nunca prevista: Contradiction tem F1 0 sem divisão por zero.
    only_entailment: list[Label] = ["Entailment", "Entailment", "Entailment"]
    mixed_truth: list[Label] = ["Entailment", "Contradiction", "Entailment"]
    assert macro_f1(only_entailment, mixed_truth) == pytest.approx(0.4)


def test_apgr_recovers_gap_and_returns_zero_when_gap_is_zero() -> None:
    assert apgr(
        cascade_accuracy=0.92, system1_accuracy=0.80, system2_accuracy=0.96
    ) == pytest.approx(0.75)
    assert apgr(cascade_accuracy=0.9, system1_accuracy=0.8, system2_accuracy=0.8) == 0.0


def test_compute_summary_metrics_excludes_warmup_only_from_percentiles() -> None:
    latencies = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 10.0, 20.0, 30.0]
    records = [
        _record(
            query_id,
            total_latency_ms=latency,
            routed_to_system_2=query_id % 2 == 0,
            tokens_prefill=100,
            tokens_decode=50,
            prediction="Entailment",
            ground_truth="Entailment" if query_id < 4 else "Contradiction",
        )
        for query_id, latency in enumerate(latencies)
    ]

    report = compute_summary_metrics(records, warmup_discard=5)

    assert report["n_requests"] == 8
    assert report["n_latency_samples"] == 3
    cutpoints = statistics.quantiles([10.0, 20.0, 30.0], n=100)
    assert report["latency_percentiles_ms"]["p50"] == pytest.approx(cutpoints[49])
    # Acurácia usa TODAS as 8 requisições (4 corretas), não só as 3 pós-warmup.
    assert report["accuracy"] == pytest.approx(0.5)
    assert report["tokens_prefill_total"] == 800
    assert report["tokens_decode_total"] == 400
    assert report["system2_call_rate"] == pytest.approx(0.5)


def test_compute_summary_metrics_reports_json_success_rate_and_apgr_with_baselines() -> None:
    records = [
        _record(0, scenario="cascade", alpha=0.85, routed_to_system_2=True, json_valid=True),
        _record(1, scenario="cascade", alpha=0.85, routed_to_system_2=True, json_valid=True),
        _record(2, scenario="cascade", alpha=0.85, routed_to_system_2=True, json_valid=False),
        _record(3, scenario="cascade", alpha=0.85, routed_to_system_2=False, json_valid=None),
    ]

    report = compute_summary_metrics(
        records,
        warmup_discard=0,
        system1_accuracy=0.5,
        system2_accuracy=1.0,
    )

    assert report["json_success_rate"] == pytest.approx(2.0 / 3.0)
    assert report["accuracy"] == 1.0
    assert report["apgr"] == pytest.approx((1.0 - 0.5) / (1.0 - 0.5))

    without_baselines = compute_summary_metrics(records, warmup_discard=0)
    assert without_baselines["apgr"] is None


def test_compute_summary_metrics_rejects_mixed_scenarios() -> None:
    with pytest.raises(ValueError, match="single scenario"):
        compute_summary_metrics(
            [_record(0, scenario="system1_pure"), _record(1, scenario="cascade")],
            warmup_discard=0,
        )


def test_evaluate_kpis_flags_each_threshold() -> None:
    cascade = _summary(p95=250.0, tokens_decode_total=300, apgr_value=0.90)
    system2 = _summary(p95=1000.0, tokens_decode_total=1000)
    kpis = {"p95_reduction": 0.50, "token_savings": 0.60, "apgr_target": 0.95}

    report = evaluate_kpis(cascade, system2, kpis)

    assert report["p95_reduction"] == (True, pytest.approx(0.75))
    assert report["token_savings"] == (True, pytest.approx(0.70))
    assert report["apgr"] == (False, pytest.approx(0.90))

    with pytest.raises(ValueError, match="p95"):
        broken = dict(system2)
        broken["latency_percentiles_ms"] = {"p95": 0.0}
        evaluate_kpis(cascade, cast("SummaryReport", broken), kpis)
