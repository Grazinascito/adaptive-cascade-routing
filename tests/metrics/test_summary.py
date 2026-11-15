"""Testes das métricas puras do Componente G (src/metrics/summary.py).

Funções puras: entradas e saídas são valores — nenhum mock é necessário.
Corresponde às Stories 2 e 3 de docs/system2-evaluator-spec.md e à Story 6
de docs/dataset-pivot-ncm-spec.md (subconjuntos head/tail e gates novos).
"""

import statistics
from typing import cast

import pytest

from src.data.loader import Subset
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
    subset: Subset = "head",
    total_latency_ms: float = 100.0,
    routed_to_system_2: bool = False,
    tokens_prefill: int = 0,
    tokens_decode: int = 0,
    prediction: str = "22071090",
    ground_truth: str = "22071090",
    system2_fallback_used: bool | None = None,
    json_valid: bool | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        query_id=query_id,
        scenario=scenario,
        alpha=alpha,
        replayed=False,
        subset=subset,
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
    accuracy_head: float | None = 0.95,
    accuracy_tail: float | None = 0.6,
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
        accuracy_head=accuracy_head,
        accuracy_tail=accuracy_tail,
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
    predictions = ["22071090", "22071090", "22042100", "22071090"]
    ground_truths = ["22071090", "22042100", "22042100", "22071090"]

    assert accuracy(predictions, ground_truths) == pytest.approx(0.75)
    # 22071090: tp=2 fp=1 fn=0 → F1 0.8; 22042100: tp=1 fp=0 fn=1 → F1 2/3.
    assert macro_f1(predictions, ground_truths) == pytest.approx((0.8 + 2.0 / 3.0) / 2.0)

    # Classe nunca prevista tem F1 0 sem divisão por zero.
    only_first = ["22071090", "22071090", "22071090"]
    mixed_truth = ["22071090", "22042100", "22071090"]
    assert macro_f1(only_first, mixed_truth) == pytest.approx(0.4)


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
            prediction="22071090",
            ground_truth="22071090" if query_id < 4 else "22042100",
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


def test_compute_summary_metrics_reports_head_and_tail_accuracy_separately() -> None:
    records = [
        _record(0, subset="head", prediction="22071090", ground_truth="22071090"),
        _record(1, subset="head", prediction="22071090", ground_truth="22071090"),
        _record(2, subset="head", prediction="22042100", ground_truth="22071090"),
        _record(3, subset="tail", prediction="22042100", ground_truth="22071090"),
        _record(4, subset="tail", prediction="33072010", ground_truth="33072010"),
    ]

    report = compute_summary_metrics(records, warmup_discard=0)

    assert report["accuracy"] == pytest.approx(0.6)
    assert report["accuracy_head"] == pytest.approx(2.0 / 3.0)
    assert report["accuracy_tail"] == pytest.approx(0.5)


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


def test_evaluate_kpis_flags_each_pivot_gate() -> None:
    cascade = _summary(p95=250.0, tokens_decode_total=300, accuracy_tail=0.75)
    system1 = _summary(accuracy_value=0.60, accuracy_head=0.65, accuracy_tail=0.50)
    system2 = _summary(p95=1000.0, tokens_decode_total=1000, accuracy_tail=0.90)
    kpis = {
        "p95_reduction": 0.50,
        "token_savings": 0.60,
        "tail_gap_pp": 10,
        "recovery_share": 0.50,
    }

    report = evaluate_kpis(cascade, system1, system2, kpis)

    assert report["p95_reduction"] == (True, pytest.approx(0.75))
    assert report["token_savings"] == (True, pytest.approx(0.70))
    # Gap na cauda: (0.90 - 0.50) * 100 = 40 pp >= 10 → PASS.
    assert report["tail_gap_pp"] == (True, pytest.approx(40.0))
    # Recuperação: (0.75 - 0.50) / 0.40 = 0.625 >= 0.5 e cascata >= S1 → PASS.
    assert report["recovery"] == (True, pytest.approx(0.625))


def test_evaluate_kpis_fails_recovery_when_cascade_drops_below_system1_overall() -> None:
    cascade = _summary(p95=250.0, tokens_decode_total=300, accuracy_value=0.55, accuracy_tail=0.75)
    system1 = _summary(accuracy_value=0.60, accuracy_head=0.65, accuracy_tail=0.50)
    system2 = _summary(p95=1000.0, tokens_decode_total=1000, accuracy_tail=0.90)
    kpis = {"p95_reduction": 0.50, "token_savings": 0.60, "tail_gap_pp": 10, "recovery_share": 0.50}

    report = evaluate_kpis(cascade, system1, system2, kpis)

    # Recuperação na cauda é 0.625, mas a cascata (0.55) caiu abaixo do S1
    # puro (0.60) no conjunto completo → FAIL.
    assert report["recovery"] == (False, pytest.approx(0.625))


def test_evaluate_kpis_requires_tail_accuracy_in_every_summary() -> None:
    cascade = _summary()
    system1 = _summary(accuracy_value=0.6, accuracy_tail=None)
    system2 = _summary(p95=1000.0, tokens_decode_total=1000, accuracy_tail=0.9)
    kpis = {"p95_reduction": 0.50, "token_savings": 0.60, "tail_gap_pp": 10, "recovery_share": 0.5}

    with pytest.raises(ValueError, match="tail"):
        evaluate_kpis(cascade, system1, system2, kpis)

    with pytest.raises(ValueError, match="p95"):
        broken = dict(system2)
        broken["latency_percentiles_ms"] = {"p95": 0.0}
        evaluate_kpis(cascade, system1, cast("SummaryReport", broken), kpis)
