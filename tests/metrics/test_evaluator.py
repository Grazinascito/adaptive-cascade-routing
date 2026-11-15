"""Testes do BenchmarkHarness do Componente G pós-pivot (src/metrics/evaluator.py).

O vLLM real nunca é chamado: as respostas vêm de httpx.MockTransport no
limite da rede, imitando o contrato do /v1/chat/completions para os dois
engines (System 1: geração curta com logprobs por token; System 2: CoT +
JSON). O stub distingue os engines pelo corpo da request: o System 1 manda
messages[0].role == "system" e logprobs; o System 2 manda só user.
"""

import json
import logging
import math
from typing import Any

import httpx
import matplotlib

matplotlib.use("Agg")  # antes do pyplot: CI não tem display

import pytest

from src.data.loader import NCMEvalSample, NCMItem
from src.engine.system1 import System1Output
from src.engine.system2 import System2Output
from src.metrics import evaluator
from src.metrics.evaluator import (
    ALPHA_SWEEP,
    BenchmarkHarness,
    BenchmarkRun,
    SampleMeasurement,
    _build_result,
    _optimal_alpha,
    confidence_degeneration_share,
    main,
    plot_tradeoff_curve,
    save_raw_results,
    sweep_alpha,
)
from src.metrics.summary import compute_summary_metrics

_S1_USAGE = {"prompt_tokens": 210, "completion_tokens": 8}
_S2_USAGE = {"prompt_tokens": 120, "completion_tokens": 80}
_S2_CONTENT = json.dumps(
    {"reasoning": "Aguardente de cana: capítulo 22.", "ncm_code": "22084000"},
    ensure_ascii=False,
)


def _samples() -> list[NCMEvalSample]:
    return [
        NCMEvalSample(
            item_id=0, name="ITEM HEAD ZERO", ncm_code="22071090", code_freq=60, subset="head"
        ),
        NCMEvalSample(
            item_id=1, name="ITEM HEAD UM", ncm_code="22042100", code_freq=55, subset="head"
        ),
        NCMEvalSample(
            item_id=2, name="ITEM TAIL DOIS", ncm_code="33072010", code_freq=3, subset="tail"
        ),
        NCMEvalSample(
            item_id=3, name="ITEM TAIL TRES", ncm_code="96190000", code_freq=2, subset="tail"
        ),
    ]


def _stub_client(
    requests: list[dict[str, Any]],
    *,
    s1_confidence: float = 0.9,
    s1_code_for: dict[str, str] | None = None,
    chat_status_after: int | None = None,
) -> httpx.Client:
    """Client de teste para os dois engines.

    O System 1 devolve o código gold do item (ou o mapa `s1_code_for`) com
    confiança geométrica `s1_confidence` (ln(p) igual em 8 tokens); o System
    2 devolve `_S2_CONTENT` (código 22084000) e falha com 500 depois de
    `chat_status_after` chamadas bem-sucedidas (teste de fallback).
    """
    chat_calls = 0
    gold = {sample["name"]: sample["ncm_code"] for sample in _samples()}
    token_logprob = math.log(s1_confidence)

    def _s1_response(request_body: dict) -> httpx.Response:
        name = request_body["messages"][1]["content"]
        code = (s1_code_for or {}).get(name, gold.get(name, "22071090"))
        content_logprobs = [{"token": digit, "logprob": token_logprob} for digit in code]
        return httpx.Response(
            200,
            json={
                "id": "chat-1",
                "object": "chat.completion",
                "created": 0,
                "model": "Qwen/Qwen3-4B-AWQ",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": code},
                        "finish_reason": "stop",
                        "logprobs": {"content": content_logprobs},
                    }
                ],
                "usage": _S1_USAGE,
            },
        )

    def _s2_response() -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chat-2",
                "object": "chat.completion",
                "created": 0,
                "model": "Qwen/Qwen3-4B-AWQ",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": _S2_CONTENT},
                        "finish_reason": "stop",
                    }
                ],
                "usage": _S2_USAGE,
            },
        )

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_calls
        request_body = json.loads(request.content)
        requests.append({"url": str(request.url), "body": request_body})
        if request_body["messages"][0]["role"] == "system":
            return _s1_response(request_body)
        if chat_status_after is not None and chat_calls >= chat_status_after:
            return httpx.Response(500, json={"error": "boom"})
        chat_calls += 1
        return _s2_response()

    return httpx.Client(transport=httpx.MockTransport(handler))


def _run_harness(
    *,
    alpha: float = 0.85,
    s1_confidence: float = 0.9,
    s1_code_for: dict[str, str] | None = None,
    chat_status_after: int | None = None,
    label_digits: int = 8,
) -> tuple[BenchmarkRun, list[dict[str, Any]]]:
    requests: list[dict[str, Any]] = []
    harness = BenchmarkHarness(
        _samples(),
        label_digits=label_digits,
        client=_stub_client(
            requests,
            s1_confidence=s1_confidence,
            s1_code_for=s1_code_for,
            chat_status_after=chat_status_after,
        ),
    )
    try:
        return harness.run_full_benchmark(confidence_threshold=alpha), requests
    finally:
        harness.close()


def _s1_calls(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in requests if r["body"]["messages"][0]["role"] == "system"]


def _s2_calls(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in requests if r["body"]["messages"][0]["role"] != "system"]


def test_run_full_benchmark_records_every_item_in_all_three_scenarios() -> None:
    run, requests = _run_harness()

    assert set(run["records"]) == {"system1_pure", "system2_pure", "cascade"}
    for records in run["records"].values():
        assert [record["query_id"] for record in records] == [0, 1, 2, 3]
        assert [record["subset"] for record in records] == ["head", "head", "tail", "tail"]
    assert set(run["measurements"]) == {0, 1, 2, 3}

    assert len(_s1_calls(requests)) == 8  # passadas system1_pure + cascade
    assert len(_s2_calls(requests)) == 4  # só a passada system2_pure (0.9 >= 0.85)

    for record in run["records"]["system1_pure"]:
        assert record["prediction"] == record["ground_truth"]
        assert record["system_2_latency_ms"] is None
        assert not record["routed_to_system_2"]
        # Fast-path também custa tokens (usage do chat completions).
        assert record["tokens_prefill"] == 210 and record["tokens_decode"] == 8
    for record in run["records"]["system2_pure"]:
        assert record["prediction"] == "22084000"
        assert record["system_1_latency_ms"] is None
        assert record["confidence"] is None and record["entropy"] is None
        assert record["routed_to_system_2"]
        assert record["tokens_prefill"] == 120 and record["tokens_decode"] == 80
        assert record["json_valid"] and not record["system2_fallback_used"]
    for record in run["records"]["cascade"]:  # confiança 0.9 >= α 0.85 → fast-path
        assert not record["routed_to_system_2"]
        assert record["prediction"] == record["ground_truth"]
        assert record["alpha"] == 0.85 and not record["replayed"]
        # Tokens da cascata no fast-path = só os do System 1.
        assert record["tokens_prefill"] == 210 and record["tokens_decode"] == 8


def test_cascade_escalation_sums_tokens_of_both_engines() -> None:
    run, requests = _run_harness(s1_confidence=0.5)

    assert len(_s2_calls(requests)) == 8  # system2_pure + escaladas da cascata
    for record in run["records"]["cascade"]:
        assert record["routed_to_system_2"]
        assert record["prediction"] == "22084000"
        assert record["system_2_latency_ms"] is not None
        # System 1 (210/8) + System 2 (120/80) — ambos contam no custo.
        assert record["tokens_prefill"] == 330
        assert record["tokens_decode"] == 88
        assert record["json_valid"] and not record["system2_error_fallback"]


def test_label_digits_truncates_predictions_and_ground_truths_identically() -> None:
    # O S1 prevê 22071000 para o item gold 22071090: errado em 8 dígitos,
    # correto no nível de posição (4 dígitos).
    run, _requests = _run_harness(s1_code_for={"ITEM HEAD ZERO": "22071000"}, label_digits=4)

    record = next(record for record in run["records"]["system1_pure"] if record["query_id"] == 0)
    assert record["prediction"] == "2207"
    assert record["ground_truth"] == "2207"
    assert record["is_correct"]

    run_full = _run_harness(s1_code_for={"ITEM HEAD ZERO": "22071000"}, label_digits=8)[0]
    record_full = next(
        record for record in run_full["records"]["system1_pure"] if record["query_id"] == 0
    )
    assert record_full["prediction"] == "22071000"
    assert not record_full["is_correct"]


def test_build_result_keeps_already_truncated_predictions_as_is() -> None:
    sample = _samples()[0]
    record = _build_result(
        sample,
        label_digits=8,
        scenario="system1_pure",
        alpha=None,
        replayed=False,
        system1=None,
        system2=None,
        prediction="2207",
    )
    # Predição já truncada (não-8-dígitos) passa como veio: a truncagem só
    # normaliza códigos completos de 8 dígitos.
    assert record["prediction"] == "2207"
    assert not record["is_correct"]


def test_system_two_http_error_falls_back_to_system1_prediction_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.ERROR):
        # 4 chamadas bem-sucedidas (passada system2_pure); as escaladas da
        # cascata (confiança 0.5 < α) recebem 500.
        run, _requests = _run_harness(s1_confidence=0.5, chat_status_after=4)

    for record in run["records"]["cascade"]:
        assert record["system2_error_fallback"]
        assert record["prediction"] == record["ground_truth"]  # código do System 1
        assert record["system_2_latency_ms"] is None
        assert record["system2_fallback_used"] is None and record["json_valid"] is None
    assert "system 2 failed" in caplog.text


def _s1_output(confidence: float) -> System1Output:
    return System1Output(
        code="22071090",
        confidence=confidence,
        entropy=0.1,
        valid_code=True,
        prompt_tokens=210,
        completion_tokens=8,
        latency_ms=10.0,
    )


def _s2_output() -> System2Output:
    return System2Output(
        reasoning="raciocínio",
        ncm_code="22084000",
        prompt_tokens=100,
        completion_tokens=50,
        latency_ms=1000.0,
        fallback_used=False,
    )


def _synthetic_run(confidences: list[float]) -> BenchmarkRun:
    samples = _samples()[: len(confidences)]
    measurements = {
        sample["item_id"]: SampleMeasurement(
            sample=sample,
            system1=_s1_output(confidence),
            system2=_s2_output(),
        )
        for sample, confidence in zip(samples, confidences, strict=True)
    }
    return BenchmarkRun(
        samples=samples,
        records={"system1_pure": [], "system2_pure": [], "cascade": []},
        measurements=measurements,
    )


def test_sweep_alpha_call_rate_is_monotonic_across_thresholds() -> None:
    run = _synthetic_run([0.99, 0.80, 0.55, 0.50])

    sweep = sweep_alpha(run)

    assert list(sweep) == list(ALPHA_SWEEP)
    rates = [
        sum(1 for record in records if record["routed_to_system_2"]) / len(records)
        for records in sweep.values()
    ]
    assert rates == sorted(rates)  # Story 3: α maior → taxa de S2 não-decrescente
    for records in sweep.values():
        assert all(record["replayed"] and record["scenario"] == "cascade" for record in records)


def test_sweep_alpha_replays_route_fast_path_on_confidence_equal_to_alpha() -> None:
    run = _synthetic_run([0.85])

    sweep = sweep_alpha(run, alphas=(0.85,))

    records = sweep[0.85]
    assert len(records) == 1 and not records[0]["routed_to_system_2"]  # >= é fast-path
    assert records[0]["prediction"] == "22071090" and records[0]["tokens_decode"] == 8


def test_confidence_degeneration_share_counts_high_confidence_items() -> None:
    assert confidence_degeneration_share(_synthetic_run([0.99, 0.97])) == pytest.approx(1.0)
    assert confidence_degeneration_share(_synthetic_run([0.99, 0.60])) == pytest.approx(0.5)


def test_optimal_alpha_prefers_lowest_call_rate_among_winners() -> None:
    from src.metrics.summary import SummaryReport

    def _summary(alpha: float, accuracy: float, rate: float) -> SummaryReport:
        return SummaryReport(
            scenario="cascade",
            alpha=alpha,
            n_requests=4,
            n_latency_samples=4,
            latency_percentiles_ms={"p50": 10.0, "p95": 50.0, "p99": 60.0},
            tokens_prefill_total=100,
            tokens_decode_total=100,
            accuracy=accuracy,
            accuracy_head=accuracy,
            accuracy_tail=accuracy,
            f1_macro=accuracy,
            system2_call_rate=rate,
            apgr=None,
            json_success_rate=None,
        )

    summaries = {0.5: _summary(0.5, 0.9, 0.8), 0.85: _summary(0.85, 0.9, 0.4)}
    alpha, summary = _optimal_alpha(summaries, system1_accuracy=0.8)
    assert alpha == 0.85 and summary["system2_call_rate"] == 0.4

    # Nenhum α vence o S1 puro → maior acurácia.
    losing = {0.5: _summary(0.5, 0.7, 0.8), 0.85: _summary(0.85, 0.75, 0.4)}
    alpha, summary = _optimal_alpha(losing, system1_accuracy=0.8)
    assert alpha == 0.85


def test_save_raw_results_writes_meta_scenarios_sweep_summaries_and_kpis(tmp_path) -> None:
    run, _requests = _run_harness()
    sweep = sweep_alpha(run)
    summaries = {
        "system1_pure": compute_summary_metrics(run["records"]["system1_pure"], warmup_discard=0),
        "cascade": compute_summary_metrics(
            run["records"]["cascade"], warmup_discard=0, system1_accuracy=0.5, system2_accuracy=0.5
        ),
    }
    path = tmp_path / "raw_benchmark_data.json"

    save_raw_results(
        path,
        meta={"model": "Qwen/Qwen3-4B-AWQ", "frozen_samples": 4},
        scenarios=run["records"],
        alpha_sweep={str(alpha): records for alpha, records in sweep.items()},
        summaries=summaries,
        kpis=None,
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data) == {"meta", "scenarios", "alpha_sweep", "summaries", "kpis"}
    assert data["meta"]["frozen_samples"] == 4
    for scenario in ("system1_pure", "system2_pure", "cascade"):
        assert [record["query_id"] for record in data["scenarios"][scenario]] == [0, 1, 2, 3]
        assert all(record["subset"] in ("head", "tail") for record in data["scenarios"][scenario])
    assert set(data["alpha_sweep"]) == {"0.5", "0.7", "0.85", "0.9", "0.95"}


def test_plot_tradeoff_curve_writes_non_empty_png(tmp_path) -> None:
    run, _requests = _run_harness(s1_confidence=0.5)
    sweep = sweep_alpha(run)
    s1 = compute_summary_metrics(run["records"]["system1_pure"], warmup_discard=0)
    s2 = compute_summary_metrics(run["records"]["system2_pure"], warmup_discard=0)
    sweep_summaries = {
        alpha: compute_summary_metrics(
            records,
            warmup_discard=0,
            system1_accuracy=s1["accuracy"],
            system2_accuracy=s2["accuracy"],
        )
        for alpha, records in sweep.items()
    }
    path = tmp_path / "latency_cost_tradeoff.png"

    plot_tradeoff_curve(path, run, sweep_summaries)

    assert path.exists() and path.stat().st_size > 0


def test_main_runs_benchmark_from_config_against_stub_server(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    config_path = tmp_path / "benchmark_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset: {name: yurifacanha/NCM-RAFT-samall, split: train, "
                "frozen_samples: 4, seed: 42, head_share: 0.5, "
                "head_min_code_freq: 50, label_digits: 8}",
                "model: {id: Qwen/Qwen3-4B-AWQ, quantization: awq}",
                'serving: {base_url: "http://127.0.0.1:8000/v1", port: 8000, prefix_caching: true}',
                "system1: {endpoint: /v1/chat/completions, max_tokens: 12, top_logprobs: 20}",
                "system2: {endpoint: /v1/chat/completions, max_tokens: 256}",
                "router: {alpha: 0.85}",
                "metrics: {percentiles: [50, 95, 99]}",
                "kpis: {p95_reduction: 0.50, token_savings: 0.60, tail_gap_pp: 10, "
                "recovery_share: 0.50}",
            ]
        ),
        encoding="utf-8",
    )
    # A frequência de um código conta nomes únicos com esse código: o pool
    # precisa de >= 50 nomes distintos por código head para cruzar o
    # head_min_code_freq=50 do config.
    pool: list[NCMItem] = []
    item_id = 0
    for code in ("22071090", "22042100"):
        for i in range(55):
            pool.append(
                NCMItem(item_id=item_id, name=f"ITEM HEAD {code[-2:]} N{i:02d} AAA", ncm_code=code)
            )
            item_id += 1
    for code in ("33072010", "96190000"):
        for i in range(2):
            pool.append(
                NCMItem(item_id=item_id, name=f"ITEM TAIL {code[-2:]} N{i:02d} BBB", ncm_code=code)
            )
            item_id += 1
    monkeypatch.setattr(evaluator, "load_ncm_pool", lambda _name: (pool, len(pool)))
    monkeypatch.setattr(evaluator, "health_check", lambda _url: True)
    requests: list[dict[str, Any]] = []
    results_dir = tmp_path / "results"

    exit_code = main(
        [f"--config={config_path}"],
        client=_stub_client(requests),
        results_dir=results_dir,
    )

    assert exit_code == 0
    assert (results_dir / "raw_benchmark_data.json").exists()
    assert (results_dir / "latency_cost_tradeoff.png").exists()
    output = capsys.readouterr().out
    assert "SC-A" in output and "SC-B" in output and "SC-C" in output
    assert "Gate F" in output and "α ótimo" in output
    assert "Taxa de fallback" in output
    meta = json.loads((results_dir / "raw_benchmark_data.json").read_text(encoding="utf-8"))
    assert meta["meta"]["pool_stats"]["n_rows"] == 114  # 55×2 head + 2×2 tail
    assert meta["meta"]["label_digits"] == 8
