"""Testes do BenchmarkHarness do Componente G (src/metrics/evaluator.py).

O vLLM real nunca é chamado: as respostas vêm de httpx.MockTransport no
limite da rede, imitando os contratos reais do /v1/systemone (probabilidades
por opção) e do /v1/chat/completions (CoT + JSON + usage). Corresponde às
Stories 2 e 3 de docs/system2-evaluator-spec.md.
"""

import json
import logging

import httpx
import matplotlib

matplotlib.use("Agg")  # antes do pyplot: CI não tem display

import pytest

from src.data.loader import ASSIN2Sample, Label
from src.engine.system1 import System1Output
from src.engine.system2 import System2Output
from src.metrics import evaluator
from src.metrics.evaluator import (
    ALPHA_SWEEP,
    BenchmarkHarness,
    BenchmarkRun,
    SampleMeasurement,
    main,
    plot_tradeoff_curve,
    save_raw_results,
    sweep_alpha,
)
from src.metrics.summary import compute_summary_metrics

_CHAT_CONTENT = '{"reasoning": "A premissa apoia a hipótese.", "label": "Contradiction"}'
_USAGE = {"prompt_tokens": 120, "completion_tokens": 80}


def _samples(n: int) -> list[ASSIN2Sample]:
    labels: list[Label] = ["Entailment", "Contradiction"]
    return [
        ASSIN2Sample(
            id=index,
            premise=f"Premissa {index}.",
            hypothesis=f"Hipótese {index}.",
            label=labels[index % 2],
            original_score=3.0,
        )
        for index in range(n)
    ]


def _systemone_response(probabilities: dict[str, float]) -> httpx.Response:
    top = max(probabilities, key=lambda key: probabilities[key])
    return httpx.Response(
        200,
        json={
            "id": "decision-1",
            "object": "structured_decision",
            "created": 0,
            "model": "Qwen/Qwen3-4B-AWQ",
            "answers": {
                "nli_label": {
                    "type": "choice",
                    "choice": top,
                    "probabilities": probabilities,
                    "confidence": probabilities[top],
                }
            },
        },
    )


def _chat_response(content: str, usage: dict[str, int]) -> httpx.Response:
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
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        },
    )


def _stub_client(
    requests: list[dict[str, object]],
    *,
    probabilities: dict[str, float] | None = None,
    chat_status_after: int | None = None,
) -> httpx.Client:
    """Client de teste: /systemone devolve `probabilities`; /chat/completions
    devolve `_CHAT_CONTENT` (label Contradiction) e falha com 500 depois de
    `chat_status_after` chamadas bem-sucedidas (para o teste de fallback)."""

    chat_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_calls
        requests.append({"url": str(request.url), "body": json.loads(request.content)})
        if request.url.path.endswith("/systemone"):
            return _systemone_response(probabilities or {"Entailment": 0.9, "Contradiction": 0.1})
        if chat_status_after is not None and chat_calls >= chat_status_after:
            return httpx.Response(500, json={"error": "boom"})
        chat_calls += 1
        return _chat_response(_CHAT_CONTENT, _USAGE)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _run_harness(
    n: int,
    *,
    alpha: float = 0.85,
    probabilities: dict[str, float] | None = None,
    chat_status_after: int | None = None,
) -> tuple[BenchmarkRun, list[dict[str, object]]]:
    requests: list[dict[str, object]] = []
    harness = BenchmarkHarness(
        _samples(n),
        client=_stub_client(
            requests, probabilities=probabilities, chat_status_after=chat_status_after
        ),
    )
    try:
        return harness.run_full_benchmark(confidence_threshold=alpha), requests
    finally:
        harness.close()


def _s1_output(confidence: float) -> System1Output:
    return System1Output(
        top_choice="Entailment",
        top_probability=confidence,
        logprobs={},
        probabilities={"Entailment": confidence, "Contradiction": 1.0 - confidence},
        entropy=0.0,
        latency_ms=10.0,
    )


def _s2_output() -> System2Output:
    return System2Output(
        reasoning="raciocínio",
        label="Contradiction",
        prompt_tokens=100,
        completion_tokens=50,
        latency_ms=1000.0,
        fallback_used=False,
    )


def test_run_full_benchmark_records_every_query_id_in_all_three_scenarios() -> None:
    run, requests = _run_harness(10)

    assert set(run["records"]) == {"system1_pure", "system2_pure", "cascade"}
    for records in run["records"].values():
        assert len(records) == 10
        assert [record["query_id"] for record in records] == [
            sample["id"] for sample in run["samples"]
        ]
    assert set(run["measurements"]) == {sample["id"] for sample in run["samples"]}

    systemone_calls = [r for r in requests if str(r["url"]).endswith("/systemone")]
    chat_calls = [r for r in requests if str(r["url"]).endswith("/chat/completions")]
    assert len(systemone_calls) == 20  # passadas system1_pure + cascade
    assert len(chat_calls) == 10  # só a passada system2_pure (0.9 >= 0.85)

    for record in run["records"]["system1_pure"]:
        assert record["prediction"] == "Entailment"
        assert record["system_2_latency_ms"] is None
        assert not record["routed_to_system_2"]
        assert record["tokens_prefill"] == 0 and record["tokens_decode"] == 0
    for record in run["records"]["system2_pure"]:
        assert record["prediction"] == "Contradiction"
        assert record["system_1_latency_ms"] is None
        assert record["confidence"] is None and record["entropy"] is None
        assert record["routed_to_system_2"]
        assert record["tokens_prefill"] == 120 and record["tokens_decode"] == 80
        assert record["json_valid"] and not record["system2_fallback_used"]
    for record in run["records"]["cascade"]:  # confiança 0.9 >= α 0.85 → fast-path
        assert not record["routed_to_system_2"]
        assert record["prediction"] == "Entailment"
        assert record["alpha"] == 0.85 and not record["replayed"]


def test_cascade_fast_path_uses_system1_top_choice_and_skips_system_two() -> None:
    run, requests = _run_harness(2, probabilities={"Entailment": 0.95, "Contradiction": 0.05})

    chat_calls = [r for r in requests if str(r["url"]).endswith("/chat/completions")]
    assert len(chat_calls) == 2  # exatamente a passada system2_pure
    for record in run["records"]["cascade"]:
        assert record["system_2_latency_ms"] is None
        assert record["prediction"] == "Entailment"
        assert record["tokens_decode"] == 0


def test_cascade_routes_fifty_fifty_confidence_to_system_two() -> None:
    run, _requests = _run_harness(2, probabilities={"Entailment": 0.5, "Contradiction": 0.5})

    for record in run["records"]["cascade"]:
        assert record["routed_to_system_2"]
        assert record["prediction"] == "Contradiction"
        assert record["system_2_latency_ms"] is not None
        assert record["tokens_prefill"] == 120 and record["tokens_decode"] == 80
        assert record["json_valid"] and not record["system2_error_fallback"]


def test_system_two_http_error_falls_back_to_system1_prediction_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.ERROR):
        # 2 chamadas bem-sucedidas (passada system2_pure); a escalada da
        # cascata (confiança 0.5 < α) recebe 500.
        run, _requests = _run_harness(
            2,
            probabilities={"Entailment": 0.5, "Contradiction": 0.5},
            chat_status_after=2,
        )

    for record in run["records"]["cascade"]:
        assert record["system2_error_fallback"]
        assert record["prediction"] == "Entailment"  # top_choice do System 1
        assert record["system_2_latency_ms"] is None
        assert record["system2_fallback_used"] is None and record["json_valid"] is None
    assert "system 2 failed" in caplog.text


def _synthetic_run(confidences: list[float]) -> BenchmarkRun:
    samples = _samples(len(confidences))
    measurements = {
        sample["id"]: SampleMeasurement(
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
    assert records[0]["prediction"] == "Entailment" and records[0]["tokens_decode"] == 0


def test_save_raw_results_writes_meta_scenarios_sweep_summaries_and_kpis(tmp_path) -> None:
    run, _requests = _run_harness(4)
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
    assert set(data["alpha_sweep"]) == {"0.5", "0.7", "0.85", "0.9", "0.95"}


def test_plot_tradeoff_curve_writes_non_empty_png(tmp_path) -> None:
    run, _requests = _run_harness(3, probabilities={"Entailment": 0.5, "Contradiction": 0.5})
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
                "dataset: {name: nilc-nlp/assin2, split: test, frozen_samples: 4, seed: 42}",
                "model: {id: Qwen/Qwen3-4B-AWQ, quantization: awq}",
                'serving: {base_url: "http://127.0.0.1:8000/v1", port: 8000, prefix_caching: true}',
                "system1: {endpoint: /v1/systemone}",
                "system2: {endpoint: /v1/chat/completions, max_tokens: 256}",
                "router: {alpha: 0.85}",
                "metrics: {percentiles: [50, 95, 99]}",
                "kpis: {p95_reduction: 0.50, token_savings: 0.60, apgr_target: 0.95}",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluator, "get_frozen_eval_set", lambda **_kwargs: _samples(4))
    monkeypatch.setattr(evaluator, "health_check", lambda _url: True)
    requests: list[dict[str, object]] = []
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
    assert "KPI SC-002" in output and "KPI SC-003" in output
    assert "Gate F" in output and "α ótimo" in output
