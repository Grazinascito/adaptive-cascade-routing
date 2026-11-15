"""BenchmarkHarness — orquestra os 3 cenários e gera relatórios e gráficos de métricas.

Componente G do design doc (docs/adaptive-cascade-design-doc.md §5.7);
stories correspondentes em docs/system2-evaluator-spec.md (Stories 2 e 3).
Tipos e métricas puras ficam em src/metrics/summary.py; este módulo faz a
orquestração dos engines C/F sobre o router D/E, a persistência dos
artefatos de results/ e a CLI `python -m src.metrics.evaluator --config ...`.

Latências vêm do `post_json` (FR-04, `time.perf_counter` na camada de
serving). Tokens do System 2 vêm do `usage` do `/v1/chat/completions`
(FR-005); os do System 1 ficam 0 porque o `/v1/systemone` não devolve
`usage`. O sweep de α (Story 3) é um replay sobre as medições gravadas —
"sweep alpha across the benchmark results" — sem novas chamadas HTTP.
"""

import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, TypedDict, cast

import httpx
import yaml

from src.data.loader import ASSIN2Sample, Label, format_system1_prompt, get_frozen_eval_set
from src.engine.router import RoutingQuery, System2Handler, route
from src.engine.server import health_check
from src.engine.serving import DEFAULT_BASE_URL, DEFAULT_MODEL
from src.engine.system1 import System1Output, evaluate_typed_decision
from src.engine.system2 import System2Output, execute_reasoning
from src.metrics.summary import (
    BenchmarkResult,
    Scenario,
    SummaryReport,
    compute_summary_metrics,
    evaluate_kpis,
)

logger = logging.getLogger(__name__)

ALPHA_SWEEP: Final = (0.50, 0.70, 0.85, 0.90, 0.95)
WARMUP_DISCARD: Final = 5
P95_SYSTEM1_TARGET_MS: Final = 60.0
JSON_GATE_TARGET: Final = 0.95


class SampleMeasurement(TypedDict):
    """Medições de UMA amostra nas duas passadas puras (base do replay)."""

    sample: ASSIN2Sample
    system1: System1Output
    system2: System2Output


class BenchmarkRun(TypedDict):
    """Saída de `run_full_benchmark`: registros live + tabela de medições."""

    samples: list[ASSIN2Sample]
    records: dict[Scenario, list[BenchmarkResult]]
    measurements: dict[int, SampleMeasurement]


def _build_result(
    sample: ASSIN2Sample,
    *,
    scenario: Scenario,
    alpha: float | None,
    replayed: bool,
    system1: System1Output | None,
    system2: System2Output | None,
    prediction: Label,
    error_fallback: bool = False,
) -> BenchmarkResult:
    """Compõe o registro bruto a partir das saídas dos engines.

    `routed_to_system_2` é derivado da presença da saída do System 2 (no
    cenário `system2_pure` ela está sempre presente). `total_latency_ms` é a
    soma das latências dos engines invocados.
    """
    ground_truth = sample["label"]
    return BenchmarkResult(
        query_id=sample["id"],
        scenario=scenario,
        alpha=alpha,
        replayed=replayed,
        system_1_latency_ms=None if system1 is None else system1["latency_ms"],
        system_2_latency_ms=None if system2 is None else system2["latency_ms"],
        total_latency_ms=(0.0 if system1 is None else system1["latency_ms"])
        + (0.0 if system2 is None else system2["latency_ms"]),
        confidence=None if system1 is None else system1["top_probability"],
        entropy=None if system1 is None else system1["entropy"],
        routed_to_system_2=system2 is not None,
        tokens_prefill=0 if system2 is None else system2["prompt_tokens"],
        tokens_decode=0 if system2 is None else system2["completion_tokens"],
        prediction=prediction,
        ground_truth=ground_truth,
        is_correct=prediction == ground_truth,
        system2_fallback_used=None if system2 is None else system2["fallback_used"],
        system2_error_fallback=error_fallback,
        json_valid=None if system2 is None else not system2["fallback_used"],
    )


class BenchmarkHarness:
    """Executa os 3 cenários sobre as mesmas amostras congeladas (FR-003).

    Um único `httpx.Client` é compartilhado entre os engines (evita
    reabrir conexão por requisição). Erros de HTTP no System 1 e no cenário
    `system2_pure` propagam (fail fast); na cascata, falha do System 2 cai
    no fallback gracioso da spec: predição do System 1 + log de erro.
    """

    def __init__(
        self,
        samples: Sequence[ASSIN2Sample],
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        client: httpx.Client | None = None,
        system1_timeout_s: float = 10.0,
        system2_timeout_s: float = 120.0,
        system2_max_tokens: int = 256,
    ) -> None:
        self._samples = list(samples)
        self._model = model
        self._base_url = base_url
        self._system1_timeout_s = system1_timeout_s
        self._system2_timeout_s = system2_timeout_s
        self._system2_max_tokens = system2_max_tokens
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client()

    def close(self) -> None:
        """Fecha o client quando o harness o criou (client injetado é do chamador)."""
        if self._owns_client:
            self._client.close()

    def _evaluate_system1(self, sample: ASSIN2Sample) -> System1Output:
        payload = format_system1_prompt(sample["premise"], sample["hypothesis"])
        # `FormattedPromptPayload.choices` é tipado como list[str] (payload
        # serializável); o cast só acalma o mypy — em runtime são os Labels.
        choices = cast("Sequence[Label]", payload["choices"])
        return evaluate_typed_decision(
            payload["user_state"],
            choices,
            model=self._model,
            base_url=self._base_url,
            timeout_s=self._system1_timeout_s,
            client=self._client,
        )

    def _execute_system2(self, premise: str, hypothesis: str) -> System2Output:
        return execute_reasoning(
            premise,
            hypothesis,
            model=self._model,
            base_url=self._base_url,
            max_tokens=self._system2_max_tokens,
            timeout_s=self._system2_timeout_s,
            client=self._client,
        )

    def _run_system1_pure(self) -> tuple[list[BenchmarkResult], dict[int, System1Output]]:
        records: list[BenchmarkResult] = []
        outputs: dict[int, System1Output] = {}
        for sample in self._samples:
            output = self._evaluate_system1(sample)
            outputs[sample["id"]] = output
            records.append(
                _build_result(
                    sample,
                    scenario="system1_pure",
                    alpha=None,
                    replayed=False,
                    system1=output,
                    system2=None,
                    prediction=output["top_choice"],
                )
            )
        return records, outputs

    def _run_system2_pure(self) -> tuple[list[BenchmarkResult], dict[int, System2Output]]:
        records: list[BenchmarkResult] = []
        outputs: dict[int, System2Output] = {}
        for sample in self._samples:
            output = self._execute_system2(sample["premise"], sample["hypothesis"])
            outputs[sample["id"]] = output
            records.append(
                _build_result(
                    sample,
                    scenario="system2_pure",
                    alpha=None,
                    replayed=False,
                    system1=None,
                    system2=output,
                    prediction=output["label"],
                )
            )
        return records, outputs

    def _make_cascade_handler(
        self,
        sample: ASSIN2Sample,
        system1: System1Output,
        holder: list[System2Output],
    ) -> System2Handler:
        """Adapter do `execute_reasoning` para o contrato `System2Handler`.

        Captura o `System2Output` completo em `holder` (o router só devolve o
        label) e aplica o fallback gracioso da spec em erro de HTTP: log de
        erro + predição do System 1.
        """

        def handler(query: RoutingQuery) -> Label:
            try:
                output = self._execute_system2(query["premise"], query["hypothesis"])
            except httpx.HTTPError as error:
                logger.error(
                    "system 2 failed for query %s; falling back to system 1 prediction '%s' (%s)",
                    sample["id"],
                    system1["top_choice"],
                    error,
                )
                return system1["top_choice"]
            holder.append(output)
            return output["label"]

        return handler

    def _run_cascade(self, alpha: float) -> list[BenchmarkResult]:
        records: list[BenchmarkResult] = []
        for sample in self._samples:
            system1 = self._evaluate_system1(sample)
            query = RoutingQuery(premise=sample["premise"], hypothesis=sample["hypothesis"])
            holder: list[System2Output] = []
            outcome = route(
                query,
                system1["top_probability"],
                alpha,
                self._make_cascade_handler(sample, system1, holder),
            )
            system2 = holder[0] if holder else None
            error_fallback = outcome["routed_to_system_2"] and system2 is None
            prediction = outcome["system2_prediction"] or system1["top_choice"]
            records.append(
                _build_result(
                    sample,
                    scenario="cascade",
                    alpha=alpha,
                    replayed=False,
                    system1=system1,
                    system2=system2,
                    prediction=prediction,
                    error_fallback=error_fallback,
                )
            )
        return records

    def run_full_benchmark(self, *, confidence_threshold: float = 0.85) -> BenchmarkRun:
        """Roda os 3 cenários live nas mesmas amostras (FR-003, Story 2).

        Retorna os registros por cenário e a tabela de medições por
        `query_id` (saída do S1 na passada pura + saída do S2 na passada
        pura) que alimenta o replay do sweep.
        """
        system1_records, system1_outputs = self._run_system1_pure()
        system2_records, system2_outputs = self._run_system2_pure()
        cascade_records = self._run_cascade(confidence_threshold)
        measurements = {
            sample["id"]: SampleMeasurement(
                sample=sample,
                system1=system1_outputs[sample["id"]],
                system2=system2_outputs[sample["id"]],
            )
            for sample in self._samples
        }
        return BenchmarkRun(
            samples=list(self._samples),
            records={
                "system1_pure": system1_records,
                "system2_pure": system2_records,
                "cascade": cascade_records,
            },
            measurements=measurements,
        )


def sweep_alpha(
    run: BenchmarkRun,
    *,
    alphas: Sequence[float] = ALPHA_SWEEP,
) -> dict[float, list[BenchmarkResult]]:
    """Replay da decisão de rota por α sobre as medições gravadas (Story 3).

    Sem novas chamadas HTTP: `temperature=0.0` torna a saída do System 2
    determinística na mesma sessão, então label/tokens/latência medidos na
    passada pura são a resposta que a cascata obteria ao escalar. Cada
    decisão invoca `route()` de verdade (Componente E) com a confiança
    medida; registros saem com `replayed=True`.
    """
    sweep: dict[float, list[BenchmarkResult]] = {}
    for alpha in alphas:
        records: list[BenchmarkResult] = []
        for measurement in run["measurements"].values():
            sample = measurement["sample"]
            system1 = measurement["system1"]
            system2 = measurement["system2"]

            def handler(_query: RoutingQuery, s2: System2Output = system2) -> Label:
                return s2["label"]

            outcome = route(
                RoutingQuery(premise=sample["premise"], hypothesis=sample["hypothesis"]),
                system1["top_probability"],
                alpha,
                handler,
            )
            routed = outcome["routed_to_system_2"]
            prediction = outcome["system2_prediction"] or system1["top_choice"]
            records.append(
                _build_result(
                    sample,
                    scenario="cascade",
                    alpha=alpha,
                    replayed=True,
                    system1=system1,
                    system2=system2 if routed else None,
                    prediction=prediction,
                )
            )
        sweep[alpha] = records
    return sweep


def save_raw_results(
    path: Path,
    *,
    meta: Mapping[str, object],
    scenarios: Mapping[Scenario, Sequence[BenchmarkResult]],
    alpha_sweep: Mapping[str, Sequence[BenchmarkResult]],
    summaries: Mapping[str, object],
    kpis: Mapping[str, object] | None,
) -> None:
    """Grava `results/raw_benchmark_data.json` (NFR-002: auditabilidade)."""
    payload = {
        "meta": dict(meta),
        "scenarios": {name: list(records) for name, records in scenarios.items()},
        "alpha_sweep": {name: list(records) for name, records in alpha_sweep.items()},
        "summaries": dict(summaries),
        "kpis": dict(kpis) if kpis is not None else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def plot_tradeoff_curve(
    path: Path,
    run: BenchmarkRun,
    sweep_summaries: Mapping[float, SummaryReport],
    *,
    apgr_target: float = 0.95,
) -> None:
    """Grava `results/latency_cost_tradeoff.png` (Story 3): dispersão de
    latência por requisição + curva APGR × taxa de chamadas ao System 2."""
    import matplotlib.pyplot as plt

    fig, (ax_scatter, ax_curve) = plt.subplots(1, 2, figsize=(12.0, 5.0))

    cascade = run["records"]["cascade"]
    system2_pure = {record["query_id"]: record for record in run["records"]["system2_pure"]}
    system2_latencies = [system2_pure[record["query_id"]]["total_latency_ms"] for record in cascade]
    cascade_latencies = [record["total_latency_ms"] for record in cascade]
    ax_scatter.scatter(system2_latencies, cascade_latencies, s=12.0, alpha=0.6)
    ceiling = max([*system2_latencies, *cascade_latencies, 1.0]) * 1.05
    ax_scatter.plot([0.0, ceiling], [0.0, ceiling], linestyle="--", color="gray", linewidth=1.0)
    ax_scatter.set_xlabel("System 2 puro — latência total (ms)")
    ax_scatter.set_ylabel("Cascata (α padrão) — latência total (ms)")
    ax_scatter.set_title("Latência por requisição")

    alphas = sorted(sweep_summaries)
    call_rates = [sweep_summaries[alpha]["system2_call_rate"] for alpha in alphas]
    apgr_values = [sweep_summaries[alpha]["apgr"] or 0.0 for alpha in alphas]
    ax_curve.plot(call_rates, apgr_values, marker="o")
    for alpha, call_rate, apgr_value in zip(alphas, call_rates, apgr_values, strict=True):
        marker = "*" if alpha == 0.85 else "o"
        ax_curve.plot(
            call_rate, apgr_value, marker=marker, markersize=12.0 if marker == "*" else 6.0
        )
        ax_curve.annotate(
            f"α={alpha}", (call_rate, apgr_value), textcoords="offset points", xytext=(4, 4)
        )
    ax_curve.axhline(apgr_target, linestyle="--", color="gray", linewidth=1.0)
    ax_curve.set_xlabel("Taxa de chamadas ao System 2")
    ax_curve.set_ylabel("APGR")
    ax_curve.set_title("APGR × custo")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _optimal_alpha(
    sweep_summaries: Mapping[float, SummaryReport], apgr_target: float
) -> tuple[float, SummaryReport]:
    """α ótimo (Story 3): entre os α com APGR >= alvo, o de menor taxa de S2;
    sem candidato, o de maior APGR alcançada."""
    candidates = [
        (alpha, summary)
        for alpha, summary in sweep_summaries.items()
        if (summary["apgr"] or 0.0) >= apgr_target
    ]
    if candidates:
        return min(candidates, key=lambda pair: pair[1]["system2_call_rate"])
    return max(sweep_summaries.items(), key=lambda pair: pair[1]["apgr"] or 0.0)


def _print_summary(title: str, summary: SummaryReport) -> None:
    percentiles = " ".join(
        f"{key}={value:.1f}ms" for key, value in summary["latency_percentiles_ms"].items()
    )
    print(f"== {title} ==")
    print(
        f"  requisições={summary['n_requests']} (latência em {summary['n_latency_samples']}) "
        f"| {percentiles}"
    )
    print(
        f"  tokens prefill={summary['tokens_prefill_total']}"
        f" decode={summary['tokens_decode_total']}"
        f" | acurácia={summary['accuracy']:.3f}"
        f" f1_macro={summary['f1_macro']:.3f}"
        f" | taxa S2={summary['system2_call_rate']:.2f}"
    )
    apgr_text = f"{summary['apgr']:.3f}" if summary["apgr"] is not None else "n/a"
    json_text = (
        f"{summary['json_success_rate']:.3f}" if summary["json_success_rate"] is not None else "n/a"
    )
    print(f"  APGR={apgr_text} | json_success_rate={json_text}")


def _print_kpi(name: str, passed: bool, value: float, target: str) -> None:
    print(f"KPI {name}: {value:.3f} vs alvo {target} → {'PASS' if passed else 'FAIL'}")


def main(
    argv: Sequence[str] | None = None,
    *,
    client: httpx.Client | None = None,
    results_dir: Path | None = None,
) -> int:
    """CLI: `python -m src.metrics.evaluator --config config/benchmark_config.yaml`.

    Executa o benchmark completo (3 cenários + sweep de α), grava os
    artefatos em `results/` e imprime os KPIs com PASS/FAIL. Exit 0 sempre
    que o benchmark roda (KPI é achado de medição, não erro); exit 1 só
    para falha de setup (config, servidor, dataset).
    """
    parser = argparse.ArgumentParser(
        prog="src.metrics.evaluator",
        description="Roda o benchmark de cascata adaptativa (Componente G).",
    )
    parser.add_argument("--config", required=True, help="caminho do benchmark_config.yaml")
    parser.add_argument("--results-dir", default=None, help="diretório de saída (padrão: results)")
    args = parser.parse_args(argv)

    try:
        config: dict[str, Any] = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        dataset = config["dataset"]
        model_id = str(config["model"]["id"])
        base_url = str(config["serving"]["base_url"])
        max_tokens = int(config["system2"]["max_tokens"])
        alpha_default = float(config["router"]["alpha"])
        percentiles = [int(value) for value in config["metrics"]["percentiles"]]
        kpis = {key: float(value) for key, value in config["kpis"].items()}
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(f"config inválido em {args.config}: {error}", file=sys.stderr)
        return 1

    root_url = base_url.removesuffix("/v1")
    if not health_check(root_url):
        print(
            f"servidor vLLM indisponível em {root_url} — suba o servidor antes do benchmark",
            file=sys.stderr,
        )
        return 1

    try:
        samples = get_frozen_eval_set(
            split=str(dataset["split"]),
            sample_limit=int(dataset["frozen_samples"]),
            seed=int(dataset["seed"]),
        )
    except (OSError, ValueError) as error:
        print(f"falha ao carregar o dataset {dataset['name']}: {error}", file=sys.stderr)
        return 1

    output_dir = results_dir if results_dir is not None else Path(args.results_dir or "results")
    # Dry-runs pequenos não têm 5 requisições sobrando depois do descarte.
    warmup_discard = min(WARMUP_DISCARD, max(0, len(samples) - 2))
    harness = BenchmarkHarness(
        samples,
        model=model_id,
        base_url=base_url,
        client=client,
        system2_max_tokens=max_tokens,
    )
    try:
        run = harness.run_full_benchmark(confidence_threshold=alpha_default)
        sweep = sweep_alpha(run)
        system1_summary = compute_summary_metrics(
            run["records"]["system1_pure"], percentiles=percentiles, warmup_discard=warmup_discard
        )
        system2_summary = compute_summary_metrics(
            run["records"]["system2_pure"], percentiles=percentiles, warmup_discard=warmup_discard
        )
        cascade_summary = compute_summary_metrics(
            run["records"]["cascade"],
            percentiles=percentiles,
            warmup_discard=warmup_discard,
            system1_accuracy=system1_summary["accuracy"],
            system2_accuracy=system2_summary["accuracy"],
        )
        sweep_summaries = {
            alpha: compute_summary_metrics(
                records,
                percentiles=percentiles,
                warmup_discard=warmup_discard,
                system1_accuracy=system1_summary["accuracy"],
                system2_accuracy=system2_summary["accuracy"],
            )
            for alpha, records in sweep.items()
        }
        kpi_report = evaluate_kpis(cascade_summary, system2_summary, kpis)
    finally:
        harness.close()

    _print_summary("system1_pure", system1_summary)
    _print_summary("system2_pure", system2_summary)
    _print_summary("cascade", cascade_summary)
    for alpha in sorted(sweep_summaries):
        _print_summary(f"cascade α={alpha}", sweep_summaries[alpha])

    _print_kpi(
        "SC-002 redução de p95", *kpi_report["p95_reduction"], f">= {kpis['p95_reduction']:.2f}"
    )
    _print_kpi(
        "SC-004 economia de decode tokens",
        *kpi_report["token_savings"],
        f">= {kpis['token_savings']:.2f}",
    )
    _print_kpi("SC-003 APGR", *kpi_report["apgr"], f">= {kpis['apgr_target']:.2f}")

    gate_f = system2_summary["json_success_rate"]
    if gate_f is not None:
        verdict = "PASS" if gate_f >= JSON_GATE_TARGET else "FAIL"
        print(
            f"Gate F (re-medição): json_success_rate {gate_f:.3f}"
            f" vs alvo >= {JSON_GATE_TARGET:.2f} → {verdict}"
        )
    s1_p95 = system1_summary["latency_percentiles_ms"].get("p95")
    if s1_p95 is not None:
        verdict = "OK" if s1_p95 <= P95_SYSTEM1_TARGET_MS else "ACIMA"
        print(
            f"SC-001 (informativo): system1 p95 {s1_p95:.1f} ms"
            f" vs alvo <= {P95_SYSTEM1_TARGET_MS:.0f} ms → {verdict}"
        )

    optimal_alpha, optimal_summary = _optimal_alpha(sweep_summaries, kpis["apgr_target"])
    print(
        f"α ótimo: {optimal_alpha}"
        f" (APGR {optimal_summary['apgr'] or 0.0:.3f},"
        f" taxa S2 {optimal_summary['system2_call_rate']:.2f})"
    )

    save_raw_results(
        output_dir / "raw_benchmark_data.json",
        meta={
            "model": model_id,
            "base_url": base_url,
            "frozen_samples": len(samples),
            "seed": int(dataset["seed"]),
            "alpha_default": alpha_default,
            "alphas": list(sweep_summaries),
            "warmup_discard": warmup_discard,
            "percentiles": percentiles,
            "generated_at_utc": datetime.now(UTC).isoformat(),
        },
        scenarios=run["records"],
        alpha_sweep={str(alpha): records for alpha, records in sweep.items()},
        summaries={
            "system1_pure": system1_summary,
            "system2_pure": system2_summary,
            "cascade": cascade_summary,
            "alpha_sweep": {str(alpha): summary for alpha, summary in sweep_summaries.items()},
        },
        kpis=kpi_report,
    )
    plot_tradeoff_curve(
        output_dir / "latency_cost_tradeoff.png",
        run,
        sweep_summaries,
        apgr_target=kpis["apgr_target"],
    )
    print(
        f"artefatos gravados em {output_dir}/: raw_benchmark_data.json, latency_cost_tradeoff.png"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
