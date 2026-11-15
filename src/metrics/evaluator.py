"""BenchmarkHarness — orquestra os 3 cenários e gera relatórios e gráficos de métricas.

Componente G (tabela de componentes no README);
stories correspondentes em docs/system2-evaluator-spec.md e, pós-pivot NCM
(2026-11-14), em docs/dataset-pivot-ncm-spec.md (Story 6). Tipos e métricas
puras ficam em src/metrics/summary.py; este módulo faz a orquestração dos
engines C/F sobre o router D/E, a persistência dos artefatos de results/ e
a CLI `python -m src.metrics.evaluator --config ...`.

Latências vêm do `post_json` (FR-04, `time.perf_counter` na camada de
serving). Tokens de AMBOS os engines vêm do `usage` do
`/v1/chat/completions` (pós-pivot o fast-path também é chat completions;
sua conta entra no custo da cascata). O sweep de α (Story 3) é um replay
sobre as medições gravadas — sem novas chamadas HTTP. Predições e ground
truths são truncados para `label_digits` (granularidade congelada pela
sonda) de forma idêntica antes da comparação.
"""

import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, TypedDict

import httpx
import yaml

from src.data.loader import (
    NCMEvalSample,
    get_frozen_eval_set,
    load_ncm_pool,
    pool_stats,
    truncate_code,
)
from src.engine.router import RoutingQuery, System2Handler, route
from src.engine.server import health_check
from src.engine.serving import DEFAULT_BASE_URL, DEFAULT_MODEL
from src.engine.system1 import System1Output, evaluate_short_classification
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
JSON_GATE_TARGET: Final = 0.95
# Banda de taxa de fallback reportada pelo blog do iFood (30–50% escalam ao
# modelo caro); reportada ao lado do α ótimo para comparação, sem gate.
BLOG_FALLBACK_BAND: Final = (0.30, 0.50)
# Gate anti-degeneração da spec do pivot (edge case): se o S1 tem confiança
# alta em praticamente tudo, o par modelo+tarefa não tem faixa de ambiguidade
# e o run completo seria desperdício — o gate é impresso como aviso.
CONFIDENCE_DEGENERATION_SHARE: Final = 0.95
CONFIDENCE_DEGENERATION_LEVEL: Final = 0.95


class SampleMeasurement(TypedDict):
    """Medições de UMA amostra nas duas passadas puras (base do replay)."""

    sample: NCMEvalSample
    system1: System1Output
    system2: System2Output


class BenchmarkRun(TypedDict):
    """Saída de `run_full_benchmark`: registros live + tabela de medições."""

    samples: list[NCMEvalSample]
    records: dict[Scenario, list[BenchmarkResult]]
    measurements: dict[int, SampleMeasurement]


def _build_result(
    sample: NCMEvalSample,
    *,
    label_digits: int,
    scenario: Scenario,
    alpha: float | None,
    replayed: bool,
    system1: System1Output | None,
    system2: System2Output | None,
    prediction: str,
    error_fallback: bool = False,
) -> BenchmarkResult:
    """Compõe o registro bruto a partir das saídas dos engines.

    `routed_to_system_2` é derivado da presença da saída do System 2 (no
    cenário `system2_pure` ela está sempre presente). `total_latency_ms` é a
    soma das latências dos engines invocados. Tokens somam os engines
    invocados. Predição e ground truth são truncados para `label_digits`
    antes da comparação (a granularidade congelada pela sonda).
    """
    ground_truth = truncate_code(sample["ncm_code"], label_digits)
    truncated_prediction = (
        prediction if len(prediction) != 8 else truncate_code(prediction, label_digits)
    )
    return BenchmarkResult(
        query_id=sample["item_id"],
        scenario=scenario,
        alpha=alpha,
        replayed=replayed,
        subset=sample["subset"],
        system_1_latency_ms=None if system1 is None else system1["latency_ms"],
        system_2_latency_ms=None if system2 is None else system2["latency_ms"],
        total_latency_ms=(0.0 if system1 is None else system1["latency_ms"])
        + (0.0 if system2 is None else system2["latency_ms"]),
        confidence=None if system1 is None else system1["confidence"],
        entropy=None if system1 is None else system1["entropy"],
        routed_to_system_2=system2 is not None,
        tokens_prefill=(0 if system1 is None else system1["prompt_tokens"])
        + (0 if system2 is None else system2["prompt_tokens"]),
        tokens_decode=(0 if system1 is None else system1["completion_tokens"])
        + (0 if system2 is None else system2["completion_tokens"]),
        prediction=truncated_prediction,
        ground_truth=ground_truth,
        is_correct=truncated_prediction == ground_truth,
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
        samples: Sequence[NCMEvalSample],
        *,
        label_digits: int = 8,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        client: httpx.Client | None = None,
        system1_timeout_s: float = 10.0,
        system2_timeout_s: float = 120.0,
        system1_max_tokens: int = 12,
        system2_max_tokens: int = 256,
    ) -> None:
        self._samples = list(samples)
        self._label_digits = label_digits
        self._model = model
        self._base_url = base_url
        self._system1_timeout_s = system1_timeout_s
        self._system2_timeout_s = system2_timeout_s
        self._system1_max_tokens = system1_max_tokens
        self._system2_max_tokens = system2_max_tokens
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client()

    def close(self) -> None:
        """Fecha o client quando o harness o criou (client injetado é do chamador)."""
        if self._owns_client:
            self._client.close()

    def _evaluate_system1(self, sample: NCMEvalSample) -> System1Output:
        return evaluate_short_classification(
            sample["name"],
            model=self._model,
            base_url=self._base_url,
            max_tokens=self._system1_max_tokens,
            timeout_s=self._system1_timeout_s,
            client=self._client,
        )

    def _execute_system2(self, item_name: str) -> System2Output:
        return execute_reasoning(
            item_name,
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
            outputs[sample["item_id"]] = output
            records.append(
                _build_result(
                    sample,
                    label_digits=self._label_digits,
                    scenario="system1_pure",
                    alpha=None,
                    replayed=False,
                    system1=output,
                    system2=None,
                    prediction=output["code"],
                )
            )
        return records, outputs

    def _run_system2_pure(self) -> tuple[list[BenchmarkResult], dict[int, System2Output]]:
        records: list[BenchmarkResult] = []
        outputs: dict[int, System2Output] = {}
        for sample in self._samples:
            output = self._execute_system2(sample["name"])
            outputs[sample["item_id"]] = output
            records.append(
                _build_result(
                    sample,
                    label_digits=self._label_digits,
                    scenario="system2_pure",
                    alpha=None,
                    replayed=False,
                    system1=None,
                    system2=output,
                    prediction=output["ncm_code"],
                )
            )
        return records, outputs

    def _make_cascade_handler(
        self,
        sample: NCMEvalSample,
        system1: System1Output,
        holder: list[System2Output],
    ) -> System2Handler:
        """Adapter do `execute_reasoning` para o contrato `System2Handler`.

        Captura o `System2Output` completo em `holder` (o router só devolve a
        predição) e aplica o fallback gracioso da spec em erro de HTTP: log de
        erro + predição do System 1.
        """

        def handler(query: RoutingQuery) -> str:
            try:
                output = self._execute_system2(query["item_name"])
            except httpx.HTTPError as error:
                logger.error(
                    "system 2 failed for item %s; falling back to system 1 prediction '%s' (%s)",
                    sample["item_id"],
                    system1["code"],
                    error,
                )
                return system1["code"]
            holder.append(output)
            return output["ncm_code"]

        return handler

    def _run_cascade(self, alpha: float) -> list[BenchmarkResult]:
        records: list[BenchmarkResult] = []
        for sample in self._samples:
            system1 = self._evaluate_system1(sample)
            query = RoutingQuery(item_name=sample["name"])
            holder: list[System2Output] = []
            outcome = route(
                query,
                system1["confidence"],
                alpha,
                self._make_cascade_handler(sample, system1, holder),
            )
            system2 = holder[0] if holder else None
            error_fallback = outcome["routed_to_system_2"] and system2 is None
            prediction = outcome["system2_prediction"] or system1["code"]
            records.append(
                _build_result(
                    sample,
                    label_digits=self._label_digits,
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
        `item_id` (saída do S1 na passada pura + saída do S2 na passada
        pura) que alimenta o replay do sweep.
        """
        system1_records, system1_outputs = self._run_system1_pure()
        system2_records, system2_outputs = self._run_system2_pure()
        cascade_records = self._run_cascade(confidence_threshold)
        measurements = {
            sample["item_id"]: SampleMeasurement(
                sample=sample,
                system1=system1_outputs[sample["item_id"]],
                system2=system2_outputs[sample["item_id"]],
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


def confidence_degeneration_share(run: BenchmarkRun) -> float:
    """Fração de itens com confiança do S1 >= nível degeneração (gate da spec).

    Se > `CONFIDENCE_DEGENERATION_SHARE` dos itens têm confiança >=
    `CONFIDENCE_DEGENERATION_LEVEL`, o par modelo+tarefa não tem faixa de
    ambiguidade (o cenário que o Marco 1 mediu no ASSIN 2) e o α é
    incalibrável — o harness imprime o aviso antes do run completo.
    """
    confidences = [m["system1"]["confidence"] for m in run["measurements"].values()]
    if not confidences:
        return 0.0
    high = sum(1 for value in confidences if value >= CONFIDENCE_DEGENERATION_LEVEL)
    return high / len(confidences)


def sweep_alpha(
    run: BenchmarkRun,
    *,
    alphas: Sequence[float] = ALPHA_SWEEP,
    label_digits: int = 8,
) -> dict[float, list[BenchmarkResult]]:
    """Replay da decisão de rota por α sobre as medições gravadas (Story 3).

    Sem novas chamadas HTTP: `temperature=0.0` torna a saída do System 2
    determinística na mesma sessão, então código/tokens/latência medidos na
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

            def handler(_query: RoutingQuery, s2: System2Output = system2) -> str:
                return s2["ncm_code"]

            outcome = route(
                RoutingQuery(item_name=sample["name"]),
                system1["confidence"],
                alpha,
                handler,
            )
            routed = outcome["routed_to_system_2"]
            prediction = outcome["system2_prediction"] or system1["code"]
            records.append(
                _build_result(
                    sample,
                    label_digits=label_digits,
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
    """Grava `results/raw_benchmark_data.json` (auditabilidade do run)."""
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
    recovery_target: float = 0.50,
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
    ax_curve.axhline(recovery_target, linestyle="--", color="gray", linewidth=1.0)
    ax_curve.set_xlabel("Taxa de chamadas ao System 2")
    ax_curve.set_ylabel("APGR (informativo)")
    ax_curve.set_title("APGR × custo")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _optimal_alpha(
    sweep_summaries: Mapping[float, SummaryReport], system1_accuracy: float
) -> tuple[float, SummaryReport]:
    """α ótimo: entre os α com acurácia >= S1 puro, o de menor taxa de S2;
    sem candidato, o de maior acurácia alcançada."""
    candidates = [
        (alpha, summary)
        for alpha, summary in sweep_summaries.items()
        if summary["accuracy"] >= system1_accuracy
    ]
    if candidates:
        return min(candidates, key=lambda pair: pair[1]["system2_call_rate"])
    return max(sweep_summaries.items(), key=lambda pair: pair[1]["accuracy"])


def _print_summary(title: str, summary: SummaryReport) -> None:
    percentiles = " ".join(
        f"{key}={value:.1f}ms" for key, value in summary["latency_percentiles_ms"].items()
    )

    def _acc(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    print(f"== {title} ==")
    print(
        f"  requisições={summary['n_requests']} (latência em {summary['n_latency_samples']}) "
        f"| {percentiles}"
    )
    print(
        f"  tokens prefill={summary['tokens_prefill_total']}"
        f" decode={summary['tokens_decode_total']}"
        f" | acurácia={summary['accuracy']:.3f}"
        f" (cabeça={_acc(summary['accuracy_head'])} cauda={_acc(summary['accuracy_tail'])})"
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
        description="Roda o benchmark de cascata adaptativa (Componente G, pivot NCM).",
    )
    parser.add_argument("--config", required=True, help="caminho do benchmark_config.yaml")
    parser.add_argument("--results-dir", default=None, help="diretório de saída (padrão: results)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="substitui frozen_samples (dry-run: usa as N primeiras amostras)",
    )
    args = parser.parse_args(argv)

    try:
        config: dict[str, Any] = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        dataset = config["dataset"]
        model_id = str(config["model"]["id"])
        base_url = str(config["serving"]["base_url"])
        max_tokens_s1 = int(config["system1"]["max_tokens"])
        max_tokens = int(config["system2"]["max_tokens"])
        alpha_default = float(config["router"]["alpha"])
        percentiles = [int(value) for value in config["metrics"]["percentiles"]]
        kpis = {key: float(value) for key, value in config["kpis"].items()}
        label_digits = int(dataset.get("label_digits", 8))
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
        pool, n_rows = load_ncm_pool(str(dataset["name"]))
        stats = pool_stats(pool, n_rows=n_rows)
        sample_limit = int(args.limit or dataset["frozen_samples"])
        samples = get_frozen_eval_set(
            pool,
            sample_limit=sample_limit,
            head_share=float(dataset["head_share"]),
            head_min_code_freq=int(dataset["head_min_code_freq"]),
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
        label_digits=label_digits,
        model=model_id,
        base_url=base_url,
        client=client,
        system1_max_tokens=max_tokens_s1,
        system2_max_tokens=max_tokens,
    )
    try:
        run = harness.run_full_benchmark(confidence_threshold=alpha_default)
        sweep = sweep_alpha(run, label_digits=label_digits)
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
        kpi_report = evaluate_kpis(cascade_summary, system1_summary, system2_summary, kpis)
    finally:
        harness.close()

    degeneration = confidence_degeneration_share(run)
    if degeneration > CONFIDENCE_DEGENERATION_SHARE:
        print(
            f"AVISO (gate anti-degeneração): {degeneration:.2f} dos itens têm confiança"
            f" >= {CONFIDENCE_DEGENERATION_LEVEL} no S1 — o par modelo+tarefa não tem"
            " faixa de ambiguidade; o α é incalibrável (cenário do Marco 1 no ASSIN 2)."
        )

    _print_summary("system1_pure", system1_summary)
    _print_summary("system2_pure", system2_summary)
    _print_summary("cascade", cascade_summary)
    for alpha in sorted(sweep_summaries):
        _print_summary(f"cascade α={alpha}", sweep_summaries[alpha])

    _print_kpi(
        "SC-C redução de p95", *kpi_report["p95_reduction"], f">= {kpis['p95_reduction']:.2f}"
    )
    _print_kpi(
        "SC-C economia de decode tokens",
        *kpi_report["token_savings"],
        f">= {kpis['token_savings']:.2f}",
    )
    _print_kpi(
        "SC-A gap de acurácia S2−S1 na cauda (pp)",
        *kpi_report["tail_gap_pp"],
        f">= {kpis['tail_gap_pp']:.0f}",
    )
    _print_kpi(
        "SC-B recuperação do gap pela cascata",
        *kpi_report["recovery"],
        f">= {kpis['recovery_share']:.2f} (e cascata >= S1 no conjunto completo)",
    )

    gate_f = system2_summary["json_success_rate"]
    if gate_f is not None:
        verdict = "PASS" if gate_f >= JSON_GATE_TARGET else "FAIL"
        print(
            f"Gate F (re-medição): json_success_rate {gate_f:.3f}"
            f" vs alvo >= {JSON_GATE_TARGET:.2f} → {verdict}"
        )

    optimal_alpha, optimal_summary = _optimal_alpha(sweep_summaries, system1_summary["accuracy"])
    fallback_rate = optimal_summary["system2_call_rate"]
    low, high = BLOG_FALLBACK_BAND
    band_text = "dentro" if low <= fallback_rate <= high else "fora"
    print(
        f"α ótimo: {optimal_alpha} (acurácia {optimal_summary['accuracy']:.3f},"
        f" taxa S2 {fallback_rate:.2f})"
    )
    print(
        f"Taxa de fallback no α ótimo: {fallback_rate:.2f} — {band_text} da banda"
        f" {low:.0%}–{high:.0%} reportada pelo blog do iFood"
    )

    save_raw_results(
        output_dir / "raw_benchmark_data.json",
        meta={
            "model": model_id,
            "base_url": base_url,
            "dataset": str(dataset["name"]),
            "pool_stats": dict(stats),
            "frozen_samples": len(samples),
            "seed": int(dataset["seed"]),
            "head_share": float(dataset["head_share"]),
            "head_min_code_freq": int(dataset["head_min_code_freq"]),
            "label_digits": label_digits,
            "alpha_default": alpha_default,
            "alphas": list(sweep_summaries),
            "warmup_discard": warmup_discard,
            "percentiles": percentiles,
            "degeneration_share": degeneration,
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
        recovery_target=kpis["recovery_share"],
    )
    print(
        f"artefatos gravados em {output_dir}/: raw_benchmark_data.json, latency_cost_tradeoff.png"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
