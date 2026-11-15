"""Tipos do registro bruto e agregações puras do benchmark (Componente G).

Componente G (tabela de componentes no README);
stories correspondentes em docs/system2-evaluator-spec.md e, após o pivot
NCM (2026-11-14), em docs/dataset-pivot-ncm-spec.md (Story 6). Este módulo é
puro — sem HTTP, sem router — para que as métricas sejam testáveis
isoladamente; a orquestração de engines e a persistência ficam em
src/metrics/evaluator.py.

Pós-pivot: predições e ground truths são códigos NCM (str), cada registro
carrega o subconjunto `head`/`tail` e os KPIs são os gates SC-A/SC-B/SC-C da
spec do pivot (gap na cauda, recuperação, custo).
"""

import statistics
from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict

from src.data.loader import Subset

Scenario = Literal["system1_pure", "system2_pure", "cascade"]


class BenchmarkResult(TypedDict):
    """Registro bruto de uma requisição do benchmark (FR-006 de docs/engine-router-spec.md).

    Campos `None` significam "não aplicável ao cenário": `system_1_latency_ms`,
    `confidence` e `entropy` são None só em `system2_pure`; `system_2_*` é
    None/0 quando o System 2 não foi invocado. Tokens do System 1 são sempre
    0 para manter o comparativo de decode tokens restrito ao System 2 (o
    `usage` do chat completions do fast-path entra na conta de custo geral
    via `tokens_decode` apenas em cenários sem escalada — ver evaluator).
    `json_valid` é False quando o parser do System 2 recorreu a
    regex/default. `subset` particiona o conjunto congelado em cabeça
    (códigos frequentes no pool) e cauda (códigos raros — o long tail do
    blog do iFood).
    """

    query_id: int
    scenario: Scenario
    alpha: float | None
    replayed: bool
    subset: Subset
    system_1_latency_ms: float | None
    system_2_latency_ms: float | None
    total_latency_ms: float
    confidence: float | None
    entropy: float | None
    routed_to_system_2: bool
    tokens_prefill: int
    tokens_decode: int
    prediction: str
    ground_truth: str
    is_correct: bool
    system2_fallback_used: bool | None
    system2_error_fallback: bool
    json_valid: bool | None


class SummaryReport(TypedDict):
    """Agregados de um cenário: percentis, tokens, qualidade e taxas.

    `accuracy_head`/`accuracy_tail` são None quando o cenário não tem
    registros do subconjunto correspondente.
    """

    scenario: Scenario
    alpha: float | None
    n_requests: int
    n_latency_samples: int
    latency_percentiles_ms: dict[str, float]
    tokens_prefill_total: int
    tokens_decode_total: int
    accuracy: float
    accuracy_head: float | None
    accuracy_tail: float | None
    f1_macro: float
    system2_call_rate: float
    apgr: float | None
    json_success_rate: float | None


class KPIReport(TypedDict):
    """Veredito dos gates do pivot (SC-A/SC-B/SC-C): (passou, valor medido).

    `tail_gap_pp`: (acc S2 - acc S1) na cauda em pontos percentuais (SC-A).
    `recovery`: fração do gap da cauda que a cascata recupera (SC-B);
    `recovery_ok` exige também que a cascata não fique abaixo do S1 puro no
    conjunto completo.
    """

    p95_reduction: tuple[bool, float]
    token_savings: tuple[bool, float]
    tail_gap_pp: tuple[bool, float]
    recovery: tuple[bool, float]


def latency_percentiles_ms(
    latencies: Sequence[float], percentiles: Sequence[int]
) -> dict[str, float]:
    """Percentis de latência via `statistics.quantiles(n=100)` (mesmo método
    da célula C do notebook). Requer ao menos 2 amostras e percentis
    inteiros em [1, 99]; a chave de saída é `p{valor}` (ex.: `p95`)."""
    if len(latencies) < 2:
        raise ValueError(f"latencies needs at least two samples, got {len(latencies)}")
    for percentile in percentiles:
        if not 1 <= percentile <= 99:
            raise ValueError(f"percentiles must be integers in [1, 99], got {percentile!r}")
    cutpoints = statistics.quantiles(latencies, n=100)
    return {f"p{percentile}": cutpoints[percentile - 1] for percentile in percentiles}


def accuracy(predictions: Sequence[str], ground_truths: Sequence[str]) -> float:
    """Fração de acertos sobre o ground truth NCM."""
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"predictions and ground_truths must have the same length, "
            f"got {len(predictions)} and {len(ground_truths)}"
        )
    if not predictions:
        raise ValueError("predictions must not be empty")
    hits = sum(
        1
        for prediction, truth in zip(predictions, ground_truths, strict=True)
        if prediction == truth
    )
    return hits / len(predictions)


def macro_f1(predictions: Sequence[str], ground_truths: Sequence[str]) -> float:
    """F1 macro sobre os códigos presentes no ground truth ou previstos.

    A classe nunca prevista (precision 0) e a classe sem apoio no ground
    truth (recall 0) contribuem com F1 0.0, sem exceção (FR-006). Divisões
    por zero valem 0.0.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"predictions and ground_truths must have the same length, "
            f"got {len(predictions)} and {len(ground_truths)}"
        )
    if not predictions:
        raise ValueError("predictions must not be empty")

    labels = sorted(set(predictions) | set(ground_truths))
    per_class_f1: list[float] = []
    for label in labels:
        true_positive = sum(
            1
            for prediction, truth in zip(predictions, ground_truths, strict=True)
            if prediction == label and truth == label
        )
        false_positive = sum(
            1
            for prediction, truth in zip(predictions, ground_truths, strict=True)
            if prediction == label and truth != label
        )
        false_negative = sum(
            1
            for prediction, truth in zip(predictions, ground_truths, strict=True)
            if prediction != label and truth == label
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class_f1.append(f1)
    return sum(per_class_f1) / len(per_class_f1) if per_class_f1 else 0.0


def apgr(cascade_accuracy: float, system1_accuracy: float, system2_accuracy: float) -> float:
    """Average Performance Gap Recovered (métrica definida pelo RouteLLM).

    APGR = (Acc(cascata) - Acc(S1)) / (Acc(S2) - Acc(S1)). Quando System 1 e
    System 2 têm a mesma acurácia, o gap é zero e não há nada a recuperar:
    devolve 0.0 em vez de dividir por zero (edge case da spec). Pós-pivot o
    APGR é informativo; os gates são tail_gap/recovery (SC-A/SC-B).
    """
    gap = system2_accuracy - system1_accuracy
    if gap == 0.0:
        return 0.0
    return (cascade_accuracy - system1_accuracy) / gap


def _subset_accuracy(records: Sequence[BenchmarkResult], subset: Subset) -> float | None:
    selected = [record for record in records if record["subset"] == subset]
    if not selected:
        return None
    return accuracy(
        [record["prediction"] for record in selected],
        [record["ground_truth"] for record in selected],
    )


def compute_summary_metrics(
    records: Sequence[BenchmarkResult],
    *,
    percentiles: Sequence[int] = (50, 95, 99),
    warmup_discard: int = 5,
    system1_accuracy: float | None = None,
    system2_accuracy: float | None = None,
) -> SummaryReport:
    """Agrega os registros de UM cenário em um SummaryReport.

    O warmup discard (mitigação da oscilação de latência em ambiente
    compartilhado como o Colab) remove as `warmup_discard`
    primeiras requisições SÓ dos percentis de latência (cache frio distorce
    latência, não corretude); acurácia, F1, tokens e taxas usam todas as
    requisições. `apgr` só é calculado para a cascata com as baselines S1/S2
    informadas; `json_success_rate` só existe em cenários com chamadas ao
    System 2. Levanta `ValueError` se os registros misturarem cenários ou α.
    """
    if not records:
        raise ValueError("records must not be empty")
    scenarios = {record["scenario"] for record in records}
    if len(scenarios) != 1:
        raise ValueError(f"records must belong to a single scenario, got {sorted(scenarios)}")
    alphas = {record["alpha"] for record in records}
    if len(alphas) != 1:
        raise ValueError(f"records must share one alpha, got {sorted(alphas, key=repr)}")
    if warmup_discard < 0:
        raise ValueError(f"warmup_discard must be >= 0, got {warmup_discard}")
    if len(records) <= warmup_discard:
        raise ValueError(
            f"records needs more than warmup_discard={warmup_discard} samples "
            f"for latency percentiles, got {len(records)}"
        )

    scenario = records[0]["scenario"]
    alpha = records[0]["alpha"]
    latencies = [record["total_latency_ms"] for record in records[warmup_discard:]]

    predictions = [record["prediction"] for record in records]
    ground_truths = [record["ground_truth"] for record in records]
    accuracy_value = accuracy(predictions, ground_truths)
    f1_value = macro_f1(predictions, ground_truths)

    s2_records = [record for record in records if record["json_valid"] is not None]
    json_success_rate = (
        sum(1 for record in s2_records if record["json_valid"]) / len(s2_records)
        if s2_records
        else None
    )

    apgr_value = (
        apgr(accuracy_value, system1_accuracy, system2_accuracy)
        if scenario == "cascade" and system1_accuracy is not None and system2_accuracy is not None
        else None
    )

    return SummaryReport(
        scenario=scenario,
        alpha=alpha,
        n_requests=len(records),
        n_latency_samples=len(records) - warmup_discard,
        latency_percentiles_ms=latency_percentiles_ms(latencies, percentiles),
        tokens_prefill_total=sum(record["tokens_prefill"] for record in records),
        tokens_decode_total=sum(record["tokens_decode"] for record in records),
        accuracy=accuracy_value,
        accuracy_head=_subset_accuracy(records, "head"),
        accuracy_tail=_subset_accuracy(records, "tail"),
        f1_macro=f1_value,
        system2_call_rate=sum(1 for record in records if record["routed_to_system_2"])
        / len(records),
        apgr=apgr_value,
        json_success_rate=json_success_rate,
    )


def evaluate_kpis(
    cascade: SummaryReport,
    system1: SummaryReport,
    system2: SummaryReport,
    kpis: Mapping[str, float],
) -> KPIReport:
    """Compara a cascata contra S1/S2 puros nos gates do pivot (SC-A/B/C).

    SC-C: redução de p95 >= `kpis["p95_reduction"]` e redução de decode
    tokens >= `kpis["token_savings"]` (cascata vs System 2 puro). SC-A: gap
    de acurácia S2−S1 na cauda >= `kpis["tail_gap_pp"]` pontos percentuais.
    SC-B: a cascata recupera >= `kpis["recovery_share"]` do gap da cauda E
    não fica abaixo do S1 puro no conjunto completo. Levanta `ValueError`
    quando um baseline necessário falta (cauda sem registros, p95/decode
    zero no System 2).
    """
    system2_p95 = system2["latency_percentiles_ms"].get("p95", 0.0)
    if system2_p95 <= 0.0:
        raise ValueError("system2 summary must have a positive p95 to compare against")
    system2_decode = system2["tokens_decode_total"]
    if system2_decode <= 0:
        raise ValueError("system2 summary must have positive decode tokens to compare against")
    if system1["accuracy_tail"] is None or system2["accuracy_tail"] is None:
        raise ValueError("system1 and system2 summaries need tail-subset accuracy")
    if cascade["accuracy_tail"] is None:
        raise ValueError("cascade summary needs tail-subset accuracy")

    cascade_p95 = cascade["latency_percentiles_ms"].get("p95", 0.0)
    p95_reduction = 1.0 - cascade_p95 / system2_p95
    token_savings = 1.0 - cascade["tokens_decode_total"] / system2_decode

    tail_gap = system2["accuracy_tail"] - system1["accuracy_tail"]
    tail_gap_pp = tail_gap * 100.0
    recovery = (
        (cascade["accuracy_tail"] - system1["accuracy_tail"]) / tail_gap if tail_gap > 0.0 else 0.0
    )
    recovery_ok = recovery >= kpis["recovery_share"] and cascade["accuracy"] >= system1["accuracy"]

    return KPIReport(
        p95_reduction=(p95_reduction >= kpis["p95_reduction"], p95_reduction),
        token_savings=(token_savings >= kpis["token_savings"], token_savings),
        tail_gap_pp=(tail_gap_pp >= kpis["tail_gap_pp"], tail_gap_pp),
        recovery=(recovery_ok, recovery),
    )
