"""Orquestrador do benchmark DARU (PR5): loader + engines + router + métricas.

Dois modos (SPEC-DARU.md §5–6): o pilot (100 itens de validação, gate de cauda
antes de gastar GPU no run completo) e o run completo — validação 200 com S1 e
S2 medidos uma única vez e varredura de α por aritmética pura sobre as saídas
salvas, depois teste congelado 600 nos cenários SC-A/SC-B/SC-C. O resultado
vai para results/raw_benchmark_data.json.
"""

import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import httpx
import yaml

from src.data.loader import BenchmarkSets, CatalogSample, load_catalog_benchmark_set
from src.engine.router import CascadeRouter
from src.engine.system1 import System1Output, evaluate_typed_decision
from src.engine.system2 import System2Engine, System2Output
from src.metrics.evaluator import BenchmarkRecord, apgr, slice_accuracy
from src.metrics.summary import ScenarioSummary, summarize

SWEEP_ALPHAS: Final = (0.70, 0.80, 0.85, 0.90)
WARMUP_REQUESTS: Final = 5
MAX_WORKERS: Final = 8
PILOT_HEAD: Final = 70
PILOT_TAIL: Final = 30
SYSTEM1_TIMEOUT_S: Final = 60.0
SYSTEM2_TIMEOUT_S: Final = 240.0
RESULTS_PATH: Final = Path("results/raw_benchmark_data.json")
PILOT_RESULTS_PATH: Final = Path("results/pilot_gate.json")


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping, got {type(config).__name__}: {path}")
    return config


@dataclass(frozen=True)
class BenchmarkSettings:
    """Contrato congelado do config/benchmark_config.yaml já validado e tipado."""

    categories: tuple[str, ...]
    model: str
    base_url: str
    port: int
    alpha: float
    max_tokens: int
    n_validation: int
    n_test: int
    head_share: float
    tail_max_chars: int
    seed: int
    tail_gap_pp: float
    recovery_share: float
    token_share_vs_s2: float
    max_workers: int = MAX_WORKERS
    warmup_requests: int = WARMUP_REQUESTS

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "BenchmarkSettings":
        dataset = config["dataset"]
        return cls(
            categories=tuple(str(name) for name in dataset["categories"]),
            model=str(config["model"]["id"]),
            base_url=str(config["serving"]["base_url"]),
            port=int(config["serving"]["port"]),
            alpha=float(config["router"]["alpha"]),
            max_tokens=int(config["system2"]["max_tokens"]),
            n_validation=int(dataset["validation"]),
            n_test=int(dataset["test"]),
            head_share=float(dataset["head_share"]),
            tail_max_chars=int(dataset["tail_max_chars"]),
            seed=int(dataset["seed"]),
            tail_gap_pp=float(config["kpis"]["tail_gap_pp"]),
            recovery_share=float(config["kpis"]["recovery_share"]),
            token_share_vs_s2=float(config["kpis"]["token_share_vs_s2"]),
        )


def build_benchmark_sets(settings: BenchmarkSettings, xlsx_path: str | Path) -> BenchmarkSets:
    """Amostra validação+teste congelados (SPEC-DARU.md §3) com o seed do config."""
    return load_catalog_benchmark_set(
        xlsx_path,
        settings.categories,
        n_validation=settings.n_validation,
        n_test=settings.n_test,
        head_share=settings.head_share,
        tail_max_chars=settings.tail_max_chars,
        seed=settings.seed,
    )


def run_parallel[TIn, TOut](
    fn: Callable[[TIn], TOut], items: Sequence[TIn], *, max_workers: int
) -> list[TOut]:
    """Map com paralelismo moderado; a primeira exceção propaga."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(fn, items))


def _git_rev(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _s1_only_record(sample: CatalogSample, output: System1Output) -> BenchmarkRecord:
    return BenchmarkRecord(
        item_id=str(sample.sample_id),
        raw_text=sample.raw_text,
        canonical_category=sample.canonical_category,
        is_long_tail=sample.is_long_tail,
        s1_prediction=output.top_choice,
        s1_p_top=output.top_probability,
        s1_entropy=output.entropy,
        s1_latency_ms=output.latency_ms,
        routed_to_s2=False,
        s2_prediction=None,
        s2_reasoning=None,
        s2_revisor_insight=None,
        s2_decode_tokens=None,
        s2_latency_ms=None,
        final_prediction=output.top_choice,
        is_correct=output.top_choice == sample.canonical_category,
    )


def _s2_only_record(sample: CatalogSample, output: System2Output) -> BenchmarkRecord:
    """SC-B: sem chamada ao System 1 — os campos s1 ficam nos valores vazios."""
    return BenchmarkRecord(
        item_id=str(sample.sample_id),
        raw_text=sample.raw_text,
        canonical_category=sample.canonical_category,
        is_long_tail=sample.is_long_tail,
        s1_prediction="",
        s1_p_top=0.0,
        s1_entropy=0.0,
        s1_latency_ms=0.0,
        routed_to_s2=True,
        s2_prediction=output.predicted_category,
        s2_reasoning=output.reasoning,
        s2_revisor_insight=output.revisor_insight,
        s2_decode_tokens=output.decode_tokens,
        s2_latency_ms=output.latency_ms,
        final_prediction=output.predicted_category,
        is_correct=output.predicted_category == sample.canonical_category,
    )


def _cascade_record(
    sample: CatalogSample, s1: System1Output, s2: System2Output
) -> BenchmarkRecord:
    """Item roteado: saídas dos dois sistemas; final = System 2."""
    return BenchmarkRecord(
        item_id=str(sample.sample_id),
        raw_text=sample.raw_text,
        canonical_category=sample.canonical_category,
        is_long_tail=sample.is_long_tail,
        s1_prediction=s1.top_choice,
        s1_p_top=s1.top_probability,
        s1_entropy=s1.entropy,
        s1_latency_ms=s1.latency_ms,
        routed_to_s2=True,
        s2_prediction=s2.predicted_category,
        s2_reasoning=s2.reasoning,
        s2_revisor_insight=s2.revisor_insight,
        s2_decode_tokens=s2.decode_tokens,
        s2_latency_ms=s2.latency_ms,
        final_prediction=s2.predicted_category,
        is_correct=s2.predicted_category == sample.canonical_category,
    )


class BenchmarkHarness:
    """Executa S1/S2 contra um único servidor, com warmup e ThreadPool(8)."""

    def __init__(
        self, settings: BenchmarkSettings, *, client: httpx.Client | None = None
    ) -> None:
        self._settings = settings
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(
            timeout=SYSTEM2_TIMEOUT_S
        )
        self._s2_engine = System2Engine(
            base_url=settings.base_url,
            model=settings.model,
            max_tokens=settings.max_tokens,
            client=self._client,
        )

    def close(self) -> None:
        self._s2_engine.close()
        if self._owns_client:
            self._client.close()

    def _system1(self, item_text: str) -> System1Output:
        return evaluate_typed_decision(
            item_text,
            self._settings.categories,
            model=self._settings.model,
            base_url=self._settings.base_url,
            timeout_s=SYSTEM1_TIMEOUT_S,
            client=self._client,
        )

    def _system2(self, item_text: str, s1_suggested: str | None = None) -> System2Output:
        return self._s2_engine.reason(
            item_text, self._settings.categories, s1_suggested=s1_suggested
        )

    def warmup(self, samples: Sequence[CatalogSample]) -> None:
        """Requisições descartadas por engine antes de cada cenário medido."""
        texts = [
            samples[index % len(samples)].raw_text
            for index in range(self._settings.warmup_requests)
        ]
        for text in texts:
            self._system1(text)
        for text in texts:
            self._system2(text)

    def run_system1_scenario(
        self, samples: Sequence[CatalogSample]
    ) -> list[BenchmarkRecord]:
        """SC-A: System 1 sozinho em todos os itens."""
        self.warmup(samples)
        outputs = run_parallel(
            lambda sample: self._system1(sample.raw_text),
            samples,
            max_workers=self._settings.max_workers,
        )
        return [
            _s1_only_record(sample, output)
            for sample, output in zip(samples, outputs, strict=True)
        ]

    def run_system2_scenario(
        self, samples: Sequence[CatalogSample]
    ) -> list[BenchmarkRecord]:
        """SC-B: System 2 sozinho em todos os itens (sem sugestão do S1)."""
        self.warmup(samples)
        outputs = run_parallel(
            lambda sample: self._system2(sample.raw_text),
            samples,
            max_workers=self._settings.max_workers,
        )
        return [
            _s2_only_record(sample, output)
            for sample, output in zip(samples, outputs, strict=True)
        ]

    def run_cascade_scenario(
        self, samples: Sequence[CatalogSample], *, alpha: float
    ) -> list[BenchmarkRecord]:
        """SC-C: S1 em todos, S2 só nos itens com P_top < alpha (regra R_alpha)."""
        self.warmup(samples)
        s1_outputs = run_parallel(
            lambda sample: self._system1(sample.raw_text),
            samples,
            max_workers=self._settings.max_workers,
        )
        routed_flags = [
            not CascadeRouter.decide(output.top_probability, alpha)
            for output in s1_outputs
        ]
        routed_pairs = [
            (sample, output.top_choice)
            for sample, output, routed in zip(samples, s1_outputs, routed_flags, strict=True)
            if routed
        ]
        s2_outputs = run_parallel(
            lambda pair: self._system2(pair[0].raw_text, s1_suggested=pair[1]),
            routed_pairs,
            max_workers=self._settings.max_workers,
        )
        s2_by_id = {
            str(sample.sample_id): output
            for (sample, _), output in zip(routed_pairs, s2_outputs, strict=True)
        }
        records: list[BenchmarkRecord] = []
        for sample, output, routed in zip(samples, s1_outputs, routed_flags, strict=True):
            if routed:
                records.append(_cascade_record(sample, output, s2_by_id[str(sample.sample_id)]))
            else:
                records.append(_s1_only_record(sample, output))
        return records

    def run_validation_phase(
        self, samples: Sequence[CatalogSample]
    ) -> list[BenchmarkRecord]:
        """S1 e S2 uma única vez por item; o sweep de alpha é aritmética pura.

        Cada registro carrega as duas saídas (`routed_to_s2=True` marca que o S2
        está disponível); o roteamento real por alpha é decidido depois, sem
        novas chamadas HTTP.
        """
        self.warmup(samples)
        s1_outputs = run_parallel(
            lambda sample: self._system1(sample.raw_text),
            samples,
            max_workers=self._settings.max_workers,
        )
        pairs = [
            (sample, output.top_choice)
            for sample, output in zip(samples, s1_outputs, strict=True)
        ]
        s2_outputs = run_parallel(
            lambda pair: self._system2(pair[0].raw_text, s1_suggested=pair[1]),
            pairs,
            max_workers=self._settings.max_workers,
        )
        return [
            _cascade_record(sample, s1, s2)
            for sample, s1, s2 in zip(samples, s1_outputs, s2_outputs, strict=True)
        ]


def cascade_metrics_for_alpha(
    records: Sequence[BenchmarkRecord], alpha: float
) -> dict[str, float]:
    """Arimética pura sobre as saídas salvas: acurácia, taxa roteada, tokens."""
    if not records:
        raise ValueError("records must not be empty")
    n_correct = 0
    n_routed = 0
    decode_tokens = 0
    for record in records:
        routed = not CascadeRouter.decide(record.s1_p_top, alpha)
        if routed:
            n_routed += 1
            decode_tokens += record.s2_decode_tokens or 0
            prediction: str | None = record.s2_prediction
        else:
            prediction = record.s1_prediction
        n_correct += prediction == record.canonical_category
    return {
        "accuracy": n_correct / len(records),
        "routed_rate": n_routed / len(records),
        "decode_tokens": decode_tokens,
    }


def best_alpha(metrics: Mapping[str, Mapping[str, float]]) -> float:
    """Maior acurácia da cascata; empate decide pelo menor custo de tokens."""
    if not metrics:
        raise ValueError("metrics must not be empty")
    chosen = min(
        metrics,
        key=lambda alpha: (
            -metrics[alpha]["accuracy"],
            metrics[alpha]["decode_tokens"],
            float(alpha),
        ),
    )
    return float(chosen)


def alpha_sweep(
    records: Sequence[BenchmarkRecord],
    alphas: Sequence[float] = SWEEP_ALPHAS,
) -> dict[str, Any]:
    metrics = {f"{alpha:.2f}": cascade_metrics_for_alpha(records, alpha) for alpha in alphas}
    return {
        "alphas": metrics,
        "best_alpha": best_alpha(metrics),
        "rule": "maior acurácia da cascata; empate decide pelo menor custo de tokens",
    }


GATE_EPSILON: Final = 1e-9


def compute_kpis(
    records_a: Sequence[BenchmarkRecord],
    records_b: Sequence[BenchmarkRecord],
    records_c: Sequence[BenchmarkRecord],
    settings: BenchmarkSettings,
) -> dict[str, Any]:
    """Gates medidos uma única vez no teste congelado (SPEC-DARU.md §1).

    Comparações com tolerância de `GATE_EPSILON`: as métricas vêm de divisões
    de inteiros em ponto flutuante, e um gate no limite exato (APGR = 0.5)
    não pode reprovar por erro de arredondamento.
    """
    summary_a = summarize(records_a)
    summary_b = summarize(records_b)
    summary_c = summarize(records_c)

    tail_gap_pp: float | None = None
    if summary_a.accuracy_tail is not None and summary_b.accuracy_tail is not None:
        tail_gap_pp = (summary_b.accuracy_tail - summary_a.accuracy_tail) * 100.0
    recovery = apgr(summary_a.accuracy, summary_b.accuracy, summary_c.accuracy)
    token_share: float | None = None
    if summary_b.decode_tokens_total > 0:
        token_share = summary_c.decode_tokens_total / summary_b.decode_tokens_total

    return {
        "tail_gap_pp": {
            "required": settings.tail_gap_pp,
            "achieved": tail_gap_pp,
            "passed": tail_gap_pp is not None
            and tail_gap_pp >= settings.tail_gap_pp - GATE_EPSILON,
        },
        "recovery_share": {
            "required": settings.recovery_share,
            "achieved": recovery,
            "passed": recovery is not None
            and recovery >= settings.recovery_share - GATE_EPSILON,
        },
        "token_share_vs_s2": {
            "required": settings.token_share_vs_s2,
            "achieved": token_share,
            "passed": token_share is not None
            and token_share <= settings.token_share_vs_s2 + GATE_EPSILON,
        },
    }


@dataclass(frozen=True)
class PilotOutcome:
    """Resultado do gate de 100 itens (SPEC-DARU.md §6)."""

    n_items: int
    n_tail: int
    s1_tail_accuracy: float | None
    s2_tail_accuracy: float | None
    tail_gap_pp: float | None
    required_gap_pp: float
    passed: bool
    scenario_a: ScenarioSummary
    scenario_b: ScenarioSummary


def select_pilot_samples(
    samples: Sequence[CatalogSample],
    *,
    n_head: int | None = None,
    n_tail: int | None = None,
) -> list[CatalogSample]:
    """Primeiros itens de validação estratificados cabeça/cauda (70/30)."""
    if n_head is None:
        n_head = PILOT_HEAD
    if n_tail is None:
        n_tail = PILOT_TAIL
    head = [sample for sample in samples if not sample.is_long_tail]
    tail = [sample for sample in samples if sample.is_long_tail]
    if len(head) < n_head or len(tail) < n_tail:
        raise ValueError(
            f"pilot needs {n_head} head/{n_tail} tail items, "
            f"got {len(head)}/{len(tail)}"
        )
    return head[:n_head] + tail[:n_tail]


def run_pilot(
    harness: BenchmarkHarness,
    validation_samples: Sequence[CatalogSample],
    *,
    required_gap_pp: float,
    n_head: int | None = None,
    n_tail: int | None = None,
) -> PilotOutcome:
    """SC-A e SC-B nos itens do pilot; gate = S2 supera S1 na cauda em >= X pp."""
    pilot = select_pilot_samples(validation_samples, n_head=n_head, n_tail=n_tail)
    records_a = harness.run_system1_scenario(pilot)
    records_b = harness.run_system2_scenario(pilot)
    tail_a = slice_accuracy(records_a, is_long_tail=True)
    tail_b = slice_accuracy(records_b, is_long_tail=True)
    gap_pp = (
        (tail_b - tail_a) * 100.0
        if tail_a is not None and tail_b is not None
        else None
    )
    return PilotOutcome(
        n_items=len(pilot),
        n_tail=n_tail if n_tail is not None else PILOT_TAIL,
        s1_tail_accuracy=tail_a,
        s2_tail_accuracy=tail_b,
        tail_gap_pp=gap_pp,
        required_gap_pp=required_gap_pp,
        passed=gap_pp is not None and gap_pp >= required_gap_pp,
        scenario_a=summarize(records_a),
        scenario_b=summarize(records_b),
    )


def format_pilot_report(outcome: PilotOutcome) -> str:
    def _pct(value: float | None) -> str:
        return f"{value:.1%}" if value is not None else "n/a"

    gap_line = (
        f"gap: {outcome.tail_gap_pp:.1f} pp "
        f"(necessário >= {outcome.required_gap_pp:.1f} pp)"
        if outcome.tail_gap_pp is not None
        else "gap: n/a (sem itens de cauda)"
    )
    return "\n".join(
        [
            "== PILOT (SPEC-DARU.md §6) ==",
            f"itens: {outcome.n_items} ({outcome.n_items - outcome.n_tail} cabeça / "
            f"{outcome.n_tail} cauda)",
            f"acurácia na cauda — System 1: {_pct(outcome.s1_tail_accuracy)} | "
            f"System 2: {_pct(outcome.s2_tail_accuracy)}",
            gap_line,
            "GATE: PASSOU" if outcome.passed else "GATE: REPROVADO",
        ]
    )


def write_results(payload: Mapping[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
    return output


def assemble_payload(
    *,
    config: Mapping[str, Any],
    settings: BenchmarkSettings,
    started_at: str,
    finished_at: str,
    git_rev: str,
    alpha_used: float,
    alpha_source: str,
    validation_records: Sequence[BenchmarkRecord],
    records_a: Sequence[BenchmarkRecord],
    records_b: Sequence[BenchmarkRecord],
    records_c: Sequence[BenchmarkRecord],
) -> dict[str, Any]:
    """Monta o raw_benchmark_data.json: meta, cenários, sweep, sumários, gates."""
    return {
        "meta": {
            "started_at": started_at,
            "finished_at": finished_at,
            "model": settings.model,
            "base_url": settings.base_url,
            "git_rev": git_rev,
            "alpha_used": alpha_used,
            "alpha_source": alpha_source,
            "config": dict(config),
            "dataset_sizes": {
                "validation": len(validation_records),
                "test": len(records_a),
            },
        },
        "scenarios": {
            "SC-A": [asdict(record) for record in records_a],
            "SC-B": [asdict(record) for record in records_b],
            "SC-C": [asdict(record) for record in records_c],
            "VALIDATION": [asdict(record) for record in validation_records],
        },
        "alpha_sweep": alpha_sweep(validation_records),
        "summaries": {
            "SC-A": asdict(summarize(records_a)),
            "SC-B": asdict(summarize(records_b)),
            "SC-C": asdict(summarize(records_c)),
            "VALIDATION": asdict(summarize(validation_records)),
        },
        "kpis": compute_kpis(records_a, records_b, records_c, settings),
    }


def execute_pilot(
    settings: BenchmarkSettings,
    *,
    config: Mapping[str, Any],
    xlsx_path: str | Path,
    results_path: str | Path = PILOT_RESULTS_PATH,
) -> int:
    """Roda o pilot, imprime o gate e devolve o código de saída (0 passou)."""
    sets = build_benchmark_sets(settings, xlsx_path)
    harness = BenchmarkHarness(settings)
    try:
        outcome = run_pilot(
            harness, sets.validation, required_gap_pp=settings.tail_gap_pp
        )
    finally:
        harness.close()
    print(format_pilot_report(outcome))
    write_results(asdict(outcome) | {"written_at": datetime.now(UTC).isoformat()}, results_path)
    return 0 if outcome.passed else 1


def execute_full_benchmark(
    settings: BenchmarkSettings,
    *,
    config: Mapping[str, Any],
    xlsx_path: str | Path,
    alpha_override: float | None = None,
    results_path: str | Path = RESULTS_PATH,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Validação (sweep de alpha) + teste congelado nos três cenários."""
    started_at = datetime.now(UTC)
    sets = build_benchmark_sets(settings, xlsx_path)
    harness = BenchmarkHarness(settings)
    try:
        validation_records = harness.run_validation_phase(sets.validation)
        sweep = alpha_sweep(validation_records)
        if alpha_override is not None:
            alpha_used, alpha_source = float(alpha_override), "override"
        else:
            alpha_used, alpha_source = sweep["best_alpha"], "validation_sweep"
        records_a = harness.run_system1_scenario(sets.test)
        records_b = harness.run_system2_scenario(sets.test)
        records_c = harness.run_cascade_scenario(sets.test, alpha=alpha_used)
    finally:
        harness.close()
    payload = assemble_payload(
        config=config,
        settings=settings,
        started_at=started_at.isoformat(),
        finished_at=datetime.now(UTC).isoformat(),
        git_rev=_git_rev(Path(repo_root)),
        alpha_used=alpha_used,
        alpha_source=alpha_source,
        validation_records=validation_records,
        records_a=records_a,
        records_b=records_b,
        records_c=records_c,
    )
    write_results(payload, results_path)
    return payload
