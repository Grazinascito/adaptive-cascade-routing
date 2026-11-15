"""Métricas puras do benchmark DARU (PR4) e o registro bruto por item.

Sem HTTP e sem router: recebem `BenchmarkRecord` prontos (ou números) para
serem testáveis isoladamente. `summary.py` compõe estas funções por cenário.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    """Medições de um item nos três papéis da cascata (spec DARU §5)."""

    item_id: str
    raw_text: str
    canonical_category: str
    is_long_tail: bool
    s1_prediction: str
    s1_p_top: float
    s1_entropy: float
    s1_latency_ms: float
    routed_to_s2: bool
    s2_prediction: str | None
    s2_reasoning: str | None
    s2_revisor_insight: str | None
    s2_decode_tokens: int | None
    s2_latency_ms: float | None
    final_prediction: str
    is_correct: bool


def accuracy(records: Sequence[BenchmarkRecord]) -> float:
    """Fração de acertos (`is_correct`) sobre todos os registros."""
    if not records:
        raise ValueError("records must not be empty")
    return sum(1 for record in records if record.is_correct) / len(records)


def slice_accuracy(
    records: Sequence[BenchmarkRecord], *, is_long_tail: bool
) -> float | None:
    """Acurácia restrita a um corte (cauda = descrição com <= 20 caracteres).

    None quando o corte não tem registros (cenário só-cabeça/só-cauda).
    """
    selected = [record for record in records if record.is_long_tail == is_long_tail]
    if not selected:
        return None
    return sum(1 for record in selected if record.is_correct) / len(selected)


def apgr(acc_s1: float, acc_s2: float, acc_cascade: float) -> float | None:
    """Average Performance Gap Recovered (RouteLLM, spec DARU §5).

    None quando o denominador (gap S2 − S1) é <= 0: sem gap não há nada a
    recuperar, e gap negativo (S1 >= S2) torna a razão sem sentido.
    """
    gap = acc_s2 - acc_s1
    if gap <= 0.0:
        return None
    return (acc_cascade - acc_s1) / gap


def percentile(values: Sequence[float], q: float) -> float:
    """Percentil q (0–100) com interpolação linear, estilo numpy.

    rank = q/100 * (n − 1); entre dois vizinhos, pondera pela fração do rank.
    """
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be in [0, 100], got {q!r}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (q / 100.0) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def latency_percentiles(values: Sequence[float]) -> dict[str, float]:
    """p50/p95/p99 exigidos pela spec DARU §5 (interpolação linear)."""
    return {
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def total_decode_tokens(records: Sequence[BenchmarkRecord]) -> int:
    """Soma dos tokens de saída do System 2 (itens não roteados contam 0)."""
    return sum(record.s2_decode_tokens or 0 for record in records)
