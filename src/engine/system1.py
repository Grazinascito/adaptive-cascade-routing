"""System1Engine — decisão tipada de 1 token no endpoint /v1/systemone do vLLM.

Componente C do design doc (docs/adaptive-cascade-design-doc.md §5.3);
stories correspondentes em docs/engine-router-spec.md (Story 2). O endpoint
é o Structured Decisions API do vLLM (>= 0.32.0): a request declara a
pergunta `choice` com as opções em `criteria` e a resposta já traz as
probabilidades por opção (softmax do servidor); o engine converte para
ln(P) e delega as métricas ao `compute_metrics` (Componente D).
"""

import math
import time
from collections.abc import Mapping, Sequence
from typing import Any, Final, TypedDict

import httpx

from src.data.loader import SYSTEM1_SYSTEM_PREFIX, Label
from src.engine.router import compute_metrics

DEFAULT_MODEL: Final = "Qwen/Qwen3-4B-AWQ"
QUESTION_ID: Final = "nli_label"
QUESTION_INSTRUCTIONS: Final = (
    "Does the premise entail the hypothesis? Choose exactly one of the allowed options."
)


class System1Output(TypedDict):
    top_choice: Label

    top_probability: float

    logprobs: dict[str, float]

    probabilities: dict[Label, float]

    entropy: float

    latency_ms: float


def _build_request_body(
    state: str,
    choices: Sequence[Label],
    model: str,
    instructions: str,
) -> dict[str, Any]:
    return {
        "model": model,
        "state": state,
        "instructions": instructions,
        "questions": {
            QUESTION_ID: {
                "type": "choice",
                "instructions": QUESTION_INSTRUCTIONS,
                "criteria": {choice: None for choice in choices},
            }
        },
    }


def _logprobs_from_probabilities(
    raw: Mapping[str, float],
    choices: Sequence[Label],
) -> dict[str, float]:
    """Converte as probabilidades do servidor em logprobs naturais dos `choices`.

    Chaves fora de `choices` são ignoradas (mesma política do router); opções
    ausentes ou com probabilidade 0 ficam de fora e recebem `LOGPROB_FLOOR`
    via `compute_metrics`. Retorna `dict[str, float]` (e não `dict[Label, ...]`)
    porque `Mapping` é invariante na chave e `compute_metrics` espera
    `Mapping[str, float]`.
    """
    logprobs: dict[str, float] = {
        choice: math.log(raw[choice]) for choice in choices if raw.get(choice, 0.0) > 0.0
    }
    return logprobs


def evaluate_typed_decision(
    state: str,
    choices: Sequence[Label],
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = "http://127.0.0.1:8000/v1",
    instructions: str = SYSTEM1_SYSTEM_PREFIX,
    timeout_s: float = 10.0,
    client: httpx.Client | None = None,
) -> System1Output:
    """Executa a decisão tipada de 1 token (FR-002) e retorna as métricas de saída.

    Valida `choices` antes de qualquer HTTP (cenário de aceite 2 da Story 2):
    pelo menos dois rótulos únicos e não vazios. O prompt fixo vai no campo
    `instructions` (igual em toda query, para o prefix caching) e o par
    variável no `state`. Erros de status/timeout/conexão do httpx propagam —
    o fallback gracioso é responsabilidade do harness (Componente G).
    """
    choices_list = list(choices)
    if len(choices_list) < 2 or len(set(choices_list)) != len(choices_list):
        raise ValueError(f"choices must have at least two unique labels, got {choices_list!r}")
    if any(not choice.strip() for choice in choices_list):
        raise ValueError(f"choices must be non-blank labels, got {choices_list!r}")

    body = _build_request_body(state, choices_list, model, instructions)

    owns_client = client is None
    active_client = client if client is not None else httpx.Client()
    try:
        started = time.perf_counter()
        response = active_client.post(f"{base_url}/systemone", json=body, timeout=timeout_s)
        latency_ms = (time.perf_counter() - started) * 1000.0
        response.raise_for_status()
    finally:
        if owns_client:
            active_client.close()

    answer = response.json()["answers"][QUESTION_ID]
    logprobs = _logprobs_from_probabilities(answer["probabilities"], choices_list)
    metrics = compute_metrics(logprobs, choices_list)
    return System1Output(
        top_choice=metrics["top_choice"],
        top_probability=metrics["confidence"],
        logprobs=logprobs,
        probabilities=metrics["probabilities"],
        entropy=metrics["entropy"],
        latency_ms=latency_ms,
    )
