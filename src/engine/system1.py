"""System1Engine — typed 1-token decision on the vLLM /v1/systemone endpoint.

Restored from commit 79a8654 (vLLM >= 0.32.0 Structured Decisions API) and
adapted to DARU: the food/beverage categories arrive as a parameter, each
becomes an option with a 1-letter alias (A–L) in the `choice` question, and
the response already carries the per-alias probability (server softmax).
The client renormalizes (softmax over the log-probs, with a floor for
zero-probability options) before computing P_top and entropy. The request
carries NO `max_tokens` and no `logprobs` — the endpoint rejects both with
HTTP 400.
"""

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import httpx

DEFAULT_MODEL: Final = "Qwen/Qwen3-4B-AWQ"
DEFAULT_BASE_URL: Final = "http://127.0.0.1:8000/v1"
DEFAULT_TIMEOUT_S: Final = 10.0

ALIASES: Final = "ABCDEFGHIJKL"
LOGPROB_FLOOR: Final = -100.0

QUESTION_ID: Final = "category"
QUESTION_INSTRUCTIONS: Final = (
    "Qual apelido classifica corretamente o item? Escolha exatamente uma das opções permitidas."
)
SYSTEM1_INSTRUCTIONS: Final = (
    "Você é o System 1 do roteador de categorias de comida e bebida. Classifique o item "
    "de nota fiscal em exatamente uma das categorias permitidas, cada uma identificada "
    "pelo seu apelido de uma letra de A a L."
)


@dataclass(frozen=True, slots=True)
class System1Output:
    top_choice: str
    top_probability: float
    raw_probabilities: dict[str, float]
    normalized_probabilities: dict[str, float]
    entropy: float
    latency_ms: float


def assign_aliases(allowed_categories: Sequence[str]) -> dict[str, str]:
    """Maps alias (A–L, by position) -> original category name.

    Compares lowercase to detect duplicates; validates before any HTTP.
    """
    categories = list(allowed_categories)
    if not categories:
        raise ValueError("allowed_categories must not be empty")
    if len(categories) > len(ALIASES):
        raise ValueError(
            f"allowed_categories supports at most {len(ALIASES)} entries, got {len(categories)}"
        )
    if len(categories) < 2:
        raise ValueError(f"allowed_categories needs at least two entries, got {categories!r}")
    if any(not category.strip() for category in categories):
        raise ValueError(f"allowed_categories must be non-blank names, got {categories!r}")
    normalized = [category.strip().lower() for category in categories]
    if len(set(normalized)) != len(normalized):
        raise ValueError(
            f"allowed_categories must be unique after lowercase normalization, got {categories!r}"
        )
    return dict(zip(ALIASES[: len(categories)], categories, strict=True))


def _renormalized_probabilities(
    raw: Mapping[str, float],
    aliases: Sequence[str],
) -> dict[str, float]:
    """Softmax over ln(P) of the aliases; missing or zero P gets LOGPROB_FLOOR.

    Renormalizes server outputs that do not sum to 1: exp(ln(P)) = P and
    the softmax divides by the total. Subtracting the max avoids overflow
    in exp.
    """
    if not any(raw.get(alias, 0.0) > 0.0 for alias in aliases):
        raise ValueError(f"no positive probability for any of the aliases {list(aliases)}")
    logprobs = {
        alias: math.log(raw[alias]) if raw.get(alias, 0.0) > 0.0 else LOGPROB_FLOOR
        for alias in aliases
    }
    top_logprob = max(logprobs.values())
    exps = {alias: math.exp(value - top_logprob) for alias, value in logprobs.items()}
    total = sum(exps.values())
    return {alias: value / total for alias, value in exps.items()}


def _shannon_entropy_bits(probabilities: Mapping[str, float]) -> float:
    return -sum(p * math.log2(p) for p in probabilities.values() if p > 0.0)


def _build_request_body(
    item_text: str,
    aliases: Sequence[str],
    model: str,
    instructions: str,
) -> dict[str, object]:
    return {
        "model": model,
        "state": item_text,
        "instructions": instructions,
        "questions": {
            QUESTION_ID: {
                "type": "choice",
                "instructions": QUESTION_INSTRUCTIONS,
                "criteria": {alias: None for alias in aliases},
            }
        },
    }


def evaluate_typed_decision(
    item_text: str,
    allowed_categories: Sequence[str],
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    instructions: str = SYSTEM1_INSTRUCTIONS,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    client: httpx.Client | None = None,
) -> System1Output:
    """Runs the 1-token decision; returns P_top/entropy already renormalized.

    `instructions` (fixed prefix, identical in every query for vLLM prefix
    caching) goes in the field of the same name; the variable item text
    goes in `state`. httpx status errors propagate — graceful fallback is
    the harness's job.
    """
    if not item_text.strip():
        raise ValueError(f"item_text must be non-blank, got {item_text!r}")

    alias_to_category = assign_aliases(allowed_categories)
    aliases = list(alias_to_category)
    body = _build_request_body(item_text, aliases, model, instructions)

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
    raw_probabilities = {
        alias: float(probability) for alias, probability in answer["probabilities"].items()
    }
    alias_probabilities = _renormalized_probabilities(raw_probabilities, aliases)
    normalized_probabilities = {
        alias_to_category[alias]: probability for alias, probability in alias_probabilities.items()
    }
    top_probability = max(normalized_probabilities.values())
    top_choice = next(
        category
        for category in alias_to_category.values()
        if normalized_probabilities[category] == top_probability
    )
    return System1Output(
        top_choice=top_choice,
        top_probability=top_probability,
        raw_probabilities=raw_probabilities,
        normalized_probabilities=normalized_probabilities,
        entropy=_shannon_entropy_bits(normalized_probabilities),
        latency_ms=latency_ms,
    )
