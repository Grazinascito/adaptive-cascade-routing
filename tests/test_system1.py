"""Testes do System 1 DARU: contrato /v1/systemone restaurado do 79a8654.

O vLLM real nunca é chamado — httpx.MockTransport captura a request e serve
respostas enlatadas imitando o Structured Decisions API (probabilidades prontas
por apelido; sem max_tokens/logprobs na request, que o endpoint rejeita).
"""

import json
import math
from typing import Any

import httpx
import pytest

from src.engine import system1

CATEGORIES: list[str] = [
    "MERCEARIA",
    "FRIOS E LATICINIOS",
    "AÇOUGUE E PEIXARIA",
    "HORTIFRUTI",
    "PREPARADOS E CONGELADOS",
    "PADARIA E CONFEITARIA",
    "BEBIDAS ALCOÓLICAS",
    "SUCOS E REFRESCOS",
    "REFRIGERANTES",
    "CAFÉ E CHÁ",
    "CONSERVAS",
    "TEMPEROS E CONDIMENTOS",
]
ITEM = "APRESUNTADO REZENDE PEC KG"


def _uniform() -> dict[str, float]:
    return {alias: 1.0 / 12.0 for alias in system1.ALIASES}


def _client_returning(
    probabilities: dict[str, float],
    requests: list[dict[str, Any]],
    status: int = 200,
) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append({"url": str(request.url), "body": json.loads(request.content)})
        if status != 200:
            return httpx.Response(status, json={"error": "boom"})
        top = max(probabilities, key=lambda alias: probabilities[alias])
        return httpx.Response(
            200,
            json={
                "id": "decision-1",
                "object": "structured_decision",
                "created": 0,
                "model": system1.DEFAULT_MODEL,
                "answers": {
                    system1.QUESTION_ID: {
                        "type": "choice",
                        "choice": top,
                        "probabilities": probabilities,
                        "confidence": probabilities[top],
                    }
                },
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def _all_keys(node: Any) -> set[str]:
    if isinstance(node, dict):
        return set(node) | {key for value in node.values() for key in _all_keys(value)}
    if isinstance(node, list):
        return {key for value in node for key in _all_keys(value)}
    return set()


def _evaluate(
    probabilities: dict[str, float],
    requests: list[dict[str, Any]],
    **kwargs: Any,
) -> system1.System1Output:
    client = _client_returning(probabilities, requests)
    return system1.evaluate_typed_decision(ITEM, CATEGORIES, client=client, **kwargs)


def test_request_body_restores_exact_systemone_contract() -> None:
    requests: list[dict[str, Any]] = []
    _evaluate(_uniform(), requests)

    assert len(requests) == 1
    request = requests[0]
    assert request["url"] == "http://127.0.0.1:8000/v1/systemone"
    body = request["body"]
    assert set(body) == {"model", "state", "instructions", "questions"}
    assert body["model"] == system1.DEFAULT_MODEL
    assert body["state"] == ITEM
    assert body["instructions"] == system1.SYSTEM1_INSTRUCTIONS
    questions = body["questions"]
    assert set(questions) == {system1.QUESTION_ID}
    question = questions[system1.QUESTION_ID]
    assert set(question) == {"type", "instructions", "criteria"}
    assert question["type"] == "choice"
    assert question["instructions"] == system1.QUESTION_INSTRUCTIONS
    assert set(question["criteria"]) == set(system1.ALIASES)
    assert all(value is None for value in question["criteria"].values())


def test_request_body_has_no_max_tokens_or_logprobs_anywhere() -> None:
    requests: list[dict[str, Any]] = []
    _evaluate(_uniform(), requests)

    assert not _all_keys(requests[0]["body"]) & {"max_tokens", "logprobs"}


def test_instruction_prefix_is_identical_across_requests() -> None:
    requests: list[dict[str, Any]] = []
    _evaluate(_uniform(), requests)
    _evaluate(_uniform(), requests)

    assert requests[0]["body"]["instructions"] == requests[1]["body"]["instructions"]
    assert requests[0]["body"]["questions"][system1.QUESTION_ID]["instructions"] == (
        requests[1]["body"]["questions"][system1.QUESTION_ID]["instructions"]
    )


def test_twelve_categories_map_bijectively_to_aliases_a_through_l() -> None:
    alias_to_category = system1.assign_aliases(CATEGORIES)

    assert set(alias_to_category) == set("ABCDEFGHIJKL")
    assert len(set(alias_to_category.values())) == 12
    assert set(alias_to_category.values()) == set(CATEGORIES)
    assert alias_to_category["A"] == CATEGORIES[0]
    assert alias_to_category["L"] == CATEGORIES[-1]


def test_duplicate_categories_after_lowercase_normalization_raise_value_error() -> None:
    with pytest.raises(ValueError, match="lowercase normalization"):
        system1.assign_aliases(["MERCEARIA", "mercearia", "HORTIFRUTI"])


def test_softmax_renormalizes_input_that_does_not_sum_to_one() -> None:
    requests: list[dict[str, Any]] = []
    raw = {alias: 0.0 for alias in system1.ALIASES}
    raw["A"] = 0.40
    raw["C"] = 0.20

    output = _evaluate(raw, requests)

    assert sum(raw.values()) == pytest.approx(0.60)
    assert sum(output.normalized_probabilities.values()) == pytest.approx(1.0, abs=1e-12)
    assert set(output.normalized_probabilities) == set(CATEGORIES)
    assert output.normalized_probabilities[CATEGORIES[0]] == pytest.approx(2.0 / 3.0, abs=1e-12)
    assert output.normalized_probabilities[CATEGORIES[2]] == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert output.raw_probabilities == raw


def test_entropy_of_uniform_over_twelve_is_log2_12() -> None:
    requests: list[dict[str, Any]] = []

    output = _evaluate(_uniform(), requests)

    assert output.entropy == pytest.approx(math.log2(12.0), abs=1e-9)


def test_entropy_of_one_hot_is_approximately_zero() -> None:
    requests: list[dict[str, Any]] = []
    raw = {alias: 0.0 for alias in system1.ALIASES}
    raw["F"] = 1.0

    output = _evaluate(raw, requests)

    assert output.entropy == pytest.approx(0.0, abs=1e-12)
    assert output.top_choice == CATEGORIES[5]
    assert output.top_probability == pytest.approx(1.0, abs=1e-12)


def test_top_choice_and_probability_on_skewed_distribution() -> None:
    requests: list[dict[str, Any]] = []
    raw = {alias: 0.01 for alias in system1.ALIASES}
    raw["A"] = 0.30
    raw["C"] = 0.55

    output = _evaluate(raw, requests)

    assert sum(raw.values()) == pytest.approx(0.95)
    assert output.top_choice == CATEGORIES[2]
    assert output.top_probability == pytest.approx(0.55 / 0.95, abs=1e-12)
    assert output.top_probability > output.normalized_probabilities[CATEGORIES[0]]
    assert output.latency_ms >= 0.0


def test_empty_allowed_categories_raises_value_error_before_http() -> None:
    requests: list[dict[str, Any]] = []
    client = _client_returning(_uniform(), requests)

    with pytest.raises(ValueError, match="allowed_categories"):
        system1.evaluate_typed_decision(ITEM, [], client=client)

    assert requests == []


def test_http_400_propagates_as_http_status_error() -> None:
    requests: list[dict[str, Any]] = []
    client = _client_returning({}, requests, status=400)

    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        system1.evaluate_typed_decision(ITEM, CATEGORIES, client=client)

    assert excinfo.value.response.status_code == 400
