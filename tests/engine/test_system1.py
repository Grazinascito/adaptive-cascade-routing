"""Testes do Componente C: decisão tipada de 1 token no endpoint /v1/systemone.

O vLLM real nunca é chamado: as respostas vêm de httpx.MockTransport no
limite da rede, imitando o contrato real do Structured Decisions API
(probabilities prontas por opção; sem max_tokens/logprobs/choices na request).
Corresponde à Story 2 de docs/engine-router-spec.md.
"""

import json
import math
from typing import cast

import httpx
import pytest

from src.data.loader import SYSTEM1_SYSTEM_PREFIX, Label
from src.engine import system1

_CHOICES: list[Label] = ["Entailment", "Contradiction"]
_STATE = (
    "<premise>\nO gato dorme no sofá.\n</premise>\n"
    "<hypothesis>\nO gato está dormindo.\n</hypothesis>"
)


def _systemone_response(probabilities: dict[str, float]) -> httpx.Response:
    top = max(probabilities, key=lambda key: probabilities[key])
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
            "diagnostics": {system1.QUESTION_ID: {"label_mass": 1.0, "argmax_is_label": True}},
        },
    )


def _client_returning(
    probabilities: dict[str, float],
    requests: list[dict[str, object]],
    status: int = 200,
) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append({"url": str(request.url), "body": json.loads(request.content)})
        if status != 200:
            return httpx.Response(status, json={"error": "boom"})
        return _systemone_response(probabilities)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_evaluate_typed_decision_returns_metrics_consistent_with_server_probabilities() -> None:
    requests: list[dict[str, object]] = []
    client = _client_returning({"Entailment": 0.9, "Contradiction": 0.1}, requests)

    output = system1.evaluate_typed_decision(_STATE, _CHOICES, client=client)

    assert requests == [
        {
            "url": "http://127.0.0.1:8000/v1/systemone",
            "body": {
                "model": system1.DEFAULT_MODEL,
                "state": _STATE,
                "instructions": SYSTEM1_SYSTEM_PREFIX,
                "questions": {
                    system1.QUESTION_ID: {
                        "type": "choice",
                        "instructions": system1.QUESTION_INSTRUCTIONS,
                        "criteria": {"Entailment": None, "Contradiction": None},
                    }
                },
            },
        }
    ]
    assert output["top_choice"] == "Entailment"
    assert output["top_probability"] == pytest.approx(0.9, abs=1e-9)
    assert output["logprobs"]["Entailment"] == pytest.approx(math.log(0.9))
    assert output["logprobs"]["Contradiction"] == pytest.approx(math.log(0.1))
    assert output["probabilities"]["Contradiction"] == pytest.approx(0.1, abs=1e-9)
    assert output["entropy"] == pytest.approx(0.468996, abs=1e-6)
    assert output["latency_ms"] >= 0.0


def test_evaluate_typed_decision_applies_logprob_floor_when_server_omits_one_choice() -> None:
    requests: list[dict[str, object]] = []
    client = _client_returning({"Contradiction": 0.8}, requests)

    output = system1.evaluate_typed_decision(_STATE, _CHOICES, client=client)

    assert output["top_choice"] == "Contradiction"
    assert output["top_probability"] == pytest.approx(1.0, abs=1e-9)
    assert "Entailment" not in output["logprobs"]


def test_evaluate_typed_decision_raises_value_error_before_http_when_choices_are_invalid() -> None:
    requests: list[dict[str, object]] = []
    client = _client_returning({"Entailment": 0.9, "Contradiction": 0.1}, requests)

    with pytest.raises(ValueError, match="choices"):
        system1.evaluate_typed_decision(_STATE, ["Entailment"], client=client)

    assert requests == []


def test_evaluate_typed_decision_rejects_out_of_vocabulary_choice_before_http() -> None:
    requests: list[dict[str, object]] = []
    client = _client_returning({"Entailment": 0.9, "Contradiction": 0.1}, requests)
    # "Neutral" fica fora do vocabulário de Label de propósito; o cast só
    # acalma o mypy — o guard testado é o de runtime.
    oov_choices = cast("list[Label]", ["Entailment", "Neutral"])

    with pytest.raises(ValueError, match="choices"):
        system1.evaluate_typed_decision(_STATE, oov_choices, client=client)

    assert requests == []


def test_evaluate_typed_decision_raises_value_error_when_no_choice_has_probability() -> None:
    requests: list[dict[str, object]] = []
    client = _client_returning({"Neutral": 1.0}, requests)

    with pytest.raises(ValueError, match="signal"):
        system1.evaluate_typed_decision(_STATE, _CHOICES, client=client)


def test_evaluate_typed_decision_ignores_probability_keys_outside_choices() -> None:
    requests: list[dict[str, object]] = []
    client = _client_returning(
        {"Entailment": 0.25, "Contradiction": 0.25, "Neutral": 0.5}, requests
    )

    output = system1.evaluate_typed_decision(_STATE, _CHOICES, client=client)

    assert set(output["probabilities"]) == {"Entailment", "Contradiction"}
    assert output["top_choice"] == "Entailment"
    assert output["top_probability"] == pytest.approx(0.5)


def test_evaluate_typed_decision_propagates_http_status_errors() -> None:
    requests: list[dict[str, object]] = []
    client = _client_returning({}, requests, status=500)

    with pytest.raises(httpx.HTTPStatusError):
        system1.evaluate_typed_decision(_STATE, _CHOICES, client=client)
