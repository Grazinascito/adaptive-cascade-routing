"""Testes do Componente F: System 2 com Chain-of-Thought e saída JSON estrita.

O vLLM real nunca é chamado: as respostas vêm de httpx.MockTransport no
limite da rede, imitando /v1/chat/completions. Corresponde à Story 1 de
docs/system2-evaluator-spec.md (edge cases de JSON inválido, regex e vazio).
"""

import json
import logging

import httpx
import pytest

from src.engine import system1, system2

_PREMISE = "O gato dorme no sofá."
_HYPOTHESIS = "O gato está acordado."

_VALID_CONTENT = json.dumps(
    {
        "reasoning": "A premissa diz que o gato dorme; acordado é o oposto.",
        "label": "Contradiction",
    },
    ensure_ascii=False,
)


def _chat_response(content: str, *, usage: dict[str, int] | None) -> httpx.Response:
    body: dict[str, object] = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": system1.DEFAULT_MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
    }
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(200, json=body)


def _client_returning(
    content: str,
    requests: list[dict[str, object]],
    *,
    usage: dict[str, int] | None = None,
) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append({"url": str(request.url), "body": json.loads(request.content)})
        return _chat_response(content, usage=usage)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_execute_reasoning_sends_cot_json_prompt_with_zero_temperature_and_max_tokens_256() -> None:
    requests: list[dict[str, object]] = []
    client = _client_returning(
        _VALID_CONTENT,
        requests,
        usage={"prompt_tokens": 120, "completion_tokens": 80},
    )

    output = system2.execute_reasoning(_PREMISE, _HYPOTHESIS, client=client)

    assert len(requests) == 1
    request = requests[0]
    assert request["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    body = request["body"]
    assert isinstance(body, dict)
    assert body["model"] == system1.DEFAULT_MODEL
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 256
    message = body["messages"][0]["content"]
    assert _PREMISE in message
    assert _HYPOTHESIS in message

    assert output["reasoning"] == "A premissa diz que o gato dorme; acordado é o oposto."
    assert output["label"] == "Contradiction"
    assert output["fallback_used"] is False
    assert output["prompt_tokens"] == 120
    assert output["completion_tokens"] == 80
    assert output["latency_ms"] >= 0.0


def test_execute_reasoning_falls_back_to_regex_when_json_decoding_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    plain_text = "O rótulo correto é Contradiction porque a hipótese contradiz a premissa."
    requests: list[dict[str, object]] = []
    client = _client_returning(
        plain_text, requests, usage={"prompt_tokens": 100, "completion_tokens": 20}
    )
    caplog.set_level(logging.WARNING)

    output = system2.execute_reasoning(_PREMISE, _HYPOTHESIS, client=client)

    assert output["label"] == "Contradiction"
    assert output["fallback_used"] is True
    assert output["reasoning"] == plain_text
    assert "regex fallback" in caplog.text


def test_execute_reasoning_defaults_to_contradiction_with_fallback_flag_on_empty_output(
    caplog: pytest.LogCaptureFixture,
) -> None:
    requests: list[dict[str, object]] = []
    client = _client_returning("", requests, usage={"prompt_tokens": 90, "completion_tokens": 0})
    caplog.set_level(logging.WARNING)

    output = system2.execute_reasoning(_PREMISE, _HYPOTHESIS, client=client)

    assert output["label"] == "Contradiction"
    assert output["fallback_used"] is True
    assert output["reasoning"] == ""
    assert "defaulting to" in caplog.text


def test_execute_reasoning_falls_back_when_json_is_valid_but_label_is_unknown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    unknown_label = json.dumps({"reasoning": "alguma análise", "label": "Neutral"})
    requests: list[dict[str, object]] = []
    client = _client_returning(
        unknown_label, requests, usage={"prompt_tokens": 90, "completion_tokens": 30}
    )
    caplog.set_level(logging.WARNING)

    output = system2.execute_reasoning(_PREMISE, _HYPOTHESIS, client=client)

    assert output["label"] == "Contradiction"
    assert output["fallback_used"] is True


def test_execute_reasoning_defaults_tokens_to_zero_when_usage_is_absent() -> None:
    requests: list[dict[str, object]] = []
    client = _client_returning(_VALID_CONTENT, requests, usage=None)

    output = system2.execute_reasoning(_PREMISE, _HYPOTHESIS, client=client)

    assert output["prompt_tokens"] == 0
    assert output["completion_tokens"] == 0
    assert output["fallback_used"] is False
