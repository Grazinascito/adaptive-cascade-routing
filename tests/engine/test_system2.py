"""Testes do Componente F pós-pivot: System 2 NCM com Chain-of-Thought e JSON.

O vLLM real nunca é chamado: as respostas vêm de httpx.MockTransport no
limite da rede, imitando /v1/chat/completions. Cobre JSON com ncm_code,
fallback regex e sentinela default.
"""

import json
import logging

import httpx
import pytest

from src.engine import serving, system2

_ITEM = "CACHACA BOAZINHA BALSAMO UND 1 X 1 670ML"

_VALID_CONTENT = json.dumps(
    {
        "reasoning": "Cachaça é aguardente de cana; capítulo 22, aguardentes.",
        "ncm_code": "22084000",
    },
    ensure_ascii=False,
)


def _chat_response(content: str, *, usage: dict[str, int] | None) -> httpx.Response:
    body: dict[str, object] = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": serving.DEFAULT_MODEL,
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


def test_json_contains_valid_ncm_code_and_item_name() -> None:
    requests: list[dict[str, object]] = []
    client = _client_returning(
        _VALID_CONTENT, requests, usage={"prompt_tokens": 420, "completion_tokens": 90}
    )

    output = system2.execute_reasoning(_ITEM, client=client)

    assert len(requests) == 1
    request = requests[0]
    assert request["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    body = request["body"]
    assert isinstance(body, dict)
    assert body["model"] == serving.DEFAULT_MODEL
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 256
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    message = body["messages"][0]["content"]
    assert _ITEM in message
    assert "Capítulos disponíveis" in message
    assert "ncm_code" in message

    assert output["reasoning"] == "Cachaça é aguardente de cana; capítulo 22, aguardentes."
    assert output["ncm_code"] == "22084000"
    assert output["fallback_used"] is False
    assert output["prompt_tokens"] == 420
    assert output["completion_tokens"] == 90
    assert output["latency_ms"] >= 0.0


def test_parses_fenced_json_without_fallback() -> None:
    fenced = f"```json\n{_VALID_CONTENT}\n```"
    requests: list[dict[str, object]] = []
    client = _client_returning(
        fenced, requests, usage={"prompt_tokens": 100, "completion_tokens": 40}
    )

    output = system2.execute_reasoning(_ITEM, client=client)

    assert output["ncm_code"] == "22084000"
    assert output["fallback_used"] is False


def test_falls_back_to_regex_when_json_is_malformed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    plain_text = "Depois de analisar, o código correto é 22084000 para esta aguardente."
    requests: list[dict[str, object]] = []
    client = _client_returning(
        plain_text, requests, usage={"prompt_tokens": 100, "completion_tokens": 20}
    )
    caplog.set_level(logging.WARNING)

    output = system2.execute_reasoning(_ITEM, client=client)

    assert output["ncm_code"] == "22084000"
    assert output["fallback_used"] is True
    assert output["reasoning"] == plain_text
    assert "regex fallback" in caplog.text


def test_defaults_to_sentinel_code_on_empty_output(caplog: pytest.LogCaptureFixture) -> None:
    requests: list[dict[str, object]] = []
    client = _client_returning("", requests, usage={"prompt_tokens": 90, "completion_tokens": 0})
    caplog.set_level(logging.WARNING)

    output = system2.execute_reasoning(_ITEM, client=client)

    assert output["ncm_code"] == system2.DEFAULT_INVALID_CODE
    assert output["fallback_used"] is True
    assert output["reasoning"] == ""
    assert "defaulting to" in caplog.text


def test_falls_back_when_json_is_valid_but_code_is_malformed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    malformed = json.dumps({"reasoning": "análise", "ncm_code": "22.08-4000"})
    requests: list[dict[str, object]] = []
    client = _client_returning(
        malformed, requests, usage={"prompt_tokens": 90, "completion_tokens": 30}
    )
    caplog.set_level(logging.WARNING)

    output = system2.execute_reasoning(_ITEM, client=client)

    # O JSON é válido mas o campo não tem 8 dígitos contíguos: regex não acha
    # código → default sentinela com fallback.
    assert output["ncm_code"] == system2.DEFAULT_INVALID_CODE
    assert output["fallback_used"] is True


def test_defaults_tokens_to_zero_when_usage_is_absent() -> None:
    requests: list[dict[str, object]] = []
    client = _client_returning(_VALID_CONTENT, requests, usage=None)

    output = system2.execute_reasoning(_ITEM, client=client)

    assert output["prompt_tokens"] == 0
    assert output["completion_tokens"] == 0
    assert output["fallback_used"] is False
