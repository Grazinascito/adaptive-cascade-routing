"""Testes do Componente C pós-pivot: classificação NCM em geração curta.

O vLLM real nunca é chamado: as respostas vêm de httpx.MockTransport no
limite da rede, imitando o contrato do /v1/chat/completions com logprobs
por token. Corresponde à Story 4 de docs/dataset-pivot-ncm-spec.md
(Decisão 1(d): confiança = média geométrica dos logprobs dos tokens).
"""

import json
import math

import httpx
import pytest

from src.data.loader import load_taxonomy
from src.engine import system1

_ITEM = "ALCOOL LIQUIDO START NAT 70% - 5VL"

# Código válido na taxonomia commitada (config/ncm_taxonomy.csv).
_VALID_CODE = "22071090"


def _token(token: str, logprob: float, top: list[dict[str, object]] | None = None) -> dict:
    entry: dict[str, object] = {"token": token, "logprob": logprob}
    if top is not None:
        entry["top_logprobs"] = top
    return entry


def _chat_response(
    content: str,
    content_logprobs: list[dict],
    *,
    usage: dict[str, int] | None = None,
) -> httpx.Response:
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
                "logprobs": {"content": content_logprobs},
            }
        ],
    }
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(200, json=body)


def _client_returning(
    content: str,
    content_logprobs: list[dict],
    requests: list[dict[str, object]],
    *,
    usage: dict[str, int] | None = None,
) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append({"url": str(request.url), "body": json.loads(request.content)})
        return _chat_response(content, content_logprobs, usage=usage)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _valid_code_logprobs() -> list[dict]:
    # 8 tokens de dígito com ln(p) = -0.01 cada → média geométrica = e^-0.01.
    return [_token(str(digit), math.log(0.99)) for digit in _VALID_CODE]


def test_returns_code_and_confidence_from_token_logprobs() -> None:
    requests: list[dict[str, object]] = []
    client = _client_returning(
        _VALID_CODE,
        _valid_code_logprobs(),
        requests,
        usage={"prompt_tokens": 210, "completion_tokens": 8},
    )

    output = system1.evaluate_short_classification(_ITEM, client=client)

    assert len(requests) == 1
    body = requests[0]["body"]
    assert isinstance(body, dict)
    assert requests[0]["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert body["max_tokens"] == system1.DEFAULT_MAX_TOKENS
    assert body["temperature"] == 0.0
    assert body["logprobs"] is True
    assert body["top_logprobs"] == system1.DEFAULT_TOP_LOGPROBS
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1] == {"role": "user", "content": _ITEM}

    assert output["code"] == _VALID_CODE
    assert output["confidence"] == pytest.approx(0.99, abs=1e-9)
    assert output["valid_code"] is True
    assert output["prompt_tokens"] == 210
    assert output["completion_tokens"] == 8
    assert output["latency_ms"] >= 0.0


def test_confidence_is_geometric_mean_not_product_of_token_probabilities() -> None:
    # Lista de logprobs com 2 entradas de ln(0.5): média geométrica 0.5,
    # produto 0.25 — prova que a fórmula é a média, não o produto.
    logprobs = [_token("2207", math.log(0.5)), _token("1090", math.log(0.5))]
    requests: list[dict[str, object]] = []
    client = _client_returning(_VALID_CODE, logprobs, requests)

    output = system1.evaluate_short_classification(_ITEM, client=client)

    assert output["confidence"] == pytest.approx(0.5, abs=1e-9)


def test_confidence_is_zero_when_code_is_not_in_taxonomy() -> None:
    # 99999999 é bem-formado (8 dígitos) mas não existe na taxonomia.
    logprobs = [_token(str(digit), -0.01) for digit in "99999999"]
    requests: list[dict[str, object]] = []
    client = _client_returning("99999999", logprobs, requests)

    output = system1.evaluate_short_classification(_ITEM, client=client)

    assert output["code"] == "99999999"
    assert output["valid_code"] is False
    assert output["confidence"] == 0.0


def test_flags_fallback_when_output_has_no_valid_code() -> None:
    requests: list[dict[str, object]] = []
    client = _client_returning("Não sei classificar", [_token("Não", -0.5)], requests)

    output = system1.evaluate_short_classification(_ITEM, client=client)

    assert output["code"] == system1.INVALID_CODE_SENTINEL
    assert output["valid_code"] is False
    assert output["confidence"] == 0.0


def test_entropy_comes_from_first_token_top_logprobs_distribution() -> None:
    # Distribuição 50/50 no primeiro token → entropia máxima de 1 bit.
    top = [
        {"token": "2", "logprob": math.log(0.5)},
        {"token": "3", "logprob": math.log(0.5)},
    ]
    logprobs = [_token("2", math.log(0.5), top)] + _valid_code_logprobs()[1:]
    requests: list[dict[str, object]] = []
    client = _client_returning(_VALID_CODE, logprobs, requests)

    output = system1.evaluate_short_classification(_ITEM, client=client)

    assert output["entropy"] == pytest.approx(1.0, abs=1e-9)


def test_rejects_blank_item_name_before_any_http_call() -> None:
    requests: list[dict[str, object]] = []
    client = _client_returning(_VALID_CODE, _valid_code_logprobs(), requests)

    with pytest.raises(ValueError, match="item_name"):
        system1.evaluate_short_classification("   ", client=client)

    assert requests == []


def test_system1_instructions_embed_the_chapter_list_from_the_taxonomy() -> None:
    taxonomy = load_taxonomy()
    chapter_titles = {entry["chapter_title"] for entry in taxonomy.values()}
    assert any(title in system1.SYSTEM1_INSTRUCTIONS for title in chapter_titles)
