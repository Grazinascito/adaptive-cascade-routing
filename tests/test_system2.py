"""System2Engine (PR3): CoT+JSON via /v1/chat/completions, sem servidor real.

Respostas vêm de httpx.MockTransport imitando o vLLM. Cobre happy path,
cerca de markdown, prosa ao redor do JSON, validação de categoria contra a
lista (com normalização de caixa/acentos), tokens do usage e latência.
"""

import json

import httpx
import pytest

from src.engine.system2 import SYSTEM2_SYSTEM_PROMPT, System2Engine, System2Output

_BASE_URL = "http://testserver/v1"
_ITEM = "APRESUNTADO REZENDE PEC KG"
_CATEGORIES = ["MERCEARIA", "AÇOUGUE E PEIXARIA", "FRIOS E LATICINIOS"]
_VALID_PAYLOAD: dict[str, str] = {
    "reasoning": "Apresuntado é produto de carne suína em fates; fica no açougue.",
    "category": "AÇOUGUE E PEIXARIA",
    "revisor_insight": "Nomes com 'apresuntado' apontam para AÇOUGUE E PEIXARIA.",
}


def _chat_response(content: str, *, completion_tokens: int = 87) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "Qwen/Qwen3-4B-AWQ",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {"prompt_tokens": 420, "completion_tokens": completion_tokens},
        },
    )


def _engine_returning(
    content: str,
    requests: list[dict[str, object]],
    *,
    completion_tokens: int = 87,
    base_url: str = _BASE_URL,
    max_tokens: int | None = None,
) -> System2Engine:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append({"url": str(request.url), "body": json.loads(request.content)})
        return _chat_response(content, completion_tokens=completion_tokens)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    kwargs: dict[str, object] = {"base_url": base_url, "client": client}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return System2Engine(**kwargs)  # type: ignore[arg-type]


def test_happy_path_json_returns_fields_tokens_and_latency() -> None:
    requests: list[dict[str, object]] = []
    engine = _engine_returning(json.dumps(_VALID_PAYLOAD, ensure_ascii=False), requests)

    output = engine.reason(_ITEM, _CATEGORIES, "MERCEARIA")

    assert isinstance(output, System2Output)
    assert output.reasoning == _VALID_PAYLOAD["reasoning"]
    assert output.predicted_category == "AÇOUGUE E PEIXARIA"
    assert output.revisor_insight == _VALID_PAYLOAD["revisor_insight"]
    assert output.decode_tokens == 87
    assert output.latency_ms >= 0.0


def test_request_carries_system_prompt_user_content_and_qwen3_kwargs() -> None:
    requests: list[dict[str, object]] = []
    engine = _engine_returning(json.dumps(_VALID_PAYLOAD, ensure_ascii=False), requests)

    engine.reason(_ITEM, _CATEGORIES, "MERCEARIA")

    assert len(requests) == 1
    request = requests[0]
    assert request["url"] == f"{_BASE_URL}/chat/completions"
    body = request["body"]
    assert isinstance(body, dict)
    assert body["model"] == "Qwen/Qwen3-4B-AWQ"
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 256
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    messages = body["messages"]
    assert isinstance(messages, list)
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM2_SYSTEM_PROMPT
    user_message = messages[1]
    assert user_message["role"] == "user"
    content = user_message["content"]
    assert isinstance(content, str)
    assert _ITEM in content
    assert all(category in content for category in _CATEGORIES)
    assert "MERCEARIA" in content.split("Categorias permitidas:")[1].split("Sugestão")[0]


def test_suggestion_line_omitted_when_s1_suggested_is_none() -> None:
    requests: list[dict[str, object]] = []
    engine = _engine_returning(json.dumps(_VALID_PAYLOAD, ensure_ascii=False), requests)

    engine.reason(_ITEM, _CATEGORIES)

    body = requests[0]["body"]
    assert isinstance(body, dict)
    content = body["messages"][1]["content"]
    assert isinstance(content, str)
    assert "Sugestão" not in content


def test_max_tokens_parameter_is_honored() -> None:
    requests: list[dict[str, object]] = []
    engine = _engine_returning(
        json.dumps(_VALID_PAYLOAD, ensure_ascii=False), requests, max_tokens=512
    )

    engine.reason(_ITEM, _CATEGORIES)

    body = requests[0]["body"]
    assert isinstance(body, dict)
    assert body["max_tokens"] == 512


def test_fenced_json_is_parsed() -> None:
    fenced = f"```json\n{json.dumps(_VALID_PAYLOAD, ensure_ascii=False)}\n```"
    engine = _engine_returning(fenced, [])

    output = engine.reason(_ITEM, _CATEGORIES)

    assert output.predicted_category == "AÇOUGUE E PEIXARIA"
    assert output.revisor_insight == _VALID_PAYLOAD["revisor_insight"]


def test_json_with_trailing_prose_is_parsed() -> None:
    noisy = (
        f"{json.dumps(_VALID_PAYLOAD, ensure_ascii=False)}"
        " Espero ter ajudado com a classificação!"
    )
    engine = _engine_returning(noisy, [])

    output = engine.reason(_ITEM, _CATEGORIES)

    assert output.predicted_category == "AÇOUGUE E PEIXARIA"
    assert output.reasoning == _VALID_PAYLOAD["reasoning"]


def test_category_outside_allowed_list_is_rejected() -> None:
    payload = dict(_VALID_PAYLOAD, category="ELETRÔNICOS")
    engine = _engine_returning(json.dumps(payload, ensure_ascii=False), [])

    with pytest.raises(ValueError, match="not in the allowed list"):
        engine.reason(_ITEM, _CATEGORIES)


def test_category_match_normalizes_case_and_accents_to_canonical_spelling() -> None:
    payload = dict(_VALID_PAYLOAD, category="açougue e peixaria")
    engine = _engine_returning(json.dumps(payload, ensure_ascii=False), [])

    output = engine.reason(_ITEM, _CATEGORIES)

    assert output.predicted_category == "AÇOUGUE E PEIXARIA"


def test_missing_optional_fields_default_to_empty_strings() -> None:
    payload = {"category": "MERCEARIA"}
    engine = _engine_returning(json.dumps(payload), [])

    output = engine.reason(_ITEM, _CATEGORIES)

    assert output.predicted_category == "MERCEARIA"
    assert output.reasoning == ""
    assert output.revisor_insight == ""


def test_non_json_content_raises() -> None:
    engine = _engine_returning("MERCEARIA parece boa.", [])

    with pytest.raises(ValueError, match="not a parsable JSON object"):
        engine.reason(_ITEM, _CATEGORIES)
