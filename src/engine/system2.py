"""System2Engine — slow-path da cascata DARU: CoT + JSON estrito via /v1/chat/completions.

Recebe o texto do item, a lista fechada de categorias do benchmark e a
sugestão opcional do System 1; devolve categoria validada contra a lista
(normalização de caixa/acentos), o raciocínio e o revisor_insight. O parse
tolera cerca de markdown e prosa ao redor do JSON (lição do commit 556000a);
categoria fora da lista levanta `ValueError` — sem fallback silencioso.
"""

import json
import re
import time
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import httpx

DEFAULT_MODEL: Final = "Qwen/Qwen3-4B-AWQ"
DEFAULT_BASE_URL: Final = "http://127.0.0.1:8000/v1"

SYSTEM2_SYSTEM_PROMPT: Final = (
    "Você é um especialista em classificação de produtos do varejo brasileiro.\n"
    "Sua tarefa: classificar a descrição de um produto de varejo brasileiro em "
    "exatamente UMA das categorias fornecidas na mensagem do usuário.\n"
    "Pense passo a passo: a descrição vem abreviada, em estilo de nota fiscal; "
    "elimine as categorias incompatíveis e escolha a única correta.\n"
    "Responda SOMENTE com um objeto JSON estrito — sem blocos de código markdown, "
    "sem texto antes ou depois — no formato:\n"
    "{\n"
    '  "reasoning": "raciocínio passo a passo, em até duas frases",\n'
    '  "category": "a categoria escolhida, copiada exatamente da lista fornecida",\n'
    '  "revisor_insight": "uma frase com a regra geral que explica o acerto"\n'
    "}\n"
    "Não invente categorias fora da lista fornecida."
)

# Cerca de markdown (```json ... ```) observada envolvendo o JSON do
# Qwen3-4B-AWQ no histórico do repo (commit 556000a); removida antes do parse.
_FENCE_PATTERN: Final = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```", re.DOTALL)


@dataclass(frozen=True, slots=True)
class System2Output:
    reasoning: str
    predicted_category: str
    revisor_insight: str
    decode_tokens: int
    latency_ms: float


def _category_key(category: str) -> str:
    decomposed = unicodedata.normalize("NFKD", category)
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(without_accents.casefold().split())


def _match_category(raw_category: object, allowed_categories: Sequence[str]) -> str:
    """Devolve a grafia canônica da categoria; `ValueError` se fora da lista."""
    if not isinstance(raw_category, str):
        raise ValueError(f"system 2 'category' must be a string, got {raw_category!r}")
    raw_key = _category_key(raw_category)
    for allowed in allowed_categories:
        if _category_key(allowed) == raw_key:
            return allowed
    raise ValueError(
        f"system 2 category {raw_category!r} is not in the allowed list "
        f"{list(allowed_categories)}"
    )


def _strip_markdown_fence(content: str) -> str:
    match = _FENCE_PATTERN.match(content)
    return match.group(1) if match else content


def _braces_span(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    return text[start : end + 1]


def _parse_system2_response(
    content: object, allowed_categories: Sequence[str]
) -> tuple[str, str, str]:
    """Extrai (reasoning, category, revisor_insight) do conteúdo do modelo.

    Tenta `json.loads` no texto (com e sem cerca de markdown) e, se a prosa
    cercar o objeto, no trecho entre a primeira `{` e a última `}`.
    """
    text = content if isinstance(content, str) else ""
    candidates = [_strip_markdown_fence(text), _braces_span(text)]
    parsed: object = None
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            break
    if not isinstance(parsed, dict):
        raise ValueError(f"system 2 output is not a parsable JSON object: {text[:200]!r}")

    reasoning = parsed.get("reasoning")
    revisor_insight = parsed.get("revisor_insight")
    category = _match_category(parsed.get("category"), allowed_categories)
    return (
        reasoning if isinstance(reasoning, str) else "",
        category,
        revisor_insight if isinstance(revisor_insight, str) else "",
    )


def _render_user_content(
    item_text: str,
    allowed_categories: Sequence[str],
    s1_suggested: str | None,
) -> str:
    lines = [
        f"Item: {item_text}",
        "",
        "Categorias permitidas:",
        *(f"- {category}" for category in allowed_categories),
    ]
    if s1_suggested is not None:
        lines += [
            "",
            f"Sugestão do System 1 (palpite rápido; confirme ou corrija): {s1_suggested}",
        ]
    return "\n".join(lines)


class System2Engine:
    """Cliente do slow-path: CoT + JSON estrito, temperatura 0, thinking off."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 256,
        temperature: float = 0.0,
        timeout_s: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=timeout_s)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def reason(
        self,
        item_text: str,
        allowed_categories: Sequence[str],
        s1_suggested: str | None = None,
    ) -> System2Output:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM2_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _render_user_content(item_text, allowed_categories, s1_suggested),
                },
            ],
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            # Qwen3 "original" pensa por padrão no template de chat; sem isso o
            # conteúdo vem precedido de blocos <think> e quebra o parse do JSON.
            "chat_template_kwargs": {"enable_thinking": False},
        }

        started = time.perf_counter()
        response = self._client.post(f"{self._base_url}/chat/completions", json=payload)
        latency_ms = (time.perf_counter() - started) * 1000.0
        response.raise_for_status()

        data = response.json()
        choices = data.get("choices") or [{}]
        message = choices[0].get("message") or {}
        reasoning, category, revisor_insight = _parse_system2_response(
            message.get("content"), allowed_categories
        )
        usage = data.get("usage") or {}
        return System2Output(
            reasoning=reasoning,
            predicted_category=category,
            revisor_insight=revisor_insight,
            decode_tokens=int(usage.get("completion_tokens") or 0),
            latency_ms=latency_ms,
        )
