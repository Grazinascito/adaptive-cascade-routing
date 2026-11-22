"""System2Engine — DARU cascade slow path: CoT + strict JSON via /v1/chat/completions.

Takes the item text, the benchmark's closed category list and the optional
System 1 suggestion; returns the category validated against the list
(case/accent normalization), the reasoning and the revisor_insight. Parsing
tolerates markdown fences and prose around the JSON (lesson from commit
556000a); a category outside the list raises `ValueError` — no silent
fallback.
"""

import difflib
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

# Markdown fence (```json ... ```) seen wrapping the JSON of Qwen3-4B-AWQ in
# the repo history (commit 556000a); stripped before parsing.
_FENCE_PATTERN: Final = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```", re.DOTALL)

# Close-match cutoff for model misspellings of a category name
# (real case from the first Colab run: "SUCCOS E REFRESCOS").
_FUZZY_CUTOFF: Final = 0.85


@dataclass(frozen=True, slots=True)
class System2Output:
    reasoning: str
    predicted_category: str
    revisor_insight: str
    decode_tokens: int
    latency_ms: float
    category_fuzzy_matched: bool = False
    parse_ok: bool = True


def _category_key(category: str) -> str:
    decomposed = unicodedata.normalize("NFKD", category)
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(without_accents.casefold().split())


def _match_category(raw_category: object, allowed_categories: Sequence[str]) -> tuple[str, bool]:
    """Exact accent/case-insensitive match, then a deterministic close match.

    Qwen3-4B occasionally misspells a category ("SUCCOS E REFRESCOS" for
    "SUCOS E REFRESCOS"); difflib at 0.85 turns that into a counted
    prediction instead of a crash. Raises ValueError when nothing matches.
    """
    if not isinstance(raw_category, str):
        raise ValueError(f"system 2 'category' must be a string, got {raw_category!r}")
    raw_key = _category_key(raw_category)
    allowed_keys = [_category_key(allowed) for allowed in allowed_categories]
    for allowed, key in zip(allowed_categories, allowed_keys, strict=True):
        if key == raw_key:
            return allowed, False
    close = difflib.get_close_matches(raw_key, allowed_keys, n=1, cutoff=_FUZZY_CUTOFF)
    if close:
        return allowed_categories[allowed_keys.index(close[0])], True
    raise ValueError(
        f"system 2 category {raw_category!r} is not in the allowed list {list(allowed_categories)}"
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
) -> tuple[str, str, str, bool]:
    """Extracts (reasoning, category, revisor_insight, fuzzy) from content.

    Tries `json.loads` on the text (with and without the markdown fence)
    and, when prose surrounds the object, on the span from the first `{`
    to the last `}`.
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
    category, fuzzy = _match_category(parsed.get("category"), allowed_categories)
    return (
        reasoning if isinstance(reasoning, str) else "",
        category,
        revisor_insight if isinstance(revisor_insight, str) else "",
        fuzzy,
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
    """Slow-path client: CoT + strict JSON, temperature 0, thinking off."""

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
            # Qwen3 "original" thinks by default in its chat template; without
            # this the content comes prefixed with <think> blocks and breaks
            # the JSON parse.
            "chat_template_kwargs": {"enable_thinking": False},
        }

        started = time.perf_counter()
        response = self._client.post(f"{self._base_url}/chat/completions", json=payload)
        latency_ms = (time.perf_counter() - started) * 1000.0
        response.raise_for_status()

        data = response.json()
        choices = data.get("choices") or [{}]
        message = choices[0].get("message") or {}
        usage = data.get("usage") or {}
        decode_tokens = int(usage.get("completion_tokens") or 0)
        try:
            reasoning, category, revisor_insight, fuzzy = _parse_system2_response(
                message.get("content"), allowed_categories
            )
        except ValueError:
            # Non-JSON text or a category outside the list: a measured failure
            # of one item, not a reason to kill the whole GPU run.
            return System2Output(
                reasoning="",
                predicted_category="",
                revisor_insight="",
                decode_tokens=decode_tokens,
                latency_ms=latency_ms,
                parse_ok=False,
            )
        return System2Output(
            reasoning=reasoning,
            predicted_category=category,
            revisor_insight=revisor_insight,
            decode_tokens=decode_tokens,
            latency_ms=latency_ms,
            category_fuzzy_matched=fuzzy,
        )
