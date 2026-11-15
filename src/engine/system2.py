"""System2Engine — invoca /v1/chat/completions com Chain-of-Thought e saída JSON estrita.

É o fallback da cascata (slow-path). Após o pivot NCM (2026-11-14), a tarefa é
classificar o nome de um item de catálogo em código NCM de 8 dígitos; o prompt
carrega a lista compacta de capítulos como contexto da taxonomia (o "mapa da
árvore" do blog do iFood). Componente F.
"""

import json
import logging
import re
from typing import Final, TypedDict

import httpx

from src.data.loader import chapter_lines
from src.engine.serving import DEFAULT_BASE_URL, DEFAULT_MODEL, post_json

logger = logging.getLogger(__name__)

SYSTEM2_PROMPT_TEMPLATE: Final = (
    "Você é um especialista em classificação fiscal de mercadorias (NCM — "
    "Nomenclatura Comum do Mercosul).\n"
    "Analise o nome do item de catálogo abaixo e forneça o raciocínio passo a "
    "passo (Chain-of-Thought) antes de determinar o código NCM de 8 dígitos.\n\n"
    "Capítulos disponíveis neste catálogo:\n"
    + "\n".join(chapter_lines())
    + "\n\nItem: {item_name}\n\n"
    "Retorne um JSON estrito no seguinte formato:\n"
    "{\n"
    '  "reasoning": "Explicação detalhada...",\n'
    '  "ncm_code": "12345678"\n'
    "}\n\n"
    "Responda apenas com o JSON, sem blocos de código markdown, e mantenha "
    "o raciocínio em até duas frases."
)

_NCM_CODE_PATTERN: Final = re.compile(r"\b(\d{8})\b")

# Cerca de markdown (```json ... ```) observada envolvendo o JSON em 12 das
# 20 respostas do Qwen3-4B-AWQ no Colab (temperatura 0) na era ASSIN; o
# json.loads rejeita o texto com cerca, então ela é removida antes do parse.
_FENCE_PATTERN: Final = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)

DEFAULT_INVALID_CODE: Final = "00000000"


class System2Output(TypedDict):
    """Saída do slow-path NCM.

    `ncm_code` é `DEFAULT_INVALID_CODE` quando nem o JSON nem o regex
    encontram um código de 8 dígitos (contado como erro na avaliação, nunca
    derruba o run). `reasoning` carrega o CoT do JSON quando
    `fallback_used=False` e o conteúdo bruto do modelo quando `True`.
    """

    reasoning: str

    ncm_code: str

    prompt_tokens: int

    completion_tokens: int

    latency_ms: float

    fallback_used: bool


def _render_prompt(item_name: str) -> str:
    return SYSTEM2_PROMPT_TEMPLATE.replace("{item_name}", item_name)


def _strip_markdown_fence(content: str) -> str:
    """Remove a cerca de markdown; sem cerca, devolve o texto inalterado."""
    match = _FENCE_PATTERN.match(content)
    return match.group(1) if match else content


def _parse_system2_response(content: str) -> tuple[str, str, bool]:
    """Parseia a saída CoT+JSON com fallbacks em cadeia: JSON → regex → default.

    Retorna (reasoning, ncm_code, fallback_used). JSON válido — com ou sem
    cerca de markdown, que é formatação e não conteúdo — com `reasoning`
    (str) e `ncm_code` de 8 dígitos retorna `fallback_used=False`; qualquer
    falha cai no regex sobre o texto bruto e, sem match, no default
    `DEFAULT_INVALID_CODE` com warning (edge case de output vazio/truncado).
    A validade do código na taxonomia é papel da avaliação, não do parser.
    """
    try:
        parsed = json.loads(_strip_markdown_fence(content))
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        reasoning = parsed.get("reasoning")
        raw_code = parsed.get("ncm_code")
        if isinstance(reasoning, str) and isinstance(raw_code, str):
            match = _NCM_CODE_PATTERN.search(raw_code)
            if match:
                return reasoning, match.group(1), False

    match = _NCM_CODE_PATTERN.search(content)
    if match:
        logger.warning(
            "system 2 output is not valid json; extracted code '%s' via regex fallback",
            match.group(1),
        )
        return content, match.group(1), True

    logger.warning(
        "system 2 output is empty or has no parsable code; defaulting to '%s'",
        DEFAULT_INVALID_CODE,
    )
    return content, DEFAULT_INVALID_CODE, True


def execute_reasoning(
    item_name: str,
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    max_tokens: int = 256,
    temperature: float = 0.0,
    timeout_s: float = 120.0,
    client: httpx.Client | None = None,
) -> System2Output:
    """Executa o slow-path CoT+JSON (FR-001/FR-002 da spec F–G) e retorna System2Output.

    A geração é limitada a `max_tokens` com `temperature=0.0`
    (reprodutibilidade do benchmark). Saída não conforme cai na cadeia de
    fallbacks com `fallback_used=True` e warning; erros de status/timeout/
    conexão do httpx propagam — o fallback para a previsão do System 1 é
    responsabilidade do harness (Componente G).
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": _render_prompt(item_name)}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        # Qwen3 "original" pensa por padrão no template de chat; sem isso o
        # conteúdo vem precedido de blocos <think> e quebra o parse do JSON.
        "chat_template_kwargs": {"enable_thinking": False},
    }

    response, latency_ms = post_json(client, f"{base_url}/chat/completions", payload, timeout_s)

    data = response.json()
    choices = data.get("choices") or [{}]
    message = choices[0].get("message") or {}
    content = message.get("content")
    reasoning, ncm_code, fallback_used = _parse_system2_response(
        content if isinstance(content, str) else ""
    )

    usage = data.get("usage") or {}
    return System2Output(
        reasoning=reasoning,
        ncm_code=ncm_code,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        latency_ms=latency_ms,
        fallback_used=fallback_used,
    )
