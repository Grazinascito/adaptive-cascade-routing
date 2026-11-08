"""System2Engine — invoca /v1/chat/completions com Chain-of-Thought e saída JSON estrita.

É o fallback da cascata (slow-path).
Componente F do design doc (docs/adaptive-cascade-design-doc.md §5.6);
requisitos completos e stories em docs/system2-evaluator-spec.md (Story 1).
"""

import json
import logging
import re
import time
from typing import Final, TypedDict

import httpx

from src.data.loader import CONTRADICTION_LABEL, ENTAILMENT_LABEL, Label
from src.engine.system1 import DEFAULT_MODEL

logger = logging.getLogger(__name__)

SYSTEM2_PROMPT_TEMPLATE: Final = (
    "Analise o par de frases abaixo e forneça o raciocínio passo a passo "
    "(Chain-of-Thought) antes de determinar o rótulo final.\n\n"
    "Premissa: {premise}\n"
    "Hipótese: {hypothesis}\n\n"
    "Retorne um JSON estrito no seguinte formato:\n"
    "{\n"
    '  "reasoning": "Explicação detalhada...",\n'
    '  "label": "Entailment" | "Contradiction"\n'
    "}"
)

_LABEL_PATTERN: Final = re.compile(r"(?i)\b(entailment|contradiction)\b")

# Mapa próprio para rótulos vindos do modelo (casefold). O `normalize_label`
# do loader mapeia os rótulos crus do ASSIN 2 (ENTAILMENT/NONE), que não é o
# vocabulário de saída do System 2.
_MODEL_LABELS: Final[dict[str, Label]] = {
    "entailment": ENTAILMENT_LABEL,
    "contradiction": CONTRADICTION_LABEL,
}


class System2Output(TypedDict):
    reasoning: str

    label: Label

    prompt_tokens: int

    completion_tokens: int

    latency_ms: float

    fallback_used: bool


def _render_prompt(premise: str, hypothesis: str) -> str:
    return SYSTEM2_PROMPT_TEMPLATE.replace("{premise}", premise).replace("{hypothesis}", hypothesis)


def _model_label(raw: object) -> Label | None:
    if not isinstance(raw, str):
        return None
    return _MODEL_LABELS.get(raw.strip().casefold())


def _parse_system2_response(content: str) -> tuple[str, Label, bool]:
    """Parseia a saída CoT+JSON com fallbacks em cadeia: regex → default Contradiction.

    Retorna (reasoning, label, fallback_used). JSON válido com `reasoning`
    (str) e `label` reconhecido retorna `fallback_used=False`; qualquer falha
    cai no regex sobre o texto bruto (edge case da spec) e, sem match, no
    default `Contradiction` com warning (edge case de output vazio/truncado).
    """
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        reasoning = parsed.get("reasoning")
        label = _model_label(parsed.get("label"))
        if isinstance(reasoning, str) and label is not None:
            return reasoning, label, False

    match = _LABEL_PATTERN.search(content)
    if match:
        label = _MODEL_LABELS[match.group(1).casefold()]
        logger.warning(
            "system 2 output is not valid json; extracted label '%s' via regex fallback",
            label,
        )
        return content, label, True

    logger.warning(
        "system 2 output is empty or has no parsable label; defaulting to '%s'",
        CONTRADICTION_LABEL,
    )
    return content, CONTRADICTION_LABEL, True


def execute_reasoning(
    premise: str,
    hypothesis: str,
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = "http://127.0.0.1:8000/v1",
    max_tokens: int = 256,
    temperature: float = 0.0,
    timeout_s: float = 120.0,
    client: httpx.Client | None = None,
) -> System2Output:
    """Executa o slow-path CoT+JSON (FR-001/FR-002, NFR-001) e retorna System2Output.

    A geração é limitada a `max_tokens` (NFR-001) com `temperature=0.0`
    (reprodutibilidade do benchmark). Saída não conforme cai na cadeia de
    fallbacks da spec com `fallback_used=True` e warning; erros de
    status/timeout/conexão do httpx propagam — o fallback para a previsão do
    System 1 é responsabilidade do harness (Componente G).
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": _render_prompt(premise, hypothesis)}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        # Qwen3 "original" pensa por padrão no template de chat; sem isso o
        # conteúdo vem precedido de blocos <think> e quebra o parse do JSON.
        "chat_template_kwargs": {"enable_thinking": False},
    }

    owns_client = client is None
    active_client = client if client is not None else httpx.Client()
    try:
        started = time.perf_counter()
        response = active_client.post(
            f"{base_url}/chat/completions", json=payload, timeout=timeout_s
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        response.raise_for_status()
    finally:
        if owns_client:
            active_client.close()

    data = response.json()
    choices = data.get("choices") or [{}]
    message = choices[0].get("message") or {}
    content = message.get("content")
    reasoning, label, fallback_used = _parse_system2_response(
        content if isinstance(content, str) else ""
    )

    usage = data.get("usage") or {}
    return System2Output(
        reasoning=reasoning,
        label=label,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        latency_ms=latency_ms,
        fallback_used=fallback_used,
    )
