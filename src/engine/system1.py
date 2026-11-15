"""System1Engine — classificação NCM em geração curta com confiança por logprobs.

Pivot NCM (2026-11-14):
com 114 códigos candidatos, a pergunta `choice` binária do `/v1/systemone`
não se aplica; o fast-path passa a ser uma chamada `/v1/chat/completions`
com `max_tokens` curto (12), sem raciocínio, produzindo o código de 8
dígitos. A confiança é a média geométrica das probabilidades dos tokens
gerados — exp(Σ ln p / n), menos punitiva que o produto puro em códigos de
8 dígitos — e código inexistente na taxonomia vira confiança 0.0 (escala
imediatamente, espelhando o "caso novo → fallback" do blog do iFood). A
implementação anterior (`/v1/systemone`) permanece no histórico do git.
"""

import math
import re
from collections.abc import Mapping, Sequence
from typing import Final, TypedDict

import httpx

from src.data.loader import chapter_lines, is_valid_code
from src.engine.router import shannon_entropy
from src.engine.serving import DEFAULT_BASE_URL, DEFAULT_MODEL, post_json

NCM_CODE_PATTERN: Final = re.compile(r"\b(\d{8})\b")

DEFAULT_MAX_TOKENS: Final = 12
DEFAULT_TOP_LOGPROBS: Final = 20

INVALID_CODE_SENTINEL: Final = "00000000"

SYSTEM1_INSTRUCTIONS: Final = (
    "Você é um classificador de itens de catálogo brasileiro.\n"
    "Receba o nome de um item e responda com o código NCM de 8 dígitos "
    "correspondente, de acordo com a Nomenclatura Comum do Mercosul.\n"
    "Capítulos disponíveis neste catálogo:\n"
    + "\n".join(chapter_lines())
    + "\nResponda apenas com o código de 8 dígitos, sem texto adicional."
)


class System1Output(TypedDict):
    """Saída do fast-path NCM.

    `code` é `INVALID_CODE_SENTINEL` quando a resposta não contém código de
    8 dígitos; nesse caso `valid_code=False` e `confidence=0.0` (escala ao
    System 2). `entropy` é a entropia de Shannon sobre a distribuição
    top-logprobs do primeiro token gerado — o ponto onde o modelo "decide"
    o início do código. `prompt_tokens`/`completion_tokens` vêm do `usage`
    do chat completions (o fast-path também custa tokens; a economia do
    KPI de custo é medida contra o slow-path, não zerada).
    """

    code: str

    confidence: float

    entropy: float

    valid_code: bool

    prompt_tokens: int

    completion_tokens: int

    latency_ms: float


def _confidence_from_logprobs(logprobs: Sequence[float]) -> float:
    """Média geométrica das probabilidades: exp(Σ ln p / n)."""
    if not logprobs:
        return 0.0
    return math.exp(sum(logprobs) / len(logprobs))


def _entropy_from_top_logprobs(
    top_entries: Sequence[Mapping[str, float]],
) -> float:
    """Entropia de Shannon (bits) sobre a distribuição top-k normalizada."""
    weights = [math.exp(entry["logprob"]) for entry in top_entries]
    total = sum(weights)
    if total <= 0.0:
        return 0.0
    # Chaves indexadas: pesos iguais são entradas distintas da distribuição
    # (uma chave única colapsaria a soma e reduziria a entropia).
    return shannon_entropy({str(index): weight / total for index, weight in enumerate(weights)})


def evaluate_short_classification(
    item_name: str,
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    instructions: str = SYSTEM1_INSTRUCTIONS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    top_logprobs: int = DEFAULT_TOP_LOGPROBS,
    temperature: float = 0.0,
    timeout_s: float = 10.0,
    client: httpx.Client | None = None,
) -> System1Output:
    """Executa o fast-path NCM e retorna código + confiança.

    Erros de status/timeout/conexão do httpx propagam — o fallback gracioso
    é responsabilidade do harness (Componente G).
    """
    if not item_name.strip():
        raise ValueError("item_name must be a non-blank string")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": item_name},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "logprobs": True,
        "top_logprobs": top_logprobs,
        # Qwen3 "original" pensa por padrão no template de chat; a geração
        # curta do fast-path não pode abrir bloco <think>.
        "chat_template_kwargs": {"enable_thinking": False},
    }

    response, latency_ms = post_json(client, f"{base_url}/chat/completions", payload, timeout_s)

    data = response.json()
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    raw_content = message.get("content")
    content = raw_content if isinstance(raw_content, str) else ""

    match = NCM_CODE_PATTERN.search(content)
    code = match.group(1) if match else INVALID_CODE_SENTINEL
    valid = match is not None and is_valid_code(code)

    token_logprobs: list[float] = []
    first_token_top: list[dict[str, float]] = []
    logprobs_payload = choice.get("logprobs") or {}
    content_logprobs = logprobs_payload.get("content") or []
    for position, entry in enumerate(content_logprobs):
        if isinstance(entry.get("logprob"), (int, float)):
            token_logprobs.append(float(entry["logprob"]))
        if position == 0:
            first_token_top = list(entry.get("top_logprobs") or [])

    confidence = _confidence_from_logprobs(token_logprobs) if valid else 0.0
    entropy = _entropy_from_top_logprobs(first_token_top)

    usage = data.get("usage") or {}
    return System1Output(
        code=code,
        confidence=confidence,
        entropy=entropy,
        valid_code=valid,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        latency_ms=latency_ms,
    )
