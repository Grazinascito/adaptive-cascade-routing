"""Constantes de serving e helper de HTTP compartilhados pelos engines.

System 1 e System 2 falam com o mesmo servidor vLLM local; este módulo
reúne o que os dois usam (modelo padrão, base URL e o POST com medição de
latência) para que nenhum engine dependa do outro.
"""

import time
from typing import Any, Final

import httpx

DEFAULT_MODEL: Final = "Qwen/Qwen3-4B-AWQ"
DEFAULT_BASE_URL: Final = "http://127.0.0.1:8000/v1"


def post_json(
    client: httpx.Client | None,
    url: str,
    body: dict[str, Any],
    timeout_s: float,
) -> tuple[httpx.Response, float]:
    """Executa o POST JSON em `url` e devolve (response, latency_ms).

    Cria e fecha o client quando `client` é None; erros de status propagam
    via `raise_for_status` (o fallback gracioso é papel do harness,
    Componente G).
    """
    owns_client = client is None
    active_client = client if client is not None else httpx.Client()
    try:
        started = time.perf_counter()
        response = active_client.post(url, json=body, timeout=timeout_s)
        latency_ms = (time.perf_counter() - started) * 1000.0
        response.raise_for_status()
    finally:
        if owns_client:
            active_client.close()
    return response, latency_ms
