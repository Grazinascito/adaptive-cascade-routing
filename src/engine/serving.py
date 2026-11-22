"""Shared serving constants and HTTP helper used by both engines.

System 1 and System 2 talk to the same local vLLM server; this module holds
what both use (default model, base URL and the latency-measured POST) so
neither engine depends on the other.
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
    """POSTs the JSON body to `url` and returns (response, latency_ms).

    Creates and closes the client when `client` is None; status errors
    propagate via `raise_for_status` (graceful fallback is the harness's
    job, Component G).
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
