"""vLLMServerManager — vLLM server lifecycle for the DARU benchmark.

Starts `Qwen/Qwen3-4B-AWQ` (4-bit AWQ) as a subprocess with prefix caching
and 90% of VRAM, ready for the /v1/systemone and /v1/chat/completions
endpoints. The module never imports vllm: CI/macOS have no CUDA — the
server runs as a subprocess (ADR-0001) and the health check is
GET /v1/models until it answers 200.
"""

import logging
import subprocess
import sys
import time
from collections.abc import Sequence
from typing import Final

import httpx

logger = logging.getLogger(__name__)

VLLM_LOG_PATH: Final = "vllm_server.log"
HEALTH_POLL_INTERVAL_S: Final = 2.0
MODELS_TIMEOUT_S: Final = 5.0
DEFAULT_GPU_MEMORY_UTILIZATION: Final = 0.90
DEFAULT_STARTUP_TIMEOUT_S: Final = 900.0
_NVIDIA_SMI_TIMEOUT_S: Final = 5.0

_process: "subprocess.Popen[bytes] | None" = None


class vLLMServerException(Exception):
    """Failed to start or keep the vLLM server up (crash, OOM or startup timeout)."""


def build_server_command(
    model: str,
    port: int,
    extra_flags: Sequence[str] = (),
    *,
    host: str = "127.0.0.1",
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    enable_prefix_caching: bool = True,
) -> list[str]:
    """vLLM launch argv: model, port, 90% of VRAM and prefix caching."""
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]
    if enable_prefix_caching:
        command.append("--enable-prefix-caching")
    command.extend(extra_flags)
    return command


def _read_vram_usage_raw() -> str | None:
    """Reads raw nvidia-smi stdout (values in MiB); None if unavailable."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_S,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def health_check(
    base_url: str = "http://127.0.0.1:8000", *, timeout_s: float = MODELS_TIMEOUT_S
) -> bool:
    """True when GET /v1/models answers 200; any error becomes False."""
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/v1/models", timeout=timeout_s)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


def _server_has_model(
    base_url: str, model_name: str, *, timeout_s: float = MODELS_TIMEOUT_S
) -> bool | None:
    """Checks via /v1/models whether the server serves `model_name`.

    Returns True/False from the loaded model list; None when the list could
    not be fetched (connection error or status != 200) — in that case reuse
    continues with a warning instead of failing.
    """
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/v1/models", timeout=timeout_s)
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    loaded = [entry.get("id") for entry in response.json().get("data", [])]
    return model_name in loaded


def _startup_failure_context(command: list[str]) -> str:
    """Common tail for startup failure messages: log, VRAM and restart."""
    vram = _read_vram_usage_raw()
    return (
        f"See '{VLLM_LOG_PATH}' and GPU VRAM (MiB used,total): "
        f"{vram or 'unavailable (nvidia-smi not found)'}. "
        f"Restart with: {' '.join(command)}"
    )


def start_server(
    model: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    extra_flags: Sequence[str] = (),
    enable_prefix_caching: bool = True,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
) -> bool:
    """Starts the vLLM server and waits for /v1/models to answer 200.

    Reuses an already healthy server — checking via /v1/models that it
    serves `model`; without the list, reuses with a warning. If the process
    dies during startup or health never passes within `startup_timeout_s`,
    logs the event and raises `vLLMServerException` with restart
    instructions and the VRAM reading (the crash/OOM edge case from the
    spec).
    """
    global _process
    command = build_server_command(
        model,
        port,
        extra_flags,
        host=host,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=enable_prefix_caching,
    )
    base_url = f"http://{host}:{port}"
    if health_check(base_url):
        has_model = _server_has_model(base_url, model)
        if has_model is False:
            raise vLLMServerException(
                f"A healthy vLLM server is already running at {base_url} but does not "
                f"serve model '{model}'. Stop it with stop_server() or pick another port."
            )
        if has_model is None:
            logger.warning(
                "could not list models of the healthy server at %s; reusing it without "
                "verifying it serves '%s'",
                base_url,
                model,
            )
        return True

    log_file = open(VLLM_LOG_PATH, "ab")
    _process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)

    deadline = time.monotonic() + startup_timeout_s
    while time.monotonic() < deadline:
        if _process.poll() is not None:
            message = (
                f"vLLM server process exited with code {_process.returncode} during startup "
                f"(possible OOM). {_startup_failure_context(command)}"
            )
            logger.error("vLLM startup failed: %s", message)
            raise vLLMServerException(message)
        if health_check(base_url):
            return True
        time.sleep(HEALTH_POLL_INTERVAL_S)

    message = (
        f"vLLM server did not become healthy at {base_url} within {startup_timeout_s}s. "
        f"{_startup_failure_context(command)}"
    )
    logger.error("vLLM startup failed: %s", message)
    raise vLLMServerException(message)


def stop_server() -> None:
    """Terminates the process this module spawned; no-op otherwise."""
    global _process
    if _process is None:
        return
    _process.terminate()
    _process.wait()
    _process = None
