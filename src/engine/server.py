"""vLLMServerManager — gerencia o ciclo de vida do processo servidor vLLM em background.

Componente B do design doc (docs/adaptive-cascade-design-doc.md §5.2);
stories correspondentes em docs/engine-router-spec.md (Story 1). O módulo
nunca importa vllm: o servidor é lançado como subprocesso (CI/macOS não têm
CUDA — ADR-0001).
"""

import logging
import subprocess
import sys
import time
from typing import Final

import httpx

logger = logging.getLogger(__name__)

VLLM_LOG_PATH: Final = "vllm_server.log"
HEALTH_POLL_INTERVAL_S: Final = 2.0
MODELS_TIMEOUT_S: Final = 5.0
_NVIDIA_SMI_TIMEOUT_S: Final = 5.0

_process: "subprocess.Popen[bytes] | None" = None


class vLLMServerException(Exception):
    """Falha ao subir ou manter o servidor vLLM (crash, OOM ou timeout de startup)."""


def _build_server_command(
    model_name: str,
    *,
    host: str,
    port: int,
    quantization: str,
    enable_prefix_caching: bool,
    gpu_memory_utilization: float,
    logprobs_mode: str,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_name,
        "--host",
        host,
        "--port",
        str(port),
        "--quantization",
        quantization,
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--logprobs-mode",
        logprobs_mode,
    ]
    if enable_prefix_caching:
        command.append("--enable-prefix-caching")
    return command


def _read_vram_usage_raw() -> str | None:
    """Lê o stdout bruto do nvidia-smi (valores em MiB); None se indisponível."""
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


def health_check(base_url: str = "http://127.0.0.1:8000", *, timeout_s: float = 5.0) -> bool:
    """Retorna True se o servidor responde 200 em /health; qualquer erro vira False."""
    try:
        response = httpx.get(f"{base_url}/health", timeout=timeout_s)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


def _server_has_model(
    base_url: str, model_name: str, *, timeout_s: float = MODELS_TIMEOUT_S
) -> bool | None:
    """Confere via /v1/models se o servidor serve `model_name`.

    Devolve True/False conforme a lista de modelos carregados; None quando
    a lista não pôde ser obtida (erro de conexão ou status != 200) — caso
    em que o reuso segue com warning em vez de falhar.
    """
    try:
        response = httpx.get(f"{base_url}/v1/models", timeout=timeout_s)
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    loaded = [entry.get("id") for entry in response.json().get("data", [])]
    return model_name in loaded


def _startup_failure_context(command: list[str]) -> str:
    """Cauda comum das mensagens de falha de startup: log, VRAM e restart."""
    vram = _read_vram_usage_raw()
    return (
        f"See '{VLLM_LOG_PATH}' and GPU VRAM (MiB used,total): "
        f"{vram or 'unavailable (nvidia-smi not found)'}. "
        f"Restart with: {' '.join(command)}"
    )


def start_server(
    model_name: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    quantization: str = "awq",
    enable_prefix_caching: bool = True,
    gpu_memory_utilization: float = 0.90,
    logprobs_mode: str = "raw_logprobs",
    startup_timeout_s: float = 90.0,
) -> bool:
    """Sobe o servidor vLLM e espera /health ficar pronto (FR-001).

    Reusa um servidor já saudável — conferindo via /v1/models que ele serve
    `model_name`; sem a lista, reusa com warning. Se o processo morrer
    durante o startup ou o health nunca passar dentro de
    `startup_timeout_s`, registra o evento de falha no log e levanta
    `vLLMServerException` com instruções de restart e a leitura de VRAM
    (edge case de crash/OOM da spec).
    """
    global _process
    command = _build_server_command(
        model_name,
        host=host,
        port=port,
        quantization=quantization,
        enable_prefix_caching=enable_prefix_caching,
        gpu_memory_utilization=gpu_memory_utilization,
        logprobs_mode=logprobs_mode,
    )
    health_url = f"http://{host}:{port}"
    if health_check(health_url):
        has_model = _server_has_model(health_url, model_name)
        if has_model is False:
            raise vLLMServerException(
                f"A healthy vLLM server is already running at {health_url} but does not "
                f"serve model '{model_name}'. Stop it with stop_server() or pick another port."
            )
        if has_model is None:
            logger.warning(
                "could not list models of the healthy server at %s; reusing it without "
                "verifying it serves '%s'",
                health_url,
                model_name,
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
        if health_check(health_url):
            return True
        time.sleep(HEALTH_POLL_INTERVAL_S)

    message = (
        f"vLLM server did not become healthy at {health_url} within {startup_timeout_s}s. "
        f"{_startup_failure_context(command)}"
    )
    logger.error("vLLM startup failed: %s", message)
    raise vLLMServerException(message)


def stop_server() -> None:
    """Termina o processo que este módulo spawnou; no-op caso contrário."""
    global _process
    if _process is None:
        return
    _process.terminate()
    _process.wait()
    _process = None
