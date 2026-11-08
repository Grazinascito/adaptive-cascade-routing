"""vLLMServerManager — gerencia o ciclo de vida do processo servidor vLLM em background.

Componente B do design doc (docs/adaptive-cascade-design-doc.md §5.2);
stories correspondentes em docs/engine-router-spec.md (Story 1). O módulo
nunca importa vllm: o servidor é lançado como subprocesso (CI/macOS não têm
CUDA — ADR-0001).
"""

import subprocess
import sys
import time
from typing import Final

import httpx

VLLM_LOG_PATH: Final = "vllm_server.log"
HEALTH_POLL_INTERVAL_S: Final = 2.0
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


def _read_vram_usage_gb() -> str | None:
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

    Reusa um servidor já saudável sem lançar novo processo. Se o processo
    morrer durante o startup ou o health nunca passar dentro de
    `startup_timeout_s`, levanta `vLLMServerException` com instruções de
    restart e a leitura de VRAM (edge case de crash/OOM da spec).
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
        return True

    log_file = open(VLLM_LOG_PATH, "ab")
    _process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)

    deadline = time.monotonic() + startup_timeout_s
    while time.monotonic() < deadline:
        if _process.poll() is not None:
            raise vLLMServerException(
                f"vLLM server process exited with code {_process.returncode} during startup "
                f"(possible OOM). See '{VLLM_LOG_PATH}' and GPU VRAM (MiB used,total): "
                f"{_read_vram_usage_gb() or 'unavailable (nvidia-smi not found)'}. "
                f"Restart with: {' '.join(command)}"
            )
        if health_check(health_url):
            return True
        time.sleep(HEALTH_POLL_INTERVAL_S)

    raise vLLMServerException(
        f"vLLM server did not become healthy at {health_url} within {startup_timeout_s}s. "
        f"See '{VLLM_LOG_PATH}' and GPU VRAM (MiB used,total): "
        f"{_read_vram_usage_gb() or 'unavailable (nvidia-smi not found)'}. "
        f"Restart with: {' '.join(command)}"
    )


def stop_server() -> None:
    """Termina o processo que este módulo spawnou; no-op caso contrário."""
    global _process
    if _process is None:
        return
    _process.terminate()
    _process.wait()
    _process = None
