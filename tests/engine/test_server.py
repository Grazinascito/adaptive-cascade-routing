"""Testes do Componente B: ciclo de vida do servidor vLLM (start/health/stop).

O processo real nunca é lançado: `subprocess.Popen` é substituído por um spy
e o health-check por respostas no limite da rede (httpx.MockTransport).
Corresponde à Story 1 de docs/engine-router-spec.md.
"""

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.engine import server


class _FakePopen:
    """Popen que registra o comando recebido e simula um processo vivo."""

    def __init__(self, command: list[str], **kwargs: Any) -> None:
        self.command = command
        self.returncode: int | None = None
        self.terminate_calls = 0
        self.wait_calls = 0

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        return 0


@pytest.fixture(autouse=True)
def _isolated_server_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(server, "_process", None)
    monkeypatch.setattr(server, "VLLM_LOG_PATH", str(tmp_path / "vllm_server.log"))
    monkeypatch.setattr(server, "HEALTH_POLL_INTERVAL_S", 0.0)


def _spawn_recording(monkeypatch: pytest.MonkeyPatch, spawned: list[_FakePopen]) -> None:
    def fake_popen(command: list[str], **kwargs: Any) -> _FakePopen:
        process = _FakePopen(command)
        spawned.append(process)
        return process

    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)


def _without_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_not_found(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(server.subprocess, "run", raise_not_found)


def _get_through_transport(
    handler: Callable[[httpx.Request], httpx.Response],
) -> Callable[..., httpx.Response]:
    def _get(url: str, timeout: float | None = None) -> httpx.Response:
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            return client.get(url, timeout=timeout)

    return _get


def test_start_server_returns_true_without_spawning_when_health_check_already_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned: list[_FakePopen] = []
    _spawn_recording(monkeypatch, spawned)
    monkeypatch.setattr(server, "health_check", lambda base_url: True)

    assert server.start_server("Qwen/Qwen2.5-7B-Instruct-AWQ") is True
    assert spawned == []


def test_start_server_launches_api_server_with_expected_flags_and_polls_until_healthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned: list[_FakePopen] = []
    _spawn_recording(monkeypatch, spawned)
    health_calls = {"count": 0}

    def healthy_on_second_poll(base_url: str) -> bool:
        health_calls["count"] += 1
        return health_calls["count"] >= 2

    monkeypatch.setattr(server, "health_check", healthy_on_second_poll)

    assert server.start_server("Qwen/Qwen2.5-7B-Instruct-AWQ") is True

    assert health_calls["count"] == 2
    assert spawned[0].command == [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--quantization",
        "awq",
        "--gpu-memory-utilization",
        "0.9",
        "--logprobs-mode",
        "raw_logprobs",
        "--enable-prefix-caching",
    ]


def test_start_server_raises_vllmserverexception_with_restart_instructions_when_never_healthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned: list[_FakePopen] = []
    _spawn_recording(monkeypatch, spawned)
    _without_nvidia_smi(monkeypatch)
    monkeypatch.setattr(server, "health_check", lambda base_url: False)

    with pytest.raises(server.vLLMServerException, match="vllm.entrypoints.openai.api_server"):
        server.start_server("Qwen/Qwen2.5-7B-Instruct-AWQ", startup_timeout_s=0.0)


def test_start_server_raises_vllmserverexception_when_process_exits_during_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dead = _FakePopen([])
    dead.returncode = 1
    monkeypatch.setattr(server.subprocess, "Popen", lambda command, **kwargs: dead)
    _without_nvidia_smi(monkeypatch)
    monkeypatch.setattr(server, "health_check", lambda base_url: False)

    with pytest.raises(server.vLLMServerException, match="vllm_server.log"):
        server.start_server("Qwen/Qwen2.5-7B-Instruct-AWQ")


def test_health_check_returns_true_on_200_and_false_on_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def ok(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    monkeypatch.setattr(server.httpx, "get", _get_through_transport(ok))
    assert server.health_check() is True

    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(server.httpx, "get", _get_through_transport(refused))
    assert server.health_check() is False


def test_stop_server_terminates_only_a_process_we_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned: list[_FakePopen] = []
    _spawn_recording(monkeypatch, spawned)
    health_calls = {"count": 0}

    def healthy_on_second_poll(base_url: str) -> bool:
        health_calls["count"] += 1
        return health_calls["count"] >= 2

    monkeypatch.setattr(server, "health_check", healthy_on_second_poll)

    assert server.start_server("Qwen/Qwen2.5-7B-Instruct-AWQ") is True

    server.stop_server()
    server.stop_server()

    assert spawned[0].terminate_calls == 1
    assert spawned[0].wait_calls == 1


def test_read_vram_usage_returns_none_when_nvidia_smi_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_nvidia_smi(monkeypatch)

    assert server._read_vram_usage_gb() is None
