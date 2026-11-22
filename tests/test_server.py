"""Tests for the DARU vLLM server argv (PR5): command building only.

No process is spawned and no port is opened — `build_server_command` is
pure. The lifecycle (spawn/health/stop) runs only in Colab.
"""

import sys

from src.engine import server


def test_build_server_command_carries_model_port_and_prefix_caching() -> None:
    command = server.build_server_command("Qwen/Qwen3-4B-AWQ", 8000)

    assert command[:3] == [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
    assert command[command.index("--model") + 1] == "Qwen/Qwen3-4B-AWQ"
    assert command[command.index("--port") + 1] == "8000"
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.9"
    assert "--enable-prefix-caching" in command


def test_build_server_command_formats_arbitrary_port_and_host() -> None:
    command = server.build_server_command("Qwen/Qwen3-4B-AWQ", 8001, host="0.0.0.0")

    assert command[command.index("--port") + 1] == "8001"
    assert command[command.index("--host") + 1] == "0.0.0.0"


def test_build_server_command_appends_extra_flags_in_order() -> None:
    command = server.build_server_command(
        "Qwen/Qwen3-4B-AWQ", 8000, ("--max-model-len", "4096", "--dtype", "half")
    )

    assert command[-4:] == ["--max-model-len", "4096", "--dtype", "half"]


def test_build_server_command_can_drop_prefix_caching() -> None:
    command = server.build_server_command("Qwen/Qwen3-4B-AWQ", 8000, enable_prefix_caching=False)

    assert "--enable-prefix-caching" not in command


def test_stop_server_without_start_is_a_noop() -> None:
    server.stop_server()
