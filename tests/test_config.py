"""Testes de fundação: o contrato congelado do benchmark vive em config/benchmark_config.yaml."""

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent.parent / "config" / "benchmark_config.yaml"


def _load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_loads_benchmark_config_yaml_and_contains_every_section_the_specs_require() -> None:
    config = _load_config()
    required_sections = (
        "dataset",
        "model",
        "serving",
        "system1",
        "system2",
        "router",
        "metrics",
        "kpis",
    )
    for section in required_sections:
        assert section in config, f"seção '{section}' faltando em benchmark_config.yaml"


def test_preserves_the_frozen_benchmark_contract_seed_42_samples_200_alpha_085_max_tokens() -> None:
    config = _load_config()
    assert config["dataset"]["seed"] == 42
    assert config["dataset"]["frozen_samples"] == 200
    assert config["router"]["alpha"] == 0.85
    assert config["system1"]["max_tokens"] == 1
    assert config["system2"]["max_tokens"] == 256
    assert config["model"]["id"] == "Qwen/Qwen3-4B-Instruct-2507-AWQ"
