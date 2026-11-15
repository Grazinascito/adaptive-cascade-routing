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


def test_preserves_the_frozen_benchmark_contract_seed_42_samples_800_alpha_085() -> None:
    config = _load_config()
    assert config["dataset"]["seed"] == 42
    assert config["dataset"]["frozen_samples"] == 800
    assert config["dataset"]["head_share"] == 0.65
    assert config["dataset"]["head_min_code_freq"] == 50
    assert config["dataset"]["label_digits"] == 8
    assert config["router"]["alpha"] == 0.85
    assert config["system2"]["max_tokens"] == 256
    assert config["model"]["id"] == "Qwen/Qwen3-4B-AWQ"


def test_system1_contract_freezes_the_short_generation_pivot_decision() -> None:
    config = _load_config()
    # Pivot NCM (Decisão 1(d) da spec): fast-path é geração curta no endpoint
    # de chat, não a pergunta choice do /v1/systemone.
    assert config["system1"]["endpoint"] == "/v1/chat/completions"
    assert config["system1"]["max_tokens"] == 12
    assert config["system1"]["top_logprobs"] == 20


def test_kpis_freeze_the_pivot_gates_tail_gap_and_recovery() -> None:
    config = _load_config()
    assert config["kpis"]["p95_reduction"] == 0.50
    assert config["kpis"]["token_savings"] == 0.60
    assert config["kpis"]["tail_gap_pp"] == 10
    assert config["kpis"]["recovery_share"] == 0.50
