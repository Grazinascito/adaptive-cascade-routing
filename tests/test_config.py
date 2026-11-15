"""Testes de fundação: contrato congelado do benchmark DARU em config/benchmark_config.yaml.

`load_benchmark_config` é o helper reutilizável: outros arquivos de teste podem
importá-lo (`from tests.test_config import load_benchmark_config`) — o YAML é
lido uma única vez por processo.
"""

import re
from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent.parent / "config" / "benchmark_config.yaml"

# Mesmo padrão de não-comida do loader (src/data/loader.py); duplicado aqui de
# propósito para o teste não herdar um bug do módulo que deveria pegar.
_NON_FOOD_TERMS = re.compile(
    r"LIMPEZA|HIGIENE|COSMET|PET|BAZAR|ELETRO|PAPELARIA|UTENSIL|VESTU|CALCAD|"
    r"TEXTIL|FARMA|JOIA|BRINQUEDO|ESPORT|AUTO|MOVEIS|JARDIM|LIVROS",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def load_benchmark_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_config_contains_every_section_the_spec_requires() -> None:
    config = load_benchmark_config()
    for section in (
        "dataset",
        "model",
        "serving",
        "system1",
        "system2",
        "router",
        "metrics",
        "kpis",
    ):
        assert section in config, f"seção '{section}' faltando em benchmark_config.yaml"


def test_dataset_freezes_the_daru_contract() -> None:
    dataset = load_benchmark_config()["dataset"]
    assert dataset["name"] == "gilsileydaru/classificao-produtos-varejo-cpg-prbr"
    assert dataset["file"] == "DATASETPRODUTOSVAREJOBRASIL.xlsx"
    assert len(dataset["categories"]) == 12
    assert len(set(dataset["categories"])) == 12
    assert all(isinstance(name, str) and name for name in dataset["categories"])
    assert dataset["validation"] == 200
    assert dataset["test"] == 600
    assert dataset["head_share"] == 0.7
    assert dataset["tail_max_chars"] == 20
    assert dataset["seed"] == 42


def test_categories_exclude_non_food_names() -> None:
    categories = load_benchmark_config()["dataset"]["categories"]
    for name in categories:
        assert not _NON_FOOD_TERMS.search(name), f"categoria não-comida pinada: {name}"


def test_model_and_router_freeze_qwen3_awq_and_alpha() -> None:
    config = load_benchmark_config()
    assert config["model"]["id"] == "Qwen/Qwen3-4B-AWQ"
    assert config["model"]["quantization"] == "awq"
    assert config["router"]["alpha"] == 0.85


def test_serving_uses_local_vllm_with_prefix_caching() -> None:
    serving = load_benchmark_config()["serving"]
    assert serving["base_url"] == "http://127.0.0.1:8000/v1"
    assert serving["port"] == 8000
    assert serving["prefix_caching"] is True


def test_system1_is_the_systemone_choice_endpoint() -> None:
    system1 = load_benchmark_config()["system1"]
    assert system1["endpoint"] == "/v1/systemone"
    assert system1["alias_letters"] == list("ABCDEFGHIJKL")


def test_system2_is_cot_chat_completions_without_thinking() -> None:
    system2 = load_benchmark_config()["system2"]
    assert system2["endpoint"] == "/v1/chat/completions"
    assert system2["max_tokens"] == 256
    assert system2["enable_thinking"] is False


def test_kpis_freeze_the_spec_gates() -> None:
    kpis = load_benchmark_config()["kpis"]
    assert kpis["tail_gap_pp"] == 10
    assert kpis["recovery_share"] == 0.5
    assert kpis["token_share_vs_s2"] == 0.4
