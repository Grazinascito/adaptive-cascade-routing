"""Testes do Componente A: ingestão, normalização e amostragem do ASSIN 2.

Cada teste corresponde a um cenário de aceite de docs/dataset-loader-spec.md.
Os testes de ingestão baixam o dataset real do Hugging Face na primeira execução
por máquina; depois, a biblioteca `datasets` usa a cópia local em cache.
"""

from collections import Counter
from typing import Any

import pytest

from src.data.loader import (
    SYSTEM1_SYSTEM_PREFIX,
    DatasetDownloadError,
    format_system1_prompt,
    load_assin2_raw_split,
    normalize_label,
)


def test_normalize_label_maps_entailment_and_none_to_the_binary_contract() -> None:
    assert normalize_label("ENTAILMENT") == "Entailment"
    assert normalize_label("NONE") == "Contradiction"


def test_normalize_label_raises_invalid_assin2_label_for_any_other_value() -> None:
    invalid_values: list[Any] = ["PARAPHRASE", "", "entailment", 1, None]
    for raw in invalid_values:
        with pytest.raises(ValueError, match="Invalid ASSIN2 label"):
            normalize_label(raw)


def test_load_assin2_raw_split_returns_all_2448_test_samples_fully_labeled() -> None:
    samples = load_assin2_raw_split(split="test")

    assert len(samples) == 2448
    for sample in samples:
        assert isinstance(sample["premise"], str) and sample["premise"].strip()
        assert isinstance(sample["hypothesis"], str) and sample["hypothesis"].strip()
        assert sample["label"] in ("Entailment", "Contradiction")
        assert 1.0 <= sample["original_score"] <= 5.0

    ids = [sample["id"] for sample in samples]
    assert len(set(ids)) == len(ids)

    label_counts = Counter(sample["label"] for sample in samples)
    assert label_counts == Counter({"Entailment": 1224, "Contradiction": 1224})


def test_load_assin2_raw_split_wraps_download_failure_in_dataset_download_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_oserror(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("hub unreachable")

    monkeypatch.setattr("src.data.loader.load_dataset", _raise_oserror)

    with pytest.raises(DatasetDownloadError):
        load_assin2_raw_split(split="test")


def test_format_system1_prompt_keeps_system_prefix_byte_identical_across_pairs() -> None:
    first = format_system1_prompt("O gato dorme no sofá.", "O gato está dormindo.")
    second = format_system1_prompt("A rua está molhada.", "Choveu ontem à noite.")

    assert first["system_prefix"] == second["system_prefix"]
    assert first["system_prefix"] == SYSTEM1_SYSTEM_PREFIX


def test_format_system1_prompt_embeds_texts_verbatim_inside_fixed_xml_markers() -> None:
    premise = "O gato dorme no sofá."
    hypothesis = "O gato está dormindo."
    payload = format_system1_prompt(premise, hypothesis)

    assert payload["user_state"] == (
        f"<premise>\n{premise}\n</premise>\n<hypothesis>\n{hypothesis}\n</hypothesis>"
    )
    assert payload["choices"] == ["Entailment", "Contradiction"]


def test_format_system1_prompt_rejects_non_string_or_blank_inputs() -> None:
    invalid_values: list[Any] = ["", "   ", None, 123]
    valid_premise = "premissa válida"

    for invalid in invalid_values:
        with pytest.raises(ValueError):
            format_system1_prompt(invalid, valid_premise)
        with pytest.raises(ValueError):
            format_system1_prompt(valid_premise, invalid)
