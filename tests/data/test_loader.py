"""Testes do Componente A: ingestão, normalização e amostragem do ASSIN 2.

Cada teste corresponde a um cenário de aceite de docs/dataset-loader-spec.md.
Os testes de ingestão baixam o dataset real do Hugging Face na primeira execução
por máquina; depois, a biblioteca `datasets` usa a cópia local em cache.
"""

import hashlib
import json
import logging
from collections import Counter
from collections.abc import Iterator
from typing import Any

import pytest

from src.data.loader import (
    SYSTEM1_SYSTEM_PREFIX,
    DatasetDownloadError,
    format_system1_prompt,
    get_frozen_eval_set,
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


class _StubDataset:
    """Stub com a interface de `datasets.Dataset` usada pelo loader.

    `entailment_judgment` chega como string (não como ClassLabel int),
    exercitando o caminho sem decoding `int2str`.
    """

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.column_names = [
            "sentence_pair_id",
            "premise",
            "hypothesis",
            "relatedness_score",
            "entailment_judgment",
        ]
        self.features: dict[str, Any] = {"entailment_judgment": "string"}
        self._rows = rows

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._rows)


def test_load_assin2_raw_split_filters_empty_or_blank_rows_with_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    rows: list[dict[str, Any]] = [
        {
            "sentence_pair_id": 1,
            "premise": "premissa válida",
            "hypothesis": "hipótese válida",
            "relatedness_score": 4.0,
            "entailment_judgment": "ENTAILMENT",
        },
        {
            "sentence_pair_id": 2,
            "premise": "",
            "hypothesis": "hipótese",
            "relatedness_score": 3.0,
            "entailment_judgment": "NONE",
        },
        {
            "sentence_pair_id": 3,
            "premise": "premissa",
            "hypothesis": None,
            "relatedness_score": 3.0,
            "entailment_judgment": "NONE",
        },
        {
            "sentence_pair_id": 4,
            "premise": "   ",
            "hypothesis": "hipótese",
            "relatedness_score": 3.0,
            "entailment_judgment": "NONE",
        },
    ]

    def _stub_load_dataset(*_args: Any, **_kwargs: Any) -> _StubDataset:
        return _StubDataset(rows)

    monkeypatch.setattr("src.data.loader.load_dataset", _stub_load_dataset)

    with caplog.at_level(logging.WARNING, logger="src.data.loader"):
        samples = load_assin2_raw_split(split="test")

    assert [sample["id"] for sample in samples] == [1]
    assert samples[0]["label"] == "Entailment"
    assert "dropped 3 rows" in caplog.text


def test_format_system1_prompt_keeps_system_prefix_byte_identical_across_pairs() -> None:
    first = format_system1_prompt("O gato dorme no sofá.", "O gato está dormindo.")
    second = format_system1_prompt("A rua está molhada.", "Choveu ontem à noite.")

    assert first["system_prefix"] == second["system_prefix"]
    assert first["system_prefix"] == SYSTEM1_SYSTEM_PREFIX


def test_format_system1_prompt_embeds_texts_verbatim_inside_fixed_xml_markers() -> None:
    payload = format_system1_prompt("O gato dorme no sofá.", "O gato está dormindo.")

    assert payload["user_state"] == (
        "<premise>\nO gato dorme no sofá.\n</premise>\n"
        "<hypothesis>\nO gato está dormindo.\n</hypothesis>"
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


def test_format_system1_prompt_rejects_texts_containing_reserved_xml_markers() -> None:
    invalid_texts: list[Any] = [
        "texto com </premise> injetado",
        "texto com <premise> injetado",
        "texto com </PREMISE> injetado",
        "texto com </hypothesis> injetado",
        "texto com <Hypothesis> injetado",
    ]
    valid_pair = "texto válido"

    for invalid in invalid_texts:
        with pytest.raises(ValueError, match="reserved XML markers"):
            format_system1_prompt(invalid, valid_pair)
        with pytest.raises(ValueError, match="reserved XML markers"):
            format_system1_prompt(valid_pair, invalid)


def test_get_frozen_eval_set_returns_200_samples_balanced_100_and_100() -> None:
    samples = get_frozen_eval_set()

    assert len(samples) == 200
    label_counts = Counter(sample["label"] for sample in samples)
    assert label_counts == Counter({"Entailment": 100, "Contradiction": 100})
    expected_fields = {"id", "premise", "hypothesis", "label", "original_score"}
    for sample in samples:
        assert set(sample) == expected_fields


def test_get_frozen_eval_set_is_bit_exact_across_repeated_calls_with_seed_42() -> None:
    first_run = get_frozen_eval_set()
    second_run = get_frozen_eval_set()

    first_hash = hashlib.sha256(json.dumps(first_run, sort_keys=True).encode()).hexdigest()
    second_hash = hashlib.sha256(json.dumps(second_run, sort_keys=True).encode()).hexdigest()
    assert first_hash == second_hash


def test_get_frozen_eval_set_rejects_odd_or_excessive_sample_limits() -> None:
    with pytest.raises(ValueError, match="even"):
        get_frozen_eval_set(sample_limit=201)

    with pytest.raises(ValueError, match="1224"):
        get_frozen_eval_set(sample_limit=2500)
