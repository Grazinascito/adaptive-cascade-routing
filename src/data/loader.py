"""DatasetLoader — carrega, filtra e formata o dataset ASSIN 2 em pares de entrada estruturados.

Componente A do design doc (docs/adaptive-cascade-design-doc.md §5.1);
requisitos completos e stories em docs/dataset-loader-spec.md.
"""

import logging
import random
from typing import Final, Literal, TypedDict

from datasets import ClassLabel, load_dataset

logger = logging.getLogger(__name__)

ASSIN2_DATASET_ID = "nilc-nlp/assin2"

ENTAILMENT_LABEL: Final = "Entailment"
CONTRADICTION_LABEL: Final = "Contradiction"

Label = Literal["Entailment", "Contradiction"]

_LABEL_MAP: dict[str, Label] = {"ENTAILMENT": ENTAILMENT_LABEL, "NONE": CONTRADICTION_LABEL}

_REQUIRED_COLUMNS = (
    "sentence_pair_id",
    "premise",
    "hypothesis",
    "relatedness_score",
    "entailment_judgment",
)


class DatasetDownloadError(Exception):
    """Falha ao baixar ou acessar o dataset ASSIN 2 no Hugging Face Hub."""


class ASSIN2Sample(TypedDict):
    id: int

    premise: str

    hypothesis: str

    label: Label

    original_score: float


SYSTEM1_SYSTEM_PREFIX = (
    "You are a Natural Language Inference classifier for Brazilian Portuguese.\n"
    "Read the premise and the hypothesis inside the XML tags and decide exactly one label:\n"
    "- Entailment: the hypothesis follows from the premise.\n"
    "- Contradiction: the hypothesis does not follow from the premise.\n"
    "Answer with the label text only."
)


class FormattedPromptPayload(TypedDict):
    system_prefix: str

    user_state: str

    choices: list[str]


_XML_MARKERS = ("<premise>", "</premise>", "<hypothesis>", "</hypothesis>")


def _require_text(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    lowered = value.casefold()
    if any(marker in lowered for marker in _XML_MARKERS):
        raise ValueError(
            f"{name} must not contain reserved XML markers "
            f"({_XML_MARKERS}); they would break the prompt structure"
        )


def _is_blank(value: object) -> bool:
    return not isinstance(value, str) or not value.strip()


def format_system1_prompt(premise: str, hypothesis: str) -> FormattedPromptPayload:
    _require_text("premise", premise)
    _require_text("hypothesis", hypothesis)
    return FormattedPromptPayload(
        system_prefix=SYSTEM1_SYSTEM_PREFIX,
        user_state=f"<premise>\n{premise}\n</premise>\n<hypothesis>\n{hypothesis}\n</hypothesis>",
        choices=[ENTAILMENT_LABEL, CONTRADICTION_LABEL],
    )


def get_frozen_eval_set(
    split: str = "test",
    sample_limit: int = 200,
    seed: int = 42,
) -> list[ASSIN2Sample]:
    if sample_limit <= 0 or sample_limit % 2 != 0:
        raise ValueError(f"sample_limit must be a positive even number, got {sample_limit}")

    samples = load_assin2_raw_split(split=split)
    per_class = sample_limit // 2

    by_label: dict[str, list[ASSIN2Sample]] = {ENTAILMENT_LABEL: [], CONTRADICTION_LABEL: []}
    for sample in samples:
        by_label[sample["label"]].append(sample)

    for label, group in by_label.items():
        if len(group) < per_class:
            logger.error(
                "not enough '%s' samples in split '%s': need %d, available %d",
                label,
                split,
                per_class,
                len(group),
            )
            raise ValueError(
                f"sample_limit={sample_limit} needs {per_class} '{label}' samples, "
                f"but split '{split}' only provides {len(group)}"
            )

    rng = random.Random(seed)
    return rng.sample(by_label["Entailment"], per_class) + rng.sample(
        by_label["Contradiction"], per_class
    )


def normalize_label(raw: str) -> Label:
    if raw in _LABEL_MAP:
        return _LABEL_MAP[raw]
    raise ValueError("Invalid ASSIN2 label")


def load_assin2_raw_split(split: str = "test") -> list[ASSIN2Sample]:
    try:
        dataset = load_dataset(ASSIN2_DATASET_ID, split=split)
    except (OSError, ValueError) as error:
        raise DatasetDownloadError(
            f"failed to load '{ASSIN2_DATASET_ID}' (split={split!r}) "
            f"from the Hugging Face Hub: {error}"
        ) from error

    missing_columns = [column for column in _REQUIRED_COLUMNS if column not in dataset.column_names]
    if missing_columns:
        raise ValueError(
            f"'{ASSIN2_DATASET_ID}' (split={split!r}) "
            f"is missing required columns: {missing_columns}"
        )

    label_feature = dataset.features["entailment_judgment"]

    samples: list[ASSIN2Sample] = []
    dropped_rows = 0
    for row in dataset:
        premise = row["premise"]
        hypothesis = row["hypothesis"]
        if _is_blank(premise) or _is_blank(hypothesis):
            dropped_rows += 1
            continue
        raw_label = row["entailment_judgment"]
        if isinstance(raw_label, int) and isinstance(label_feature, ClassLabel):
            raw_label = label_feature.int2str(raw_label)
        samples.append(
            ASSIN2Sample(
                id=row["sentence_pair_id"],
                premise=premise,
                hypothesis=hypothesis,
                label=normalize_label(raw_label),
                original_score=float(row["relatedness_score"]),
            )
        )

    if dropped_rows:
        logger.warning(
            "dropped %d rows from '%s' (split=%s) with empty or blank premise or hypothesis",
            dropped_rows,
            ASSIN2_DATASET_ID,
            split,
        )

    return samples
