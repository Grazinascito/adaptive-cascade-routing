"""DatasetLoader NCM — parseia o dataset NCM-RAFT-samall e amostra o conjunto congelado.

Pivot de dataset (2026-11-14): o ASSIN 2 saiu da linha principal porque a tese
do projeto (ganhos da cascata híbrida no eixo System 1 × System 2) exige cauda
ambígua de catálogo, o tipo de conteúdo do blog do iFood. O loader agora resolve
a tarefa "nome de item de catálogo → código NCM de 8 dígitos" sobre
`yurifacanha/NCM-RAFT-samall`; requisitos completos e stories em
docs/dataset-pivot-ncm-spec.md. A implementação ASSIN 2 permanece no histórico
do git (commit `79a8654` e anteriores).
"""

import csv
import logging
import random
import re
from collections import Counter
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Final, Literal, TypedDict

from datasets import load_dataset

logger = logging.getLogger(__name__)

NCM_DATASET_ID = "yurifacanha/NCM-RAFT-samall"
NCM_DATASET_FALLBACK_ID = "yurifacanha/NCM-dataset"

# Taxonomia derivada das descrições embutidas no próprio dataset (decisão da
# spec: derivar da fonte oficial OU do dataset; o dataset tem descrição única
# por código — verificado em 2026-11-14, 114 códigos, 0 variantes).
TAXONOMY_PATH: Final = Path(__file__).resolve().parents[2] / "config" / "ncm_taxonomy.csv"

Subset = Literal["head", "tail"]

# "o codigo da categoria do ALCOOL LIQUIDO START NAT 70% - 5VL é: 22071090"
_PAIR_PATTERN: Final = re.compile(r"o codigo da categoria do (.+?) é: (\d{8})")

MIN_ITEM_NAME_LENGTH: Final = 3


class DatasetDownloadError(Exception):
    """Falha ao baixar ou acessar o dataset NCM no Hugging Face Hub."""


class NCMTaxonomyEntry(TypedDict):
    code: str

    chapter: str

    chapter_title: str

    description: str


class NCMItem(TypedDict):
    """Um par (nome de item, código NCM) extraído do pool."""

    item_id: int

    name: str

    ncm_code: str


class NCMEvalSample(TypedDict):
    """Item do conjunto congelado com o subconjunto head/tail anotado."""

    item_id: int

    name: str

    ncm_code: str

    code_freq: int

    subset: Subset


class PoolStats(TypedDict):
    """Estatísticas da sonda do Story 1 (impressas na célula de sonda)."""

    n_rows: int

    n_parsed: int

    extraction_rate: float

    unique_pairs: int

    unique_names: int

    unique_codes: int

    conflicting_names: int

    dropped_short_names: int


@lru_cache(maxsize=1)
def load_taxonomy() -> dict[str, NCMTaxonomyEntry]:
    """Carrega config/ncm_taxonomy.csv (114 códigos, 19 capítulos).

    Levanta `ValueError` se o CSV tiver código duplicado, descrição vazia ou
    código que não seja 8 dígitos (edge case da Story 2 da spec).
    """
    taxonomy: dict[str, NCMTaxonomyEntry] = {}
    with TAXONOMY_PATH.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            code = row["code"]
            if not re.fullmatch(r"\d{8}", code):
                raise ValueError(f"taxonomy code must be 8 digits, got {code!r}")
            if code in taxonomy:
                raise ValueError(f"taxonomy has duplicate code {code}")
            if not row["description"].strip():
                raise ValueError(f"taxonomy code {code} has a blank description")
            taxonomy[code] = NCMTaxonomyEntry(
                code=code,
                chapter=row["chapter"],
                chapter_title=row["chapter_title"],
                description=row["description"],
            )
    return taxonomy


def is_valid_code(code: str, taxonomy: dict[str, NCMTaxonomyEntry] | None = None) -> bool:
    """True quando `code` existe na taxonomia (gate de escalada do System 1)."""
    entries = taxonomy if taxonomy is not None else load_taxonomy()
    return code in entries


def chapter_lines() -> list[str]:
    """Lista compacta dos capítulos para o prefixo estático dos prompts.

    Ordenada por código de capítulo; uma linha por capítulo no formato
    "22 - Bebidas, líquidos alcoólicos e vinagres.".
    """
    chapters: dict[str, str] = {}
    for entry in load_taxonomy().values():
        chapters.setdefault(entry["chapter"], entry["chapter_title"])
    return [f"{chapter} - {title}" for chapter, title in sorted(chapters.items())]


def parse_conversation(conversations: list[dict[str, str]]) -> NCMItem | None:
    """Extrai (nome, código) da mensagem do usuário; None se não casar.

    O par canônico vive na PRIMEIRA mensagem (user): "o codigo da categoria
    do {nome} é: {código}". A resposta do assistente às vezes traz o código e
    às vezes só a descrição — por isso o parse não depende dela.
    """
    if not conversations:
        return None
    user_content = conversations[0].get("content", "")
    match = _PAIR_PATTERN.search(user_content)
    if match is None:
        return None
    name = match.group(1).strip()
    if len(name) < MIN_ITEM_NAME_LENGTH:
        return None
    return NCMItem(item_id=-1, name=name, ncm_code=match.group(2))


def load_ncm_pool(dataset_id: str = NCM_DATASET_ID) -> tuple[list[NCMItem], int]:
    """Carrega o split train e parseia todas as conversas em NCMItem.

    Retorna (itens, total de linhas do split) para o cálculo honesto da taxa
    de extração da sonda. `item_id` é o índice da linha no dataset
    (determinístico). O dataset tem um único split `train`; não há split de
    teste rotulado.
    """
    try:
        dataset = load_dataset(dataset_id, split="train")
    except (OSError, ValueError) as error:
        raise DatasetDownloadError(
            f"failed to load '{dataset_id}' (split=train) from the Hugging Face Hub: {error}"
        ) from error

    items: list[NCMItem] = []
    for index, row in enumerate(dataset):
        parsed = parse_conversation(row["conversations"])
        if parsed is None:
            continue
        items.append(NCMItem(item_id=index, name=parsed["name"], ncm_code=parsed["ncm_code"]))
    return items, len(dataset)


def pool_stats(items: Sequence[NCMItem], n_rows: int) -> PoolStats:
    """Estatísticas do pool para a sonda (Story 1) — sem chamadas de API."""
    unique_pairs = {(item["name"], item["ncm_code"]) for item in items}
    codes_by_name: dict[str, set[str]] = {}
    for name, code in unique_pairs:
        codes_by_name.setdefault(name, set()).add(code)
    conflicting = sum(1 for codes in codes_by_name.values() if len(codes) > 1)
    short_names = sum(1 for name in codes_by_name if len(name) < MIN_ITEM_NAME_LENGTH)
    return PoolStats(
        n_rows=n_rows,
        n_parsed=len(items),
        extraction_rate=len(items) / n_rows if n_rows else 0.0,
        unique_pairs=len(unique_pairs),
        unique_names=len(codes_by_name),
        unique_codes=len({code for _, code in unique_pairs}),
        conflicting_names=conflicting,
        dropped_short_names=short_names,
    )


def get_frozen_eval_set(
    items: Sequence[NCMItem],
    *,
    sample_limit: int = 800,
    head_share: float = 0.65,
    head_min_code_freq: int = 50,
    seed: int = 42,
) -> list[NCMEvalSample]:
    """Amostra congelada head/tail (Story 3 da spec do pivot).

    Pipeline: dedup de pares → exclusão de nomes conflitantes (mesmo nome com
    códigos diferentes; contados e logados) → frequência de cada código no
    pool limpo → head = códigos com >= `head_min_code_freq` ocorrências →
    amostragem estratificada `round(sample_limit * head_share)` head + resto
    tail com `random.Random(seed)`.
    """
    if sample_limit <= 0:
        raise ValueError(f"sample_limit must be positive, got {sample_limit}")
    if not 0.0 < head_share < 1.0:
        raise ValueError(f"head_share must be in (0, 1), got {head_share}")
    if head_min_code_freq <= 0:
        raise ValueError(f"head_min_code_freq must be positive, got {head_min_code_freq}")

    unique_pairs = sorted({(item["name"], item["ncm_code"]) for item in items})
    first_item_id = {(item["name"], item["ncm_code"]): item["item_id"] for item in items}
    codes_by_name: dict[str, set[str]] = {}
    for name, code in unique_pairs:
        codes_by_name.setdefault(name, set()).add(code)
    conflicting_names = {name for name, codes in codes_by_name.items() if len(codes) > 1}
    if conflicting_names:
        logger.warning(
            "excluded %d conflicting item names (same name, different codes)",
            len(conflicting_names),
        )

    clean_pairs = [(name, code) for name, code in unique_pairs if name not in conflicting_names]
    code_freq = Counter(code for _, code in clean_pairs)

    head_pool = [
        (name, code) for name, code in clean_pairs if code_freq[code] >= head_min_code_freq
    ]
    tail_pool = [(name, code) for name, code in clean_pairs if code_freq[code] < head_min_code_freq]

    n_head = round(sample_limit * head_share)
    n_tail = sample_limit - n_head
    for label, pool, wanted in (("head", head_pool, n_head), ("tail", tail_pool, n_tail)):
        if len(pool) < wanted:
            raise ValueError(
                f"{label} pool has {len(pool)} unique pairs, needs {wanted}; "
                f"adjust head_share/head_min_code_freq/sample_limit"
            )

    rng = random.Random(seed)
    sampled: list[NCMEvalSample] = []

    def _draw(pool: list[tuple[str, str]], wanted: int, subset: Subset) -> None:
        for name, code in rng.sample(sorted(pool), wanted):
            sampled.append(
                NCMEvalSample(
                    item_id=first_item_id[(name, code)],
                    name=name,
                    ncm_code=code,
                    code_freq=code_freq[code],
                    subset=subset,
                )
            )

    _draw(head_pool, n_head, "head")
    _draw(tail_pool, n_tail, "tail")
    return sampled


def truncate_code(code: str, digits: int = 8) -> str:
    """Trunca o código para `digits` dígitos (granularidade do rótulo).

    `digits` válido: 2 (capítulo), 4 (posição), 6 (subposição) ou 8 (item).
    A sonda do notebook decide o nível; o truncamento é aplicado de forma
    idêntica a predições e ground truths na avaliação.
    """
    if digits not in (2, 4, 6, 8):
        raise ValueError(f"digits must be one of (2, 4, 6, 8), got {digits}")
    return code[:digits]
