"""Loader DARU — parseia DATASETPRODUTOSVAREJOBRASIL.xlsx e amostra os conjuntos congelados.

Pivot de dataset (2026-11-15, SPEC-DARU.md): o benchmark resolve "descrição de
item de catálogo → categoria" sobre o DARU (18 maiores varejistas do Brasil,
Darú et al. 2022). O Excel veio com as 5 colunas embolidas em 1 célula; a
regra de leitura é csv (delimiter=';', quotechar='"') mantendo só linhas com
exatamente 5 campos: descrição, segmento, categoria, subcategoria, produto
canônico.
"""

import csv
import random
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from openpyxl import load_workbook

DARU_DATASET_ID = "gilsileydaru/classificao-produtos-varejo-cpg-prbr"
DARU_XLSX_FILENAME = "DATASETPRODUTOSVAREJOBRASIL.xlsx"

_CATEGORIA_PREFIX = re.compile(r"^\d+\s*-\s*")

# Categorias de não-comida/bebida fora do escopo do benchmark (SPEC-DARU.md §2:
# escopo = comida e bebida, como no contexto do iFood).
_NON_FOOD_PATTERN = re.compile(
    r"LIMPEZA|HIGIENE|COSMET|PET|BAZAR|ELETRO|PAPELARIA|UTENSIL|VESTU|CALCAD|"
    r"TEXTIL|FARMA|JOIA|BRINQUEDO|ESPORT|AUTO|MOVEIS|JARDIM|LIVROS|BABY",
    re.IGNORECASE,
)

DaruRow = tuple[str, str]


class DatasetDownloadError(Exception):
    """Falha ao baixar ou localizar o xlsx do DARU no kagglehub."""


@dataclass(frozen=True)
class CatalogSample:
    """Item do conjunto congelado (nível de rótulo: categoria)."""

    sample_id: int
    raw_text: str
    canonical_category: str
    is_long_tail: bool


@dataclass(frozen=True)
class BenchmarkSets:
    """Conjuntos congelados: validação (escolha do α) e teste (número final)."""

    validation: list[CatalogSample]
    test: list[CatalogSample]


def parse_daru_cell(cell: str) -> DaruRow | None:
    """Separa uma célula embolida em (descrição, categoria); None se não casar.

    Só células com exatamente 5 campos entram (79% do arquivo segundo a spec);
    o prefixo numérico da categoria ("6-MERCEARIA") é removido.
    """
    fields = next(csv.reader([cell], delimiter=";", quotechar='"'), [])
    if len(fields) != 5:
        return None
    description = fields[0].strip()
    categoria = _CATEGORIA_PREFIX.sub("", fields[2]).strip()
    if not description or not categoria:
        return None
    return description, categoria


def parse_daru_xlsx(path: str | Path) -> list[DaruRow]:
    """Lê o xlsx de coluna única e devolve as linhas (descrição, categoria).

    A primeira célula não vazia é o cabeçalho e fica fora.
    """
    workbook = load_workbook(path, read_only=True)
    try:
        worksheet = workbook.active
        assert worksheet is not None
        rows: list[DaruRow] = []
        header_seen = False
        for row in worksheet.iter_rows(values_only=True):
            if not row or row[0] is None:
                continue
            if not header_seen:
                header_seen = True
                continue
            parsed = parse_daru_cell(str(row[0]))
            if parsed is not None:
                rows.append(parsed)
        return rows
    finally:
        workbook.close()


def select_food_categories(
    rows: Sequence[DaruRow], n: int = 12, min_rows: int = 500
) -> list[tuple[str, int]]:
    """Top-n categorias por contagem, fora as de não-comida/bebida.

    Exclui nomes que casam com o padrão de não-comida (limpeza, higiene, pet,
    bazar etc.) e categorias com menos de `min_rows` exemplos.
    """
    counts = Counter(categoria for _, categoria in rows)
    eligible = [
        (name, count)
        for name, count in counts.items()
        if count >= min_rows and not _NON_FOOD_PATTERN.search(name)
    ]
    eligible.sort(key=lambda item: (-item[1], item[0]))
    return eligible[:n]


def load_catalog_benchmark_set(
    xlsx_path: str | Path,
    categories: Sequence[str],
    *,
    n_validation: int,
    n_test: int,
    head_share: float,
    tail_max_chars: int,
    seed: int,
) -> BenchmarkSets:
    """Amostra validação e teste estratificados cabeça/cauda (SPEC-DARU.md §3).

    Pipeline: parse → filtro pelo conjunto fechado de categorias → dedup
    case-insensitive de descrição (primeira ocorrência vence) → estratos
    cabeça (> `tail_max_chars`) e cauda (<= `tail_max_chars`) ordenados por
    descrição → sorteio com `random.Random(seed)`: validação primeiro, teste
    sobre o que sobrou (sem sobreposição). Cada conjunto recebe
    `round(n * head_share)` itens de cabeça e o resto de cauda. `sample_id` é
    sequencial na ordem final (validação, depois teste).
    """
    if n_validation <= 0 or n_test <= 0:
        raise ValueError(f"n_validation and n_test must be positive, got {n_validation}/{n_test}")
    if not 0.0 < head_share < 1.0:
        raise ValueError(f"head_share must be in (0, 1), got {head_share}")
    if tail_max_chars < 0:
        raise ValueError(f"tail_max_chars must be non-negative, got {tail_max_chars}")

    allowed = set(categories)
    seen: set[str] = set()
    unique_rows: list[DaruRow] = []
    for row in parse_daru_xlsx(_resolve_xlsx_path(Path(xlsx_path))):
        if row[1] not in allowed:
            continue
        key = row[0].casefold()
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)

    head_pool = sorted(row for row in unique_rows if len(row[0].strip()) > tail_max_chars)
    tail_pool = sorted(row for row in unique_rows if len(row[0].strip()) <= tail_max_chars)

    n_val_head = round(n_validation * head_share)
    n_val_tail = n_validation - n_val_head
    n_test_head = round(n_test * head_share)
    n_test_tail = n_test - n_test_head
    for label, available, wanted in (
        ("validation head", len(head_pool), n_val_head),
        ("validation tail", len(tail_pool), n_val_tail),
        ("test head", len(head_pool) - n_val_head, n_test_head),
        ("test tail", len(tail_pool) - n_val_tail, n_test_tail),
    ):
        if available < wanted:
            raise ValueError(
                f"{label} pool has {available} unique rows, needs {wanted}; "
                f"adjust categories/head_share/tail_max_chars/sample sizes"
            )

    rng = random.Random(seed)
    validation = rng.sample(head_pool, n_val_head) + rng.sample(tail_pool, n_val_tail)
    used = {row[0].casefold() for row in validation}
    head_rest = [row for row in head_pool if row[0].casefold() not in used]
    tail_rest = [row for row in tail_pool if row[0].casefold() not in used]
    test = rng.sample(head_rest, n_test_head) + rng.sample(tail_rest, n_test_tail)

    samples = [
        CatalogSample(
            sample_id=sample_id,
            raw_text=row[0],
            canonical_category=row[1],
            is_long_tail=len(row[0].strip()) <= tail_max_chars,
        )
        for sample_id, row in enumerate([*validation, *test])
    ]
    return BenchmarkSets(
        validation=samples[:n_validation],
        test=samples[n_validation:],
    )


def _resolve_xlsx_path(xlsx_path: Path) -> Path:
    if xlsx_path.exists():
        return xlsx_path
    import kagglehub

    root = Path(kagglehub.dataset_download(DARU_DATASET_ID))
    matches = sorted(root.rglob(DARU_XLSX_FILENAME))
    if not matches:
        raise DatasetDownloadError(
            f"{DARU_XLSX_FILENAME} not found inside kagglehub download {root}"
        )
    return matches[0]
