"""Tests for the DARU loader: parser, closed-set filter, dedup and stratified sampling.

No network: data comes from tests/fixtures/daru_sample.json (40 REAL rows
sampled from the cached xlsx with random.Random(42)); the temporary xlsx
files are built with openpyxl from those rows, re-squashing the 5 fields
into 1 cell in the original file's format.
"""

import csv
import io
import json
from pathlib import Path

import pytest
from openpyxl import Workbook

from src.data import loader

FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "daru_sample.json"


def fixture_rows() -> list[tuple[str, str]]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return [(row["description"], row["categoria"]) for row in payload]


def squish(description: str, categoria: str, *, prefix: str = "6-") -> str:
    """Rebuilds a squashed 5-field cell in DARU's real format."""
    buffer = io.StringIO()
    csv.writer(buffer, delimiter=";", quotechar='"', lineterminator="").writerow(
        [description, "1-BENS DE CONSUMO", f"{prefix}{categoria}", "9-SUBCATEGORIA", "PRODUTO"]
    )
    return buffer.getvalue()


def write_xlsx(path: Path, cells: list[str]) -> Path:
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.append([squish("nm_item", "categoria", prefix="")])
    for cell in cells:
        sheet.append([cell])
    workbook.save(path)
    return path


class TestParseDaruCell:
    def test_keeps_real_five_field_cell_and_strips_categoria_prefix(self) -> None:
        description, categoria = fixture_rows()[0]
        parsed = loader.parse_daru_cell(squish(description, categoria))
        assert parsed == (description, categoria)

    def test_strips_prefix_with_spaces_around_the_dash(self) -> None:
        parsed = loader.parse_daru_cell(squish("ACUCAR REF DOCE KG", "MERCEARIA", prefix="10 - "))
        assert parsed == ("ACUCAR REF DOCE KG", "MERCEARIA")

    def test_drops_cells_without_exactly_five_fields(self) -> None:
        assert loader.parse_daru_cell("DESCRICAO SEM PONTO E VIRGULA") is None
        assert loader.parse_daru_cell('A;"1-X";"2-Y"') is None
        assert loader.parse_daru_cell('A;"1-X";"2-Y";"3-Z";"4-W";"5-V"') is None

    def test_drops_cells_with_empty_description_or_categoria(self) -> None:
        assert loader.parse_daru_cell(squish("   ", "MERCEARIA")) is None
        assert loader.parse_daru_cell(squish("ACUCAR KG", "  ")) is None


class TestParseDaruXlsx:
    def test_reads_column_skips_header_and_drops_bad_rows(self, tmp_path: Path) -> None:
        rows = fixture_rows()[:3]
        cells = [
            squish(*rows[0]),
            "APRESUNTADO SEM TAXONOMIA",
            'A;"1-X";"2-Y"',
            squish(*rows[1]),
            squish(*rows[2]),
        ]
        parsed = loader.parse_daru_xlsx(write_xlsx(tmp_path / "daru.xlsx", cells))
        assert parsed == rows


class TestSelectFoodCategories:
    def test_ranks_by_count_excludes_non_food_and_min_rows(self) -> None:
        rows = (
            [("ITEM A", "MERCEARIA")] * 600
            + [("ITEM B", "BEBIDAS ALCOOLICAS")] * 550
            + [("ITEM C", "PRODUTOS DE LIMPEZA")] * 900
            + [("ITEM D", "LACTEOS")] * 499
            + [("ITEM E", "HORTIFRUTI (FLV)")] * 500
        )
        selected = loader.select_food_categories(rows, n=2, min_rows=500)
        assert selected == [("MERCEARIA", 600), ("BEBIDAS ALCOOLICAS", 550)]

    def test_breaks_count_ties_by_name_for_determinism(self) -> None:
        rows = [("X1", "AA")] * 500 + [("X2", "AB")] * 500 + [("X3", "AC")] * 700
        assert loader.select_food_categories(rows, n=3, min_rows=500) == [
            ("AC", 700),
            ("AA", 500),
            ("AB", 500),
        ]

    def test_fixture_rows_never_trigger_the_non_food_pattern_on_food_names(self) -> None:
        rows = [(desc, "MERCEARIA") for desc, _ in fixture_rows()]
        assert loader.select_food_categories(rows, n=1, min_rows=1) == [("MERCEARIA", len(rows))]


def _big_pool() -> list[tuple[str, str]]:
    rows = fixture_rows()
    heads = [(f"CERVEJA ARTESANAL LAGOA SECA GARRAFA {i:03d}", "MERCEARIA") for i in range(30)] + [
        (f"IOGURTE MORANGO COPO 180G CASA {i:03d}", "FRIOS E LATICINIOS") for i in range(30)
    ]
    tails = [(f"REFRI {i:02d}", "MERCEARIA") for i in range(15)] + [
        (f"IOG {i:02d}", "FRIOS E LATICINIOS") for i in range(15)
    ]
    return rows + heads + tails


def _write_pool_xlsx(tmp_path: Path, rows: list[tuple[str, str]]) -> Path:
    return write_xlsx(tmp_path / "pool.xlsx", [squish(desc, cat) for desc, cat in rows])


FOOD_CATEGORIES = ["MERCEARIA", "FRIOS E LATICINIOS"]


def _load(
    path: Path,
    *,
    seed: int = 42,
    n_validation: int = 10,
    n_test: int = 30,
    head_share: float = 0.7,
) -> loader.BenchmarkSets:
    return loader.load_catalog_benchmark_set(
        path,
        FOOD_CATEGORIES,
        n_validation=n_validation,
        n_test=n_test,
        head_share=head_share,
        tail_max_chars=20,
        seed=seed,
    )


class TestLoadCatalogBenchmarkSet:
    def test_stratified_sizes_and_closed_set_filter(self, tmp_path: Path) -> None:
        path = _write_pool_xlsx(tmp_path, _big_pool())
        result = _load(path)
        assert len(result.validation) == 10
        assert len(result.test) == 30
        assert sum(not s.is_long_tail for s in result.validation) == 7
        assert sum(s.is_long_tail for s in result.validation) == 3
        assert sum(not s.is_long_tail for s in result.test) == 21
        assert sum(s.is_long_tail for s in result.test) == 9
        for sample in [*result.validation, *result.test]:
            assert sample.canonical_category in FOOD_CATEGORIES
            expected_tail = len(sample.raw_text.strip()) <= 20
            assert sample.is_long_tail is expected_tail

    def test_validation_and_test_share_no_description(self, tmp_path: Path) -> None:
        result = _load(_write_pool_xlsx(tmp_path, _big_pool()))
        validation = {s.raw_text.casefold() for s in result.validation}
        test = {s.raw_text.casefold() for s in result.test}
        assert validation.isdisjoint(test)
        all_samples = [*result.validation, *result.test]
        texts = [s.raw_text.casefold() for s in all_samples]
        assert len(set(texts)) == len(texts)
        assert sorted(s.sample_id for s in all_samples) == list(range(40))

    def test_same_seed_reproduces_and_other_seed_reorders(self, tmp_path: Path) -> None:
        path = _write_pool_xlsx(tmp_path, _big_pool())
        first = _load(path)
        assert _load(path) == first
        other = _load(path, seed=43)
        assert [s.raw_text for s in other.validation] != [s.raw_text for s in first.validation]
        assert [s.raw_text for s in other.test] != [s.raw_text for s in first.test]

    def test_exactly_twenty_chars_is_tail_and_twenty_one_is_head(self, tmp_path: Path) -> None:
        rows = [
            ("A" * 20, "MERCEARIA"),
            ("B" * 20, "MERCEARIA"),
            ("C" * 21, "MERCEARIA"),
            ("D" * 22, "MERCEARIA"),
            ("E" * 23, "MERCEARIA"),
        ]
        result = _load(_write_pool_xlsx(tmp_path, rows), n_validation=2, n_test=2, head_share=0.5)
        samples = [*result.validation, *result.test]
        assert all(s.is_long_tail for s in samples if len(s.raw_text) == 20)
        assert all(not s.is_long_tail for s in samples if len(s.raw_text) >= 21)
        assert len(samples) == 4

    def test_dedups_descriptions_case_insensitively_keeping_first(self, tmp_path: Path) -> None:
        rows = [
            ("PROD X", "MERCEARIA"),
            ("prod x", "FRIOS E LATICINIOS"),
            ("PROD Y", "FRIOS E LATICINIOS"),
            ("PRODUTOS Z HEAD LONGO PARA CABECA", "MERCEARIA"),
            ("PRODUTOS W HEAD LONGO PARA CABECA", "MERCEARIA"),
        ]
        result = _load(_write_pool_xlsx(tmp_path, rows), n_validation=2, n_test=2)
        samples = [*result.validation, *result.test]
        assert sorted(s.raw_text for s in samples) == [
            "PROD X",
            "PROD Y",
            "PRODUTOS W HEAD LONGO PARA CABECA",
            "PRODUTOS Z HEAD LONGO PARA CABECA",
        ]
        prod_x = next(s for s in samples if s.raw_text == "PROD X")
        assert prod_x.canonical_category == "MERCEARIA"

    def test_raises_when_tail_pool_is_smaller_than_requested(self, tmp_path: Path) -> None:
        path = _write_pool_xlsx(tmp_path, _big_pool())
        with pytest.raises(ValueError, match="tail pool"):
            _load(path, n_validation=20, head_share=0.1)

    def test_rejects_invalid_parameters(self, tmp_path: Path) -> None:
        path = _write_pool_xlsx(tmp_path, _big_pool())
        with pytest.raises(ValueError, match="head_share"):
            _load(path, head_share=1.0)
        with pytest.raises(ValueError, match="positive"):
            _load(path, n_test=0)


def test_fixture_has_forty_real_rows_with_mixed_strata() -> None:
    rows = fixture_rows()
    assert len(rows) == 40
    assert all(description and categoria for description, categoria in rows)
    assert any(len(description.strip()) <= 20 for description, _ in rows)
    assert any(len(description.strip()) > 20 for description, _ in rows)
    assert len({categoria for _, categoria in rows}) > 5
