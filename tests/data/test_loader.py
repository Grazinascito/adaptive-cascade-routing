"""Testes do loader NCM: parser, taxonomia e amostragem congelada head/tail.

Sem rede: o parser roda sobre stubs de conversa no formato real do dataset
(yurifacanha/NCM-RAFT-samall, colunas conversation_id/conversations com
papel 'user'/'assistant'); a taxonomia lê o CSV commitado
config/ncm_taxonomy.csv; a amostragem roda sobre itens sintéticos.
"""

import pytest

from src.data import loader


def _conversation(name: str, code: str, description: str = "Bebidas, líquidos alcoólicos.") -> list:
    return [
        {
            "role": "user",
            "content": (
                " contexto: [o codigo da categoria do "
                f"{name} é: {code}],"
                f"[a categoria {code} possui a seguinte descrição: {description}]"
            ),
        },
        {"role": "assistant", "content": f"o produto {name} possui categoria: {description}"},
    ]


class TestParseConversation:
    def test_parses_conversation_into_name_and_code(self) -> None:
        item = loader.parse_conversation(_conversation("ALCOOL LIQUIDO START NAT 70%", "22071090"))
        assert item is not None
        assert item["name"] == "ALCOOL LIQUIDO START NAT 70%"
        assert item["ncm_code"] == "22071090"

    def test_reports_none_when_answer_has_no_eight_digit_code(self) -> None:
        broken = [
            {"role": "user", "content": " contexto sem o par canonico de codigo"},
        ]
        assert loader.parse_conversation(broken) is None

    def test_reports_none_for_empty_conversation(self) -> None:
        assert loader.parse_conversation([]) is None

    def test_drops_names_shorter_than_minimum_length(self) -> None:
        item = loader.parse_conversation(_conversation("AB", "22071090"))
        assert item is None


class TestTaxonomy:
    def test_is_valid_code_accepts_known_and_rejects_unknown_codes(self) -> None:
        taxonomy = loader.load_taxonomy()
        assert loader.is_valid_code("22071090", taxonomy) is True
        assert loader.is_valid_code("99999999", taxonomy) is False
        assert loader.is_valid_code("00000000", taxonomy) is False

    def test_taxonomy_csv_has_unique_codes_and_no_blank_descriptions(self) -> None:
        taxonomy = loader.load_taxonomy()
        assert len(taxonomy) > 0
        assert len({entry["code"] for entry in taxonomy.values()}) == len(taxonomy)
        assert all(entry["description"].strip() for entry in taxonomy.values())

    def test_chapter_lines_list_every_chapter_once(self) -> None:
        lines = loader.chapter_lines()
        assert len(lines) == len({line.split(" - ")[0] for line in lines})
        assert all(line[:2].isdigit() for line in lines)


class TestPoolStats:
    def test_extraction_success_rate_counts_malformed_rows(self) -> None:
        items = [
            loader.NCMItem(item_id=i, name=f"ITEM {i} XYZ", ncm_code="22071090") for i in range(970)
        ]
        stats = loader.pool_stats(items, n_rows=1000)
        assert stats["n_rows"] == 1000
        assert stats["n_parsed"] == 970
        assert stats["extraction_rate"] == pytest.approx(0.97)
        assert stats["unique_codes"] == 1

    def test_conflicting_names_are_counted(self) -> None:
        items = [
            loader.NCMItem(item_id=0, name="AGUA MINERAL", ncm_code="22011000"),
            loader.NCMItem(item_id=1, name="AGUA MINERAL", ncm_code="22019000"),
            loader.NCMItem(item_id=2, name="REFRIGERANTE COLA", ncm_code="22021000"),
        ]
        stats = loader.pool_stats(items, n_rows=3)
        assert stats["conflicting_names"] == 1
        assert stats["unique_names"] == 2


def _pool_with_codes(codes_by_name: dict[str, str]) -> list[loader.NCMItem]:
    return [
        loader.NCMItem(item_id=index, name=name, ncm_code=code)
        for index, (name, code) in enumerate(codes_by_name.items())
    ]


class TestFrozenEvalSet:
    def _pool(self) -> list[loader.NCMItem]:
        pool: dict[str, str] = {}
        # 1 código head (60 itens) + 2 códigos tail (3 itens cada).
        for i in range(60):
            pool[f"ITEM HEAD {i} AAA"] = "22071090"
        for i in range(3):
            pool[f"ITEM TAIL UM {i} BBB"] = "33072010"
            pool[f"ITEM TAIL DOIS {i} CCC"] = "96190000"
        return _pool_with_codes(pool)

    def test_frozen_eval_set_is_deterministic_with_seed_42(self) -> None:
        first = loader.get_frozen_eval_set(self._pool(), sample_limit=10, head_share=0.6)
        second = loader.get_frozen_eval_set(self._pool(), sample_limit=10, head_share=0.6)
        assert first == second

    def test_head_tail_proportions_match_configured_target(self) -> None:
        samples = loader.get_frozen_eval_set(self._pool(), sample_limit=10, head_share=0.6)
        heads = [sample for sample in samples if sample["subset"] == "head"]
        tails = [sample for sample in samples if sample["subset"] == "tail"]
        assert len(heads) == 6
        assert len(tails) == 4
        assert all(sample["ncm_code"] == "22071090" for sample in heads)
        assert all(sample["ncm_code"] != "22071090" for sample in tails)

    def test_conflicting_names_are_excluded_and_counted(self) -> None:
        pool = self._pool()
        # Mesmo nome em dois códigos: conflito → fora da amostra.
        pool.append(loader.NCMItem(item_id=999, name="ITEM TAIL UM 0 BBB", ncm_code="22042100"))
        samples = loader.get_frozen_eval_set(pool, sample_limit=10, head_share=0.6)
        conflicting = [sample for sample in samples if sample["name"] == "ITEM TAIL UM 0 BBB"]
        assert conflicting == []

    def test_raises_when_tail_pool_is_smaller_than_requested_share(self) -> None:
        with pytest.raises(ValueError, match="tail pool"):
            loader.get_frozen_eval_set(self._pool(), sample_limit=10, head_share=0.1)

    def test_rejects_invalid_parameters(self) -> None:
        pool = self._pool()
        with pytest.raises(ValueError, match="sample_limit"):
            loader.get_frozen_eval_set(pool, sample_limit=0)
        with pytest.raises(ValueError, match="head_share"):
            loader.get_frozen_eval_set(pool, sample_limit=10, head_share=1.0)


class TestTruncateCode:
    def test_truncates_to_each_supported_granularity(self) -> None:
        assert loader.truncate_code("22071090", 2) == "22"
        assert loader.truncate_code("22071090", 4) == "2207"
        assert loader.truncate_code("22071090", 6) == "220710"
        assert loader.truncate_code("22071090", 8) == "22071090"

    def test_rejects_unsupported_granularity(self) -> None:
        with pytest.raises(ValueError, match="digits"):
            loader.truncate_code("22071090", 5)
