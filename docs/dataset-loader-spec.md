# Spec: Componente A — Pipeline de Dataset NCM (`src/data/loader.py`)

**Feature Branch**: `005-dataset-pivot-ncm` (rewrite do loader; a versão ASSIN 2 está no histórico do git, commit `79a8654` e anteriores)
**Created**: 2026-11-14 (pivot)
**Input**: Stories 1–3 de `docs/dataset-pivot-ncm-spec.md`

---

## Stories & Testing *(mandatory)*

### Story 1 - Parser do dataset NCM-RAFT e estatística do pool (Type: ENABLER, Priority: P1)

Parsear as conversas de `yurifacanha/NCM-RAFT-samall` (105.336 linhas, split `train` único, colunas `conversation_id`/`conversations` com papéis `user`/`assistant`). O par canônico (nome, código de 8 dígitos) vive na PRIMEIRA mensagem: `o codigo da categoria do {nome} é: {código}` — o parse não depende da resposta do assistente (que às vezes traz código, às vezes só descrição). `pool_stats` reporta taxa de extração, pares únicos, códigos, nomes conflitantes e nomes curtos.

**Why this priority**: sem parser verificado todo o pivot é chute.

**Independent Test**: célula de sonda do notebook imprimindo `STATS`; gate taxa >= 0,98 (medido em 2026-11-14: 0,9968 — 336 falhas em 105.336).

**Acceptance Scenarios**:
1. **Given** uma conversa no formato real, **When** `parse_conversation` roda, **Then** devolve `NCMItem` com nome (>= 3 chars) e código de 8 dígitos.
2. **Given** 1.000 stubs com 3% corrompidos, **When** `pool_stats` roda, **Then** reporta taxa 0,97.

**Testes unitários** (`tests/data/test_loader.py`): `test_parses_conversation_into_name_and_code`, `test_reports_none_when_answer_has_no_eight_digit_code`, `test_reports_none_for_empty_conversation`, `test_drops_names_shorter_than_minimum_length`, `test_extraction_success_rate_counts_malformed_rows`, `test_conflicting_names_are_counted`.

### Story 2 - Taxonomia de apoio e validade de código (Type: ENABLER, Priority: P1)

`config/ncm_taxonomy.csv` (commitado): 114 códigos × (capítulo, título do capítulo, descrição completa), derivado das descrições embutidas no próprio dataset (descrição única por código — verificada em 2026-11-14). `is_valid_code` é o gatilho de escalada do System 1; `chapter_lines` alimenta o prefixo estático dos prompts (mantém o prefix caching).

**Acceptance Scenarios**:
1. `is_valid_code("22071090")` → `True`; `is_valid_code("99999999")` → `False`.
2. CSV com código duplicado, descrição vazia ou código não-8-dígitos levanta `ValueError` no load.

**Testes unitários**: `test_is_valid_code_accepts_known_and_rejects_unknown_codes`, `test_taxonomy_csv_has_unique_codes_and_no_blank_descriptions`, `test_chapter_lines_list_every_chapter_once`.

### Story 3 - Amostragem congelada head/tail (Type: HYPOTHESIS, Priority: P2)

`get_frozen_eval_set(items, ...)`: dedup de pares → exclusão de nomes conflitantes (mesmo nome, códigos diferentes; contados e logados) → frequência de cada código sobre pares únicos limpos → head = códigos com >= `head_min_code_freq` (50) nomes únicos → amostragem estratificada `round(sample_limit × head_share)` head + resto tail, `random.Random(seed)` com seed 42. Cada `NCMEvalSample` carrega `item_id` (linha original), `code_freq` e `subset`.

**Why this priority**: a composição cabeça/cauda é o que torna a tese testável.

**Independent Test**: duas chamadas com seed 42 produzem listas idênticas; proporções batem o alvo.

**Acceptance Scenarios**:
1. **Given** seed 42, **When** roda duas vezes, **Then** as listas são idênticas.
2. **Given** head_share 0.6 e sample_limit 10, **Then** 6 head + 4 tail, códigos head só no subset head.
3. **Given** pool tail insuficiente, **Then** `ValueError` orientando ajustar parâmetros.

**Testes unitários**: `test_frozen_eval_set_is_deterministic_with_seed_42`, `test_head_tail_proportions_match_configured_target`, `test_conflicting_names_are_excluded_and_counted`, `test_raises_when_tail_pool_is_smaller_than_requested_share`, `test_rejects_invalid_parameters`.

### Granularidade do rótulo

`truncate_code(code, digits)` com `digits ∈ {2, 4, 6, 8}` (capítulo/posição/subposição/item). O nível é congelado pela sonda do notebook (o mais fino com S1 em 0.30–0.70) e aplicado de forma idêntica a predições e ground truths na avaliação (evaluator). Testes: `test_truncates_to_each_supported_granularity`, `test_rejects_unsupported_granularity`.

---

## Requirements *(mandatory)*

- **FR-001**: `load_ncm_pool()` DEVE carregar o split `train` via `datasets` e retornar `(itens, total de linhas)`; falhas de rede viram `DatasetDownloadError`.
- **FR-002**: `parse_conversation` DEVE extrair (nome, código) da primeira mensagem; nome < 3 chars ou sem código de 8 dígitos → `None` (contado como falha de extração).
- **FR-003**: `load_taxonomy` DEVE validar o CSV (8 dígitos, sem duplicatas, descrição não vazia) e cachear; `chapter_lines` DEVE listar cada capítulo uma vez.
- **FR-004**: `get_frozen_eval_set` DEVE ser determinístico com seed fixa, excluir nomes conflitantes e particionar por frequência de código.

## Success Criteria *(mandatory)*

- **SC-001**: taxa de extração >= 0,98 no pool (gate da sonda).
- **SC-002**: determinismo bit-exato da amostragem com seed 42.
- **SC-003**: zero nomes conflitantes no conjunto congelado.

## Assumptions

- O dataset permanece público no Hugging Face sem token; é baixado em runtime e não redistribuído (procedência e risco de ruído declarados no README; mitigação = revisão humana do Story 3 da spec do pivot).
- Python 3.12 com `datasets` e `pandas` (o CSV é lido com a biblioteca padrão `csv`).
