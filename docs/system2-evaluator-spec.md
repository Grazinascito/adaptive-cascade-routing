# Feature Branch: `005-dataset-pivot-ncm` (rewrite dos Componentes F–G; a versão ASSIN 2 está no histórico, commit `79a8654`)

**Created**: 2026-11-14 (pivot)
**Input**: Stories 5–6 de `docs/dataset-pivot-ncm-spec.md`

---

## Component Overview

- **Componente F (§5.6): System 2 NCM (`system2.py`)**: mesmo mecanismo do Marco 1 (CoT + JSON via `/v1/chat/completions`, `max_tokens=256`, `enable_thinking=False`), com prompt reescrito: lista de capítulos como contexto da taxonomia + nome do item + JSON `{"reasoning", "ncm_code"}`. Cadeia de fallback JSON → regex `\b\d{8}\b` → sentinela `00000000`.
- **Componente G (§5.7): Benchmark Harness (`evaluator.py` + `summary.py`)**: 3 cenários sobre o conjunto congelado head/tail; `BenchmarkResult` ganha `subset`; tokens somam os engines invocados; gates SC-A/B/C; sweep de α por replay; gate anti-degeneração; taxa de fallback comparada à banda 30–50% do blog.

---

## Stories & Testing *(mandatory)*

### Story 5 - System 2 NCM: CoT + JSON (Type: functionality, Priority: P1)

**Why this priority**: o slow-path precisa responder na mesma tarefa e no mesmo espaço de rótulos do S1.

**Independent Test**: célula F do notebook (20 itens; gate JSON >= 95% + código existente na taxonomia).

**Acceptance Scenarios**:
1. **Given** um nome de item, **When** `execute_reasoning` roda, **Then** o prompt contém o nome, a lista de capítulos e o schema JSON; a saída traz `ncm_code` de 8 dígitos.
2. **Given** JSON válido dentro de cerca de markdown, **Then** parse sem fallback.
3. **Given** JSON inválido com código no texto, **Then** regex com `fallback_used=True` e warning.
4. **Given** output vazio/sem código, **Then** sentinela `00000000` com `fallback_used=True`.

**Testes unitários** (`tests/engine/test_system2.py`): `test_json_contains_valid_ncm_code_and_item_name`, `test_parses_fenced_json_without_fallback`, `test_falls_back_to_regex_when_json_is_malformed`, `test_defaults_to_sentinel_code_on_empty_output`, `test_falls_back_when_json_is_valid_but_code_is_malformed`, `test_defaults_tokens_to_zero_when_usage_is_absent`.

### Story 6 - Benchmark com subsets e gates do pivot (Type: hypothesis, Priority: P2)

**Why this priority**: é a medição que responde a tese.

**Independent Test**: dry-run G1 (12 amostras, todo `item_id` nos 3 cenários) + G2/G3/G4 no Colab.

**Acceptance Scenarios**:
1. **Given** o conjunto congelado, **When** `run_full_benchmark` roda, **Then** cada registro carrega `subset` head/tail e tokens somados dos engines invocados; predição e ground truth truncados por `label_digits` de forma idêntica.
2. **Given** erro HTTP do System 2 na cascata, **Then** predição do System 1 + `system2_error_fallback=True` (fallback gracioso).
3. **Given** as medições gravadas, **When** `sweep_alpha` roda, **Then** replay sem HTTP com `route()` real por α.
4. **Given** confiança do S1 >= 0.95 em > 95% dos itens, **Then** `confidence_degeneration_share` dispara o aviso anti-degeneração (edge case da spec do pivot).

**Testes unitários** (`tests/metrics/`): `test_run_full_benchmark_records_every_item_in_all_three_scenarios`, `test_cascade_escalation_sums_tokens_of_both_engines`, `test_label_digits_truncates_predictions_and_ground_truths_identically`, `test_build_result_keeps_already_truncated_predictions_as_is`, `test_system_two_http_error_falls_back_to_system1_prediction_and_logs`, `test_sweep_alpha_call_rate_is_monotonic_across_thresholds`, `test_sweep_alpha_replays_route_fast_path_on_confidence_equal_to_alpha`, `test_confidence_degeneration_share_counts_high_confidence_items`, `test_optimal_alpha_prefers_lowest_call_rate_among_winners`, `test_save_raw_results_writes_meta_scenarios_sweep_summaries_and_kpis`, `test_plot_tradeoff_curve_writes_non_empty_png`, `test_main_runs_benchmark_from_config_against_stub_server`, `test_compute_summary_metrics_reports_head_and_tail_accuracy_separately`, `test_evaluate_kpis_flags_each_pivot_gate`, `test_evaluate_kpis_fails_recovery_when_cascade_drops_below_system1_overall`, `test_evaluate_kpis_requires_tail_accuracy_in_every_summary`.

---

## Requirements *(mandatory)*

- **FR-001**: `execute_reasoning` DEVE manter a instrução fixa (capítulos) + item numa única mensagem `user`, CoT antes do JSON (contrato validado no Colab).
- **FR-002**: parser DEVE aceitar JSON com ou sem cerca de markdown e extrair `ncm_code` de 8 dígitos; validade na taxonomia é papel da avaliação.
- **FR-003**: harness DEVE rodar os 3 cenários no MESMO conjunto congelado (FR-003 do Marco 1, mantido).
- **FR-004**: latência via `time.perf_counter` na camada serving; warmup discard de 5 só nos percentis.
- **FR-005**: tokens prefill/decode SOMAM os engines invocados por requisição (fast-path também custa tokens).
- **FR-006**: `compute_summary_metrics` DEVE reportar acurácia geral + head + tail, macro F1 sobre os códigos presentes, taxa de chamadas ao S2 e APGR (informativo).
- **FR-007**: `evaluate_kpis(cascade, system1, system2, kpis)` DEVE avaliar SC-A (tail_gap_pp >= 10), SC-B (recovery >= 0.50 E cascata >= S1 geral), SC-C (p95 e tokens vs S2 puro).

## Success Criteria *(mandatory)*

- **SC-001**: json_success_rate >= 0.95 no cenário system2_pure (gate F, re-medição do Marco 1).
- **SC-002**: 100% dos registros com subset válido nos 3 cenários (dry-run G1).
- **SC-003**: gates SC-A/SC-B/SC-C impressos com PASS/FAIL honesto; FAIL de premissa (ex.: SC-A) é resultado, não erro de execução.

## Assumptions

- vLLM saudável no localhost; conjunto congelado carregado deterministicamente pelo loader (seed 42).
- `temperature=0.0` torna a saída do S2 determinística na sessão — premissa do sweep por replay (mantida do Marco 1).
