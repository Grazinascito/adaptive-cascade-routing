# Feature Branch: `005-dataset-pivot-ncm` (rewrite dos Componentes B–E; a versão ASSIN 2 está no histórico, commit `79a8654`)

**Created**: 2026-11-14 (pivot)
**Input**: Stories 1–4 de `docs/dataset-pivot-ncm-spec.md` (Decisão 1(d): System 1 em geração curta)

---

## Component Overview

- **Componente B (§5.2): vLLM Serving Infrastructure (`vLLMServerManager`)**: inalterado do Marco 1 — sobe e monitora o servidor local com `--enable-prefix-caching`.
- **Componente C (§5.3): System 1 Short Classification Engine (`system1.py`)**: `/v1/chat/completions` com `max_tokens=12`, `temperature=0.0`, `logprobs=True`, `top_logprobs=20` e `enable_thinking=False`; produz o código NCM de 8 dígitos sem raciocínio. Confiança = média geométrica das probabilidades dos tokens gerados; código inexistente na taxonomia (ou ausente) → confiança 0.0 (escala). Entropia = Shannon sobre os top-logprobs do primeiro token. A pergunta `choice` do `/v1/systemone` caiu com o pivot (114 classes não cabem em opções de 1 token).
- **Componente D (§5.4): Uncertainty Signals (`router.py`)**: pós-pivot, `shannon_entropy` (reusada pelo System 1) e a confiança geométrica calculada em `system1.py`; `softmax`/`compute_metrics` da era `choice` foram removidos por não terem mais chamador.
- **Componente E (§5.5): Adaptive Cascade Router (`router.py`)**: `route(query, confidence, alpha, system2)` inalterado na lógica (RouteLLM $R_\alpha$); `RoutingQuery` agora carrega `item_name` e o handler devolve código NCM (`str`).

---

## Stories & Testing *(mandatory)*

### Story 1 - vLLM Server Setup and Prefix Caching (Componente B, Type: enabler, Priority: P1)

Inalterada do Marco 1 (ver histórico). Independent Test: health-check + queda de prefill >= 70% nas queries 2–10 — agora medida sobre a geração curta do System 1 (célula de warmup do notebook).

### Story 2 - Short Classification com confiança por logprobs (Componente C, Type: functionality, Priority: P1)

**Why this priority**: é o fast-path do experimento; sem confiança calibrável não há roteador.

**Independent Test**: células C do notebook (sonda de granularidade) + testes com `httpx.MockTransport`.

**Acceptance Scenarios**:
1. **Given** resposta com código válido e logprobs conhecidos, **When** `evaluate_short_classification` roda, **Then** código e confiança batem com a média geométrica analítica.
2. **Given** código bem-formado inexistente na taxonomia, **Then** `valid_code=False` e confiança 0.0.
3. **Given** resposta sem nenhum código de 8 dígitos, **Then** código = sentinela `00000000`, confiança 0.0.
4. **Given** `item_name` em branco, **Then** `ValueError` ANTES de qualquer HTTP.

**Testes unitários** (`tests/engine/test_system1.py`): `test_returns_code_and_confidence_from_token_logprobs`, `test_confidence_is_geometric_mean_not_product_of_token_probabilities`, `test_confidence_is_zero_when_code_is_not_in_taxonomy`, `test_flags_fallback_when_output_has_no_valid_code`, `test_entropy_comes_from_first_token_top_logprobs_distribution`, `test_rejects_blank_item_name_before_any_http_call`, `test_system1_instructions_embed_the_chapter_list_from_the_taxonomy`.

### Story 3 - Sinais de incerteza (Componente D, Type: functionality, Priority: P2)

`shannon_entropy` em bits com convenção p=0 → 0; reusada pelo System 1 (primeiro token) — verificada contra distribuições de referência (uniforme = 1 bit; 90/10 ≈ 0.469 bits). Teste: `test_shannon_entropy_returns_known_values_for_reference_distributions`.

### Story 4 - Adaptive Cascade Routing (Componente E, Type: hypothesis, Priority: P2)

`route` inalterada do Marco 1 (fast-path em `confidence >= α`; repasse ao handler injetado; validação de intervalo [0, 1]). Testes: os 7 casos de `tests/engine/test_router.py` (boundary α=confiança, α∈{0, 1}, rejeições).

---

## Edge Cases

- **Código inexistente na taxonomia**: confiança 0.0 → 100% dos casos escalam ao System 2 em qualquer α > 0 (o "caso novo → fallback" do blog).
- **Bloco `<think>` do Qwen3**: desabilitado por request (`chat_template_kwargs`); a geração curta não pode gastar tokens de raciocínio.
- **vLLM Process Crash / OOM**: `vLLMServerException` com instruções de restart (inalterado).
- **Resposta do S1 sem código**: sentinela + flag; contada como erro na avaliação, nunca derruba o run.

## Requirements *(mandatory)*

- **FR-001**: servidor local vLLM com prefix caching (inalterado).
- **FR-002**: `evaluate_short_classification` DEVE enviar a instrução fixa (com a lista de capítulos) como mensagem `system` idêntica em toda query (prefix caching) e o nome do item como `user`.
- **FR-003**: confiança DEVE ser a média geométrica `exp(Σ ln p / n)` dos tokens gerados; ausência de logprobs → confiança 0.0.
- **FR-004**: entropia DEVE ser Shannon (bits) sobre a distribuição top-logprobs normalizada do primeiro token.
- **FR-005**: `route` DEVE implementar $R_\alpha(q)$ com fast-path em `confidence >= α`.
- **FR-006**: `BenchmarkResult` (Componente G) registra por requisição `query_id`, `subset`, latências por engine, `confidence`, `entropy`, `routed_to_system_2`, tokens prefill/decode SOMADOS dos engines invocados, `prediction`, `ground_truth` e flags de fallback (lista canônica na spec F–G).

## Success Criteria *(mandatory)*

- **SC-001**: geração do System 1 conclui em <= 12 tokens com código parseável em >= 95% dos itens da sonda.
- **SC-002**: queda de prefill >= 70% nas queries 2–10 do warmup (prefix caching sobre a instrução + capítulos).
- **SC-003**: zero códigos inválidos aceitos com confiança > 0 (todo código fora da taxonomia escala).

## Assumptions

- vLLM >= 0.32 com suporte a `logprobs`/`top_logprobs` no `/v1/chat/completions` (verificado na sonda antes de congelar a Decisão 1(d)).
- Qwen3-4B-AWQ conhece os códigos NCM frequentes do pré-treino a ponto de acurácia não-aleatória na cabeça — hipótese da sonda, não premissa.
