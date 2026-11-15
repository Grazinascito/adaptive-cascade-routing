# Spec: Pivot de dataset — Classificação NCM de itens de catálogo (branch `005-dataset-pivot-ncm`)

**Created**: 2026-11-14
**Status**: Draft
**Input**: Decisão de pivot registrada em conversa de 2026-11-14. Motivo: a tese do projeto (os ganhos da arquitetura híbrida do iFood também aparecem no eixo System 1 × System 2 do mesmo modelo) exige um dataset com cauda ambígua — o tipo de conteúdo que o blog do iFood descreve (nomes ambíguos, contexto incompleto, *long tail*). O ASSIN 2 não tem esse conteúdo, e a medição do Marco 1 provou isso (S1 puro 0,920 > S2 puro 0,865; cascata degenera para "sempre S1" com 99,5% do tráfego no caminho barato). O ASSIN 2 sai da linha principal do relatório; a medição do Marco 1 permanece no histórico do git (commit `79a8654`).

---

## Termos usados nesta spec

- **NCM** (*Nomenclatura Comum do Mercosul*): taxonomia fiscal brasileira oficial que classifica mercadorias. É uma árvore: capítulo (2 dígitos) → posição (4 dígitos) → subposição (6 dígitos) → item (8 dígitos). Ex.: `22071090` = capítulo 22 (bebidas), posição 2207 (álcool etílico).
- **Cabeça e cauda** (*head/tail*): cabeça = códigos frequentes no pool (itens fáceis, onde o S1 deve bastar); cauda = códigos raros (itens ambíguos, onde se espera que o S2 recupere qualidade). É o "long tail" do blog do iFood.
- **Pool**: o conjunto inteiro do dataset, usado para estatística e amostragem. O benchmark executa numa amostra, não no pool.
- **Sonda** (*probe*): célula curta de medição que roda antes do experimento completo para decidir um parâmetro de desenho com número na mão.
- **Gold label**: o rótulo correto oficial do item (aqui: o código NCM de 8 dígitos anotado no dataset).
- **Conflito**: mesmo nome de item aparecendo com códigos NCM diferentes no pool (indício de ruído de rótulo).
- **Geração curta**: chamada de chat com `max_tokens` pequeno (8–12), sem raciocínio, produzindo só o código.
- **Logprobs**: logaritmo da probabilidade de cada token gerado; base da confiança do S1 neste pivot.
- **Replay**: recomputar decisões da cascata a partir dos logs gravados, sem novas chamadas de API (técnica já usada no sweep de α do Marco 1).

---

## Decisões de desenho em aberto (com recomendação)

1. **Mecanismo do System 1 na tarefa multi-classe.** O `/v1/systemone` só aceita perguntas `choice` com poucas opções de 1 token; NCM tem milhares de códigos, então a pergunta binária do Marco 1 não se aplica.
   - **(d) Recomendado — geração curta com confiança por logprobs**: S1 vira uma chamada `/v1/chat/completions` com `max_tokens=12`, sem raciocínio, produzindo o código de 8 dígitos; a confiança vem dos logprobs dos tokens gerados; código inexistente na taxonomia → confiança 0 → escala pro S2 (espelha o "caso novo → fallback" do blog). Mantém a tarefa fiel à do iFood (escolher na árvore inteira) e o princípio do S1 (decisão estatística sem raciocínio). Custo: o S1 deixa de ser estritamente 1 token (passa a 8–12 tokens — ainda ~20× mais barato que o S2 de até 256).
   - **(a) Alternativa — manter `/v1/systemone` com múltipla escolha**: ~8 códigos candidatas por item (a correta + distratoras). Mantém o endpoint tipado do Marco 1, mas converte a tarefa em *multiple-choice* (o Sugar do iFood escolhe na árvore inteira) e exige um gerador de candidatas novo — mais maquinário e tarefa menos fiel.
2. **Granularidade do rótulo (8, 6 ou 4 dígitos).** Decidida pela sonda do Story 1, não por preferência: o nível onde o S1 puro fica entre 30% e 70% de acurácia é o nível com ambiguidade real e jogável para o par modelo+tarefa.

---

## Stories & Testing *(mandatory)*

### Story 1 — Sonda e parser do dataset NCM (Type: ENABLER, Priority: P1)

Carregar `yurifacanha/NCM-RAFT-samall` (105.336 linhas, split `train` único, formato conversacional) e parsear cada conversa para extrair `(product_name, ncm_code, chapter_description)`. Se a taxa de extração em 1.000 linhas for < 98%, trocar pela versão maior `yurifacanha/NCM-dataset` (239.226 linhas) e repetir o gate.

A sonda de granularidade roda S1 em geração curta sobre ~100 itens em cada nível (2, 4 e 8 dígitos) e registra a acurácia por nível. O nível congelado é o mais fino que mantém o S1 na faixa 0,30–0,70.

**Why this priority**: sem parser verificado e granularidade escolhida por medição, todo o resto do pivot é chute.

**Independent Test**: célula de sonda no Colab imprimindo taxa de extração, contagem de nomes únicos, códigos distintos, conflitos e acurácia S1 por nível de granularidade.

**Acceptance Scenarios**:
1. **Given** 1.000 conversas amostradas, **When** o parser roda, **Then** ≥ 98% produzem nome não vazio + código de 8 dígitos + descrição de capítulo.
2. **Given** as acurácias do S1 por nível, **When** o nível é escolhido, **Then** a escolha e os números ficam registrados na célula e no corpo do PR.

**Testes unitários** (`tests/test_ncm_loader.py`):
- `test_parses_conversation_into_name_code_and_chapter` — stub de conversa nos 2 formatos (RAFT-samall e NCM-dataset) → campos extraídos.
- `test_reports_zero_code_when_answer_has_no_eight_digit_code` — conversa sem código de 8 dígitos → item descartado e contado.
- `test_extraction_success_rate_counts_malformed_rows` — 1.000 stubs com 3% corrompidos → taxa 0,97 reportada.

---

### Story 2 — Taxonomia de apoio e validade de código (Type: ENABLER, Priority: P1)

Construir `config/ncm_taxonomy.csv` (código, descrição por nível) a partir da tabela pública oficial da NCM; se a fonte oficial falhar, derivar das descrições embutidas no próprio dataset. O prefixo estático dos prompts (instrução + lista compacta dos 96 capítulos) fica idêntico em toda query para manter o prefix caching.

**Why this priority**: o prompt precisa do contexto da taxonomia (o "mapa da árvore"), e a validade do código é o gatilho de escalada do S1.

**Independent Test**: validação de códigos conhecidos/desconhecidos contra a tabela.

**Acceptance Scenarios**:
1. **Given** o código `22071090`, **When** `is_valid_code()` é chamado, **Then** retorna `True` (existe na tabela).
2. **Given** o código `99999999`, **When** `is_valid_code()` é chamado, **Then** retorna `False`.

**Testes unitários** (`tests/test_ncm_taxonomy.py`):
- `test_is_valid_code_accepts_known_and_rejects_unknown_codes`
- `test_taxonomy_csv_has_unique_codes_and_no_blank_descriptions`

---

### Story 3 — Amostragem congelada cabeça/cauda com revisão humana (Type: HYPOTHESIS, Priority: P2)

Deduplicar nomes, excluir conflitos (mesmo nome, códigos diferentes — contados e reportados), particionar por frequência do código no pool e congelar o conjunto de avaliação: **800 itens, alvo ~65% cabeça / ~35% cauda** (proporção medida no pool e registrada), seed 42, hash do conjunto documentado. Antes do run completo, exportar CSV com 30 itens amostrados para revisão humana do gold label (papel HITL do ciclo do iFood; também mitiga o risco de ruído do dataset, que tem card vazio).

**Why this priority**: a composição cabeça/cauda é o que torna a tese testável; a revisão humana protege contra rótulo ruim.

**Independent Test**: duas chamadas com seed 42 produzem listas idênticas; proporções batem com o alvo.

**Acceptance Scenarios**:
1. **Given** seed 42, **When** `get_frozen_eval_set()` roda duas vezes, **Then** as listas e o hash são idênticos.
2. **Given** o pool parseado, **When** a amostragem roda, **Then** itens com nome conflitante ficam de fora e o total de excluídos é impresso.

**Testes unitários** (`tests/test_ncm_sampling.py`):
- `test_frozen_eval_set_is_deterministic_with_seed_42`
- `test_conflicting_names_are_excluded_and_counted`
- `test_head_tail_proportions_match_configured_target`

---

### Story 4 — System 1 em geração curta com confiança por logprobs (Type: FUNCTIONALITY, Priority: P1) *(pendente da Decisão 1)*

`/v1/chat/completions` com `max_tokens=12`, `temperature=0.0`, `logprobs` ativado; saída validada por regex `\d{8}`; confiança = média geométrica das probabilidades dos tokens gerados (`exp(Σ ln p / n)` — documentada na spec por ser menos punitiva que o produto puro em códigos de 8 dígitos); código inexistente na taxonomia → confiança 0,0 e escalamento imediato.

**Why this priority**: é o fast-path do experimento; sem confiança calibrável não há roteador.

**Independent Test**: chamadas mockadas (padrão `httpx.MockTransport` do repo) devolvendo logprobs sintéticos → código e confiança corretos.

**Acceptance Scenarios**:
1. **Given** uma resposta com tokens do código e logprobs conhecidos, **When** o S1 roda, **Then** o código extraído e a confiança batem com o valor analítico esperado.
2. **Given** uma resposta cujo código não existe na taxonomia, **When** o S1 roda, **Then** confiança = 0,0 e `routed_to_system_2` no resultado.

**Testes unitários** (`tests/test_system1_ncm.py`):
- `test_system1_returns_code_and_confidence_from_token_logprobs`
- `test_system1_confidence_is_zero_when_code_is_not_in_taxonomy`
- `test_system1_flags_fallback_when_output_has_no_valid_code`

---

### Story 5 — System 2 e prompts NCM (Type: FUNCTIONALITY, Priority: P1)

Mesmo mecanismo do Marco 1 (CoT + JSON via `/v1/chat/completions`, `max_tokens=256`), com o prompt reescrito para a tarefa: prefixo estático com a lista de capítulos, nome do item, e saída JSON `{"reasoning": "...", "ncm_code": "........"}`. Código inválido no JSON → *fallback* regex existente + flag.

**Why this priority**: o slow-path precisa responder na mesma tarefa e no mesmo espaço de rótulos do S1.

**Independent Test**: 20 itens via mock → ≥ 95% de JSON válido com código de 8 dígitos.

**Testes unitários** (`tests/test_system2_ncm.py`):
- `test_system2_json_contains_valid_ncm_code`
- `test_system2_falls_back_to_regex_when_json_is_malformed`

---

### Story 6 — Gates redefinidos e rerun do benchmark no NCM (Type: HYPOTHESIS, Priority: P2)

Rerun do Marco 1 sobre o conjunto NCM congelado, reusando o harness, o roteador e o sweep de α (replay) sem mudança de desenho. `BenchmarkResult` ganha o campo `subset` (`head` | `tail`). Novos critérios de sucesso:

- **SC-A (gap na cauda)**: `acc(S2 puro) − acc(S1 puro) ≥ 10 pp` no subconjunto cauda. Se falhar, a tese não se sustenta neste par modelo+tarefa e o FAIL é reportado com a mesma honestidade do Marco 1.
- **SC-B (recuperação)**: `acc(cascata) ≥ acc(S1 puro) + 50% do gap da cauda` e `acc(cascata) ≥ acc(S1 puro)` no conjunto completo.
- **SC-C (custo)**: decode tokens da cascata ≤ 40% do S2 puro (redução ≥ 60%, banda do blog) e p95 −50%.
- **Relatório obrigatório**: taxa de escalamento (fallback) no α ótimo, comparada à banda de 30–50% do blog do iFood.

Runtime esperado: ~25–45 min de Colab (800 chamadas S1 curtas + 800 chamadas S2 de até 256 tokens).

**Why this priority**: é a medição que responde a tese.

**Independent Test**: dry-run de 10 itens nos 3 cenários com artefatos válidos, como no Marco 1.

**Testes unitários** (`tests/test_evaluator_ncm.py`):
- `test_kpis_evaluate_new_gates_against_measured_values`
- `test_summary_reports_head_and_tail_accuracy_separately`

---

## Reescrita de docs incluída nesta task

- `README.md`: cenários, badge de dataset, KPIs e estrutura passam a descrever o NCM; o ASSIN 2 sai da linha principal com nota histórica de uma linha apontando para o commit `79a8654`. (A reescrita acontece junto com o código novo e os resultados medidos — reescrever antes criaria um README descrevendo um benchmark que ainda não existe.)
- `docs/adaptive-cascade-design-doc.md`: RF-1, §2, §5.1, §5.3 e §5.6 atualizados para a tarefa NCM; registro do pivot mantido.
- `docs/dataset-loader-spec.md`, `docs/engine-router-spec.md`, `docs/system2-evaluator-spec.md`: stories reescritas para a tarefa NCM (as notas de pivot no topo destes arquivos são removidas ao serem incorporadas).
- `CONTEXT.md`: verbete NCM adicionado ao glossário.

## Edge Cases

- **Nome de item muito curto ou vazio** (< 3 caracteres): descartado na amostragem, contado no relatório da sonda.
- **Código com menos de 8 dígitos na conversa**: item descartado pelo parser, contado.
- **Distribuição de confiança degenerada de novo** (lição do Marco 1: 199/200 itens com confiança ≥ 0,95): gate explícito antes do rerun completo — se > 95% dos itens do S1 tiverem confiança ≥ 0,95 na sonda de 100 itens, o par modelo+tarefa não tem faixa de ambiguidade; reportar e parar antes de gastar o run completo.
- **Dataset indisponível ou card vazio de licença**: o dataset é baixado em runtime e não é redistribuído no repo; a procedência (autoria única, 9 downloads, card vazio) é declarada no README com a mitigação (revisão humana do Story 3).

## Assumptions

- `yurifacanha/NCM-RAFT-samall` permanece público e acessível sem token no Hugging Face.
- O endpoint `/v1/chat/completions` do vLLM devolve `logprobs` dos tokens gerados (OpenAI-compatible) — verificar na sonda antes de congelar a Decisão 1.
- O modelo Qwen3-4B-AWQ tem conhecimento prévio suficiente dos códigos NCM frequentes (cabeça) para produzir acurácia não-aleatória no nível de granularidade escolhido — verificado pela sonda, não assumido.
