# Adaptive Cascade Routing — Plano de Tarefas e Status

Atualizado em 2026-11-15. Estado dos PRs verificado no GitHub. Em 2026-11-14 foi decidido o **pivot de dataset** (ASSIN 2 → NCM): a tese do projeto exige um dataset com cauda ambígua (o tipo de conteúdo do blog do iFood), que o ASSIN 2 não tem — provado pela medição do Marco 1 (S1 puro 0,920 > S2 puro 0,865). O ASSIN 2 saiu da linha principal do relatório; a medição do Marco 1 permanece no histórico do git (commit `79a8654`). Em 2026-11-15 foi decidido o **reboot das docs**: o design doc inicial foi removido e o README foi reescrito do zero com o objetivo do projeto na frente (o roteiro de Marcos agora vive no README, seção "Estado e roteiro").

Este documento lista todas as tarefas do projeto: as concluídas, a tarefa em andamento e as planejadas. Cada tarefa traz descrição, links do PR, do Colab (quando existem) e da especificação.

## Termos usados neste documento

- **Cascata adaptativa** (adaptive cascade): roteamento que envia a maioria das requisições para um caminho rápido e barato (System 1) e só as requisições ambíguas para um caminho lento e caro (System 2).
- **System 1**: caminho rápido. Geração curta (`max_tokens=12`, sem raciocínio) do código NCM; confiança pela média geométrica dos logprobs dos tokens gerados.
- **System 2**: caminho lento. Raciocínio passo a passo (*Chain-of-Thought*, técnica em que o modelo escreve os passos de raciocínio antes da resposta) e saída JSON validada.
- **NCM** (*Nomenclatura Comum do Mercosul*): taxonomia fiscal brasileira em árvore; a tarefa atual é nome de item de catálogo → código de 8 dígitos.
- **Spec**: documento que especifica um componente antes da implementação. As specs deste repo ficam em `docs/`.
- **PR** (*pull request*): solicitação de merge de uma branch para a `main`.
- **α (alfa)**: limiar de confiança. Confiança acima de α fica no System 1; abaixo, vai para o System 2.
- **APGR** (*Average Performance Gap Recovered*): fração da diferença de acurácia entre System 1 puro e System 2 puro que a cascata recupera. Métrica definida pelo RouteLLM; informativa pós-pivot.
- **Marco** (*milestone*): grupo de tarefas com um objetivo comum, definido na seção "Estado e roteiro" do [README](../README.md).
- **Prefix caching**: reuso do trecho idêntico do prompt entre requisições, que reduz o custo de processamento.

## Resumo

| Status | Quantidade |
|--------|------------|
| Concluídas | 5 |
| Em andamento | 1 |
| Planejadas | 5 |

---

## Tarefas concluídas

### 1. Fundação do repositório

- **Status**: ✅ Concluída. PR [#1](https://github.com/Grazinascito/adaptive-cascade-routing/pull/1) merged (branch `000-repo-foundation`, já deletada).
- **Descrição**: criou a estrutura inicial do projeto: gerenciador de pacotes uv, árvore `src`/`tests`, configuração centralizada em `config/`, CI no GitHub Actions (roda lint e testes em cada PR), glossário e o ADR 0001 (registro da decisão de separar as dependências do Colab em um arquivo próprio).
- **Spec**: não havia — a task é anterior às specs. A numeração de componentes começa no A.
- **Colab**: sem notebook nesta task.

### 2. Componente A — Dataset Pipeline & Prompt Formatter

- **Status**: ✅ Concluída. PR [#2](https://github.com/Grazinascito/adaptive-cascade-routing/pull/2) merged (branch `001-component-a-dataset-loader`, já deletada).
- **Descrição** (versão ASSIN 2, histórica): `src/data/loader.py` baixava o dataset ASSIN 2 (benchmark de inferência textual em português) do Hugging Face, convertia os labels em classificação binária, formatava os prompts com prefixo comum para o vLLM ativar prefix caching e amostrava o conjunto de teste congelado com seed fixa. Reescrito para NCM na task 6.
- **Spec**: [`docs/dataset-loader-spec.md`](dataset-loader-spec.md), Stories 1–3 (reescrita NCM).
- **Colab**: [abrir no Colab (main)](https://colab.research.google.com/github/Grazinascito/adaptive-cascade-routing/blob/main/notebook/adaptive_cascade_benchmark.ipynb) — a `main` ainda carrega a versão ASSIN 2 até o merge do PR #6.

### 3. Componentes D + E — Confidence Evaluator e Cascade Router

- **Status**: ✅ Concluída. PR [#3](https://github.com/Grazinascito/adaptive-cascade-routing/pull/3) merged (branch `002-component-d-e-engine-router`, já deletada).
- **Descrição** (versão ASSIN 2, histórica): o Componente D calculava softmax, confiança top-1 e entropia de Shannon sobre os logprobs das opções de resposta da pergunta `choice` do `/v1/systemone`; o Componente E comparava a confiança contra o limiar α e enviava ao System 2 apenas as requisições ambíguas. Pós-pivot: D ficou com a entropia de Shannon (reusada pelo System 1) e E mantém a lógica `R_α`.
- **Spec**: [`docs/engine-router-spec.md`](engine-router-spec.md), Stories 3–4.
- **Colab**: mesma nota do Componente A.

### 4. Componentes B, C + F — Infraestrutura vLLM e engines

- **Status**: ✅ Concluída. PR [#4](https://github.com/Grazinascito/adaptive-cascade-routing/pull/4) merged (branch `003-component-b-c-f-engines`, já deletada).
- **Descrição** (versão ASSIN 2, histórica): B (`server.py`) sobe e monitora o servidor vLLM com prefix caching. C (`system1.py`) fazia a decisão tipada de 1 token via `/v1/systemone`. F (`system2.py`) executa o raciocínio passo a passo com saída JSON validada. Pós-pivot, o C virou geração curta via `/v1/chat/completions` (task 6).
- **Spec**: [`docs/engine-router-spec.md`](engine-router-spec.md), Stories 1–2 (B e C) + [`docs/system2-evaluator-spec.md`](system2-evaluator-spec.md), Story 5 (F).
- **Colab**: mesma nota do Componente A.

### 5. Componente G — Benchmark & Performance Evaluation Harness

- **Status**: ✅ Concluída. PR [#5](https://github.com/Grazinascito/adaptive-cascade-routing/pull/5) merged em 2026-11-14 (merge commit `6f2188c`); branch `004` deletada.
- **Descrição**: `src/metrics/evaluator.py` + `src/metrics/summary.py` executam o benchmark nos 3 cenários (System 1 puro, System 2 puro, cascata adaptativa), medem latência p50/p95/p99, contam tokens de prefill/decode, calculam acurácia e F1, computam o APGR e varrem α por replay. Validação Colab executada em 2026-11-14 (dataset ASSIN 2): p95 −98,3% e tokens −99,6% PASS; SC-003/APGR FAIL honesto (S2 puro 0,865 < S1 puro 0,920 — o achado que motivou o pivot da task 6). Artefatos medidos commitados (`79a8654`).
- **Spec**: [`docs/system2-evaluator-spec.md`](system2-evaluator-spec.md) (versão ASSIN 2, histórica).
- **Colab**: validação feita pela branch 004 (deletada após o merge); artefatos preservados no commit `79a8654`.

---

## Tarefa em andamento

### 6. Pivot de dataset — classificação NCM de itens de catálogo (branch `005-dataset-pivot-ncm`)

- **Status**: 🔄 Em andamento. PR [#6](https://github.com/Grazinascito/adaptive-cascade-routing/pull/6) aberto em 2026-11-14, CI verde (75 testes, ruff no repo inteiro, mypy).

Troca o ASSIN 2 pela tarefa de classificação de itens de catálogo em códigos **NCM**, usando o dataset público [`yurifacanha/NCM-RAFT-samall`](https://huggingface.co/datasets/yurifacanha/NCM-RAFT-samall) (105k linhas, PT-BR, nomes reais de itens: "ALCOOL LIQUIDO START NAT 70% - 5VL" → `22071090`). É o tipo de conteúdo do blog do iFood: nome curto e ambíguo → caminho numa taxonomia em árvore, com cauda (*long tail*) real. A máquina do Marco 1 (harness, roteador por confiança, sweep de α por replay) é reusada sem mudança de desenho; mudam o loader (parser + amostragem cabeça/cauda congelada de 800 itens), os prompts (saída = código NCM) e o mecanismo do System 1 (geração curta de 8–12 tokens com confiança pelos logprobs, no lugar da pergunta `choice` binária — decisão registrada na spec). Gates novos: gap S2−S1 ≥ 10 pp na cauda, cascata recupera ≥ 50% do gap, redução de tokens ≥ 60%, taxa de fallback comparada à banda de 30–50% do blog. Incluiu o reboot das docs (2026-11-15): README reescrito do zero com o objetivo na frente, design doc inicial removido (histórico no git), specs de componentes autônomas e glossário corrigido.

- **Spec**: [`docs/dataset-pivot-ncm-spec.md`](dataset-pivot-ncm-spec.md).
- **Sonda do dataset (medida em 2026-11-14, antes do parser)**: extração 0,9968 (gate ≥ 0,98 PASS); 69.499 pares únicos limpos; 114 códigos em 19 capítulos (69 deles com ≥ 50 nomes únicos — a cabeça); 60 nomes conflitantes excluídos (0,09%).
- **Colab**: use o [link da branch 005 no Colab](https://colab.research.google.com/github/Grazinascito/adaptive-cascade-routing/blob/005-dataset-pivot-ncm/notebook/adaptive_cascade_benchmark.ipynb). A célula de sonda carrega o pool, imprime a estatística e congela a granularidade (`LABEL_DIGITS`); depois vêm B, C, F e o benchmark G (~25–45 min).
- **O que falta para concluir**:
  1. Rodar o notebook na T4 pela branch 005 (sonda → dry-run → benchmark completo) e conferir os gates impressos (SC-A/B/C + gate F + anti-degeneração).
  2. Anexar o `.ipynb` com os outputs salvos para auditoria.
  3. Commitar os artefatos reais da execução em `results/`.
  4. Colar os números medidos no corpo do PR #6.
  5. Review (se pedida), merge e pós-merge (BRANCH e badge do Colab → "main", deletar a branch 005).

---

## Tarefas planejadas

Fonte: seção "Estado e roteiro" do [README](../README.md). Nenhuma delas tem branch criada, spec escrita ou PR aberto hoje.

### 7. Marco 4 — Componente H: juiz LLM (branch `006-judge-audit`)

Um **juiz LLM** (*LLM-as-a-Judge*: modelo que avalia a resposta de outro modelo — análogo de laboratório do Revisor do ciclo do iFood) audita uma amostra (taxa definida no config) dos registros de `results/raw_benchmark_data.json`, o que dispensa re-rodar o benchmark. O juiz verifica a predição da cascata contra o nome do item e a descrição do capítulo do código NCM correto. Quando discorda, dispara uma tentativa de reclassificação com o feedback do juiz no prompt. Métricas: taxa de concordância e efeito da reclassificação. Spec a criar em `docs/judge-spec.md`. *(Era "Marco 3, task 6" antes do pivot de 2026-11-14; renumerada.)*

### 8. Marco 4 — Componente I: loop de insights (branch `007-insights-loop`)

A etapa *aprender* do ciclo. As discordâncias do juiz viram insights estruturados em `results/insights.json`. Os principais insights são injetados no prompt do System 2, e o delta de acurácia com/sem insights é medido no mesmo conjunto congelado. Esta task fecha o ciclo: classificar → auditar → aprender → melhorar.

### 9. Marco 4 — HITL: revisão humana (branch `008-hitl-review`, opcional)

**HITL** (*human in the loop*): os casos não resolvidos (juiz discorda e a reclassificação falha) são exportados em um CSV de revisão humana. As correções humanas voltam ao sistema como insights. Marcada como opcional no roteiro do README.

### 10. Marco 5 — Hugging Face Space (Gradio)

Publicar um Space interativo com **Gradio** (biblioteca Python para construir demos web de modelos) permitindo comparar em tempo real a latência entre System 1 e System 2.

### 11. Marco 6 — Artigo técnico no LinkedIn

Publicar o estudo de caso com os números medidos no dataset NCM (gap de acurácia na cauda, recuperação da cascata, redução de tokens/latência e taxa de fallback), fundamentado nas referências do iFood e do RouteLLM citadas na seção "Referências" do [README](../README.md).

### Fora de escopo deliberado

O análogo do Pepper (terceiro agente do ciclo do iFood, que minera o catálogo sem ground truth) não será implementado: o dataset NCM tem ground truth (o código anotado), e a mineração do subconjunto de erros da cascata já fica coberta pelo loop de insights do Componente I.

---

## Sobre os links do Colab

O projeto tem um único notebook, `notebook/adaptive_cascade_benchmark.ipynb`, que cobre o pipeline inteiro. Cada PR não teve um Colab próprio. O Colab carrega o notebook direto do GitHub, e o link aponta para uma branch. Quando a branch é deletada após o merge, o link morre — por isso os links das tarefas concluídas apontam para a `main`.

| Estado do notebook | Link |
|--------------------|------|
| `main` (Marco 1/2: versão ASSIN 2 mesclada; sem as células NCM) | [abrir no Colab](https://colab.research.google.com/github/Grazinascito/adaptive-cascade-routing/blob/main/notebook/adaptive_cascade_benchmark.ipynb) |
| branch `005-dataset-pivot-ncm` (pivot NCM: sonda + células B/C/F/G reescritas) | [abrir no Colab](https://colab.research.google.com/github/Grazinascito/adaptive-cascade-routing/blob/005-dataset-pivot-ncm/notebook/adaptive_cascade_benchmark.ipynb) |

---

## Fontes

- Estado dos PRs: `gh pr list` no repo `Grazinascito/adaptive-cascade-routing`.
- Pivot de dataset (2026-11-14): decisão registrada em conversa com o autor do repo; design detalhado em [`docs/dataset-pivot-ncm-spec.md`](dataset-pivot-ncm-spec.md); dataset verificado no Hugging Face (`yurifacanha/NCM-RAFT-samall`, 105.336 linhas; alternativa `yurifacanha/NCM-dataset`, 239.226 linhas).
- Reboot das docs (2026-11-15): decisão do autor do repo; design doc inicial removido, README reescrito do zero.
- Roteiro de Marcos: seção "Estado e roteiro" do [README](../README.md).
- Descrições por componente: [`docs/dataset-loader-spec.md`](dataset-loader-spec.md), [`docs/engine-router-spec.md`](engine-router-spec.md) e [`docs/system2-evaluator-spec.md`](system2-evaluator-spec.md).
