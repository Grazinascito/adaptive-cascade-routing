# Adaptive Cascade Routing — Plano de Tarefas e Status

Atualizado em 2026-11-14. Estado dos PRs verificado no GitHub em 2026-11-13. Em 2026-11-14 foi decidido o **pivot de dataset** (ASSIN 2 → NCM): a tese do projeto exige um dataset com cauda ambígua (o tipo de conteúdo do blog do iFood), que o ASSIN 2 não tem — provado pela medição do Marco 1 (S1 puro 0,920 > S2 puro 0,865). O ASSIN 2 sai da linha principal do relatório; a medição do Marco 1 permanece no histórico do git (commit `79a8654`). Detalhes na task 6 abaixo.

Este documento lista todas as tarefas do projeto: as concluídas, a tarefa em andamento e as planejadas. Cada tarefa traz descrição, links do PR, do Colab (quando existem) e da especificação.

## Termos usados neste documento

- **Cascata adaptativa** (adaptive cascade): roteamento que envia a maioria das requisições para um caminho rápido e barato (System 1) e só as requisições ambíguas para um caminho lento e caro (System 2).
- **System 1**: caminho rápido. Decide com um único token e recebe do servidor a probabilidade de cada label.
- **System 2**: caminho lento. Faz raciocínio passo a passo (*Chain-of-Thought*, técnica em que o modelo escreve os passos de raciocínio antes da resposta) e devolve JSON validado.
- **Spec**: documento que especifica um componente antes da implementação. As specs deste repo ficam em `docs/`.
- **PR** (*pull request*): solicitação de merge de uma branch para a `main`.
- **α (alfa)**: limiar de confiança. Confiança acima de α fica no System 1; abaixo, vai para o System 2.
- **APGR** (*Average Performance Gap Recovered*): fração da diferença de acurácia entre System 1 puro e System 2 puro que a cascata recupera. Métrica definida pelo RouteLLM.
- **Marco** (*milestone*): grupo de tarefas com um objetivo comum, definido na seção "9. Rollout Plan" do design doc.
- **Prefix caching**: reuso do trecho idêntico do prompt entre requisições, que reduz o custo de processamento.

## Resumo

| Status | Quantidade |
|--------|------------|
| Concluídas | 4 |
| Em andamento | 1 |
| Planejadas | 6 |

---

## Tarefas concluídas

### 1. Fundação do repositório

- **Status**: ✅ Concluída. PR [#1](https://github.com/Grazinascito/adaptive-cascade-routing/pull/1) merged (branch `000-repo-foundation`, já deletada).
- **Descrição**: criou a estrutura inicial do projeto: gerenciador de pacotes uv, árvore `src`/`tests`, configuração centralizada em `config/`, CI no GitHub Actions (roda lint e testes em cada PR), glossário e o ADR 0001 (registro da decisão de separar as dependências do Colab em um arquivo próprio).
- **Spec**: não havia — a task é anterior às specs. A numeração de componentes começa no A.
- **Colab**: sem notebook nesta task.

### 2. Componente A — Dataset Pipeline & Prompt Formatter

- **Status**: ✅ Concluída. PR [#2](https://github.com/Grazinascito/adaptive-cascade-routing/pull/2) merged (branch `001-component-a-dataset-loader`, já deletada).
- **Descrição**: `src/data/loader.py` baixa o dataset ASSIN 2 (benchmark de inferência textual em português) do Hugging Face, converte os labels em classificação binária (`Entailment` vs `Contradiction`), formata os prompts com prefixo comum para o vLLM ativar prefix caching e amostra o conjunto de teste congelado com seed fixa.
- **Spec**: [`docs/dataset-loader-spec.md`](dataset-loader-spec.md), Stories 1–3.
- **Colab**: o notebook cobre o pipeline inteiro. Link do estado na `main`: [abrir no Colab](https://colab.research.google.com/github/Grazinascito/adaptive-cascade-routing/blob/main/notebook/adaptive_cascade_benchmark.ipynb).

### 3. Componentes D + E — Confidence Evaluator e Cascade Router

- **Status**: ✅ Concluída. PR [#3](https://github.com/Grazinascito/adaptive-cascade-routing/pull/3) merged (branch `002-component-d-e-engine-router`, já deletada).
- **Descrição**: `src/cascade/router.py` implementa os dois componentes. O Componente D calcula softmax, confiança top-1 e entropia de Shannon sobre os *logprobs* (probabilidades por token que o servidor devolve) das opções de resposta. O Componente E compara a confiança contra o limiar α e envia ao System 2 apenas as requisições ambíguas.
- **Spec**: [`docs/engine-router-spec.md`](engine-router-spec.md), Stories 3–4.
- **Colab**: mesma nota do Componente A — [abrir no Colab (main)](https://colab.research.google.com/github/Grazinascito/adaptive-cascade-routing/blob/main/notebook/adaptive_cascade_benchmark.ipynb).

### 4. Componentes B, C + F — Infraestrutura vLLM e engines

- **Status**: ✅ Concluída. PR [#4](https://github.com/Grazinascito/adaptive-cascade-routing/pull/4) merged (branch `003-component-b-c-f-engines`, já deletada).
- **Descrição**: três componentes em um PR. B (`vllm_manager.py`) sobe e monitora o servidor vLLM com prefix caching. C (`system1.py`) faz a decisão tipada de 1 token via `/v1/systemone` e recebe a probabilidade de cada label. F (`system2.py`) executa o raciocínio passo a passo com saída JSON validada.
- **Spec**: [`docs/engine-router-spec.md`](engine-router-spec.md), Stories 1–2 (B e C) + [`docs/system2-evaluator-spec.md`](system2-evaluator-spec.md), Story 1 (F).
- **Colab**: mesma nota do Componente A — [abrir no Colab (main)](https://colab.research.google.com/github/Grazinascito/adaptive-cascade-routing/blob/main/notebook/adaptive_cascade_benchmark.ipynb).

---

## Tarefa em andamento

### 5. Componente G — Benchmark & Performance Evaluation Harness

- **Status**: 🔄 Em andamento. PR [#5](https://github.com/Grazinascito/adaptive-cascade-routing/pull/5) aberto, CI verde, branch `004-component-g-evaluator`.
- **Descrição**: `src/evaluation/evaluator.py` executa o benchmark nos 3 cenários (System 1 puro, System 2 puro, cascata adaptativa), mede latência p50/p95/p99, conta tokens de prefill/decode, calcula acurácia e F1, computa o APGR e varre valores de α para responder qual limiar maximiza a acurácia mantendo a invocação do System 2 abaixo de 15% das requisições.
- **Spec**: [`docs/system2-evaluator-spec.md`](system2-evaluator-spec.md), Stories 2–3.
- **Colab**: as células G1–G4 estão apenas na branch 004. Use o [link da branch 004 no Colab](https://colab.research.google.com/github/Grazinascito/adaptive-cascade-routing/blob/004-component-g-evaluator/notebook/adaptive_cascade_benchmark.ipynb). O link da `main` não contém essas células.
- **O que falta para concluir**:
  1. ~~Executar as células G1–G4 no Colab~~ ✅ Feito em 2026-11-14 (gates conferidos; SC-003/APGR FAIL honesto — S2 puro 0,865 < S1 puro 0,920).
  2. ~~Colar os números medidos no corpo do PR #5~~ ✅ Feito (seção "Colab validation status").
  3. ~~Commitar os artefatos reais da execução em `results/`~~ ✅ Feito (commit `79a8654`, CI verde).
  4. Review com comentários inline no GitHub (se pedida), merge e passos de pós-merge.
  5. Após o merge: iniciar a task 6 (pivot de dataset NCM).

---

## Tarefas planejadas

Fonte: seção "9. Rollout Plan" do design doc ([`docs/adaptive-cascade-design-doc.md`](adaptive-cascade-design-doc.md)). Nenhuma delas tem branch criada, spec escrita ou PR aberto hoje.

### 6. Pivot de dataset — classificação NCM de itens de catálogo (branch `005-dataset-pivot-ncm`)

Troca o ASSIN 2 pela tarefa de classificação de itens de catálogo em códigos **NCM** (*Nomenclatura Comum do Mercosul*: a taxonomia fiscal brasileira oficial em árvore — capítulo 2 dígitos → posição 4 → subposição 6 → item 8), usando o dataset público [`yurifacanha/NCM-RAFT-samall`](https://huggingface.co/datasets/yurifacanha/NCM-RAFT-samall) (105k linhas, PT-BR, nomes reais de itens: "ALCOOL LIQUIDO START NAT 70% - 5VL" → 22071090). É o tipo de conteúdo do blog do iFood: nome curto e ambíguo → caminho numa taxonomia em árvore, com cauda (*long tail*) real. A máquina do Marco 1 (harness, roteador por confiança, sweep de α por replay) é reusada sem mudança de desenho; mudam o loader (parser + amostragem cabeça/cauda congelada de 800 itens), os prompts (saída = código NCM) e o mecanismo do System 1 (geração curta de 8–12 tokens com confiança pelos logprobs, no lugar da pergunta `choice` binária — decisão registrada na spec). Gates novos: gap S2−S1 ≥ 10 pp na cauda, cascata recupera ≥ 50% do gap, redução de tokens ≥ 60%, taxa de fallback comparada à banda de 30–50% do blog. Inclui a reescrita de README e specs e a remoção do ASSIN 2 da linha principal do relatório. Execução estimada: ~25–45 min de Colab.

- **Spec**: [`docs/dataset-pivot-ncm-spec.md`](dataset-pivot-ncm-spec.md).
- **Dependência**: roda após o merge do PR #5.

### 7. Marco 4 — Componente H: juiz LLM (branch `006-judge-audit`)

Um **juiz LLM** (*LLM-as-a-Judge*: modelo que avalia a resposta de outro modelo — análogo de laboratório do Revisor do ciclo do iFood) audita uma amostra (taxa definida no config) dos registros de `results/raw_benchmark_data.json`, o que dispensa re-rodar o benchmark. O juiz verifica a predição da cascata contra o nome do item e a descrição do capítulo do código NCM correto. Quando discorda, dispara uma tentativa de reclassificação com o feedback do juiz no prompt. Métricas: taxa de concordância e efeito da reclassificação. Spec a criar em `docs/judge-spec.md`. *(Era "Marco 3, task 6" antes do pivot de 2026-11-14; renumerada.)*

### 8. Marco 4 — Componente I: loop de insights (branch `007-insights-loop`)

A etapa *aprender* do ciclo. As discordâncias do juiz viram insights estruturados em `results/insights.json`. Os principais insights são injetados no prompt do System 2, e o delta de acurácia com/sem insights é medido no mesmo conjunto congelado. Esta task fecha o ciclo: classificar → auditar → aprender → melhorar.

### 9. Marco 4 — HITL: revisão humana (branch `008-hitl-review`, opcional)

**HITL** (*human in the loop*): os casos não resolvidos (juiz discorda e a reclassificação falha) são exportados em um CSV de revisão humana. As correções humanas voltam ao sistema como insights. Marcada como opcional no rollout plan.

### 10. Marco 5 — Hugging Face Space (Gradio)

Publicar um Space interativo com **Gradio** (biblioteca Python para construir demos web de modelos) permitindo comparar em tempo real a latência entre System 1 e System 2.

### 11. Marco 6 — Artigo técnico no LinkedIn

Publicar o estudo de caso com os números medidos no dataset NCM (gap de acurácia na cauda, recuperação da cascata, redução de tokens/latência e taxa de fallback), fundamentado nas referências do iFood e do RouteLLM citadas na seção "11. References" do design doc.

### Fora de escopo deliberado

O análogo do Pepper (terceiro agente do ciclo do iFood, que minera o catálogo sem ground truth) não será implementado: o dataset NCM tem ground truth (o código anotado), e a mineração do subconjunto de erros da cascata já fica coberta pelo loop de insights do Componente I.

---

## Sobre os links do Colab

O projeto tem um único notebook, `notebook/adaptive_cascade_benchmark.ipynb`, que cobre o pipeline inteiro. Cada PR não teve um Colab próprio. O Colab carrega o notebook direto do GitHub, e o link aponta para uma branch. Quando a branch é deletada após o merge, o link morre — por isso os links das tarefas concluídas apontam para a `main`, que contém todo o código mesclado.

| Estado do notebook | Link |
|--------------------|------|
| `main` (tudo mesclado; sem as células G) | [abrir no Colab](https://colab.research.google.com/github/Grazinascito/adaptive-cascade-routing/blob/main/notebook/adaptive_cascade_benchmark.ipynb) |
| branch `004-component-g-evaluator` (inclui células G1–G4) | [abrir no Colab](https://colab.research.google.com/github/Grazinascito/adaptive-cascade-routing/blob/004-component-g-evaluator/notebook/adaptive_cascade_benchmark.ipynb) |

---

## Fontes

- Estado dos PRs: `gh pr list` no repo `Grazinascito/adaptive-cascade-routing`, consultado em 2026-11-13.
- Pivot de dataset (2026-11-14): decisão registrada em conversa com o autor do repo; design detalhado em [`docs/dataset-pivot-ncm-spec.md`](dataset-pivot-ncm-spec.md); dataset verificado no Hugging Face (`yurifacanha/NCM-RAFT-samall`, 105.336 linhas; alternativa `yurifacanha/NCM-dataset`, 239.226 linhas).
- Tarefas planejadas: seção "9. Rollout Plan" de [`docs/adaptive-cascade-design-doc.md`](adaptive-cascade-design-doc.md).
- Descrições por componente: [`docs/dataset-loader-spec.md`](dataset-loader-spec.md), [`docs/engine-router-spec.md`](engine-router-spec.md) e [`docs/system2-evaluator-spec.md`](system2-evaluator-spec.md).
