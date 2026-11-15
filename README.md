# Adaptive Cascade Routing — benchmark do padrão de cascata de inferência

[![CI](https://github.com/Grazinascito/adaptive-cascade-routing/actions/workflows/ci.yml/badge.svg)](https://github.com/Grazinascito/adaptive-cascade-routing/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Grazinascito/adaptive-cascade-routing/blob/005-dataset-pivot-ncm/notebook/adaptive_cascade_benchmark.ipynb)
[![Dataset](https://img.shields.io/badge/🤗_Dataset-NCM--RAFT--samall-FFD21F)](https://huggingface.co/datasets/yurifacanha/NCM-RAFT-samall)
[![Model](https://img.shields.io/badge/🤗_Model-Qwen3--4B--AWQ-FFD21F)](https://huggingface.co/Qwen/Qwen3-4B-AWQ)

## Objetivo

Este repositório mede uma pergunta específica de engenharia de inferência:

> **Os ganhos de custo do padrão de cascata de inferência — caminho barato primeiro, caminho caro sob demanda — também aparecem quando os dois níveis da cascata são dois modos do MESMO modelo?**

Contexto da pergunta:

- O blog de engenharia do iFood (*Tokens are money* e *Sugar, Revisor & Pepper*) reporta que o motor Sugar resolve 50–70% das classificações do catálogo com modelos baratos e escala apenas os casos ambíguos para modelos de alta capacidade, com redução de custo estimada em 60–80%.
- No iFood, os dois níveis da cascata são **modelos diferentes** (pequeno na frente, maior e mais capaz atrás). Neste benchmark de laboratório, os dois níveis são **modos do mesmo modelo** (Qwen3-4B-AWQ em um único servidor vLLM local): o System 1 gera a resposta em tokens poucos e sem raciocínio; o System 2 raciocina passo a passo.
- A formulação do roteador segue o RouteLLM (UC Berkeley / LMSYS): um limiar de confiança α decide entre aceitar o caminho barato ou escalar. Se o modo caro for genuinamente mais preciso, a cascata recupera qualidade a custo menor. Se não for, isso também é resultado — e o Marco 1 já mediu uma refutação dessa premissa (ver estado abaixo).

**O que este repositório NÃO é**: não é o sistema do iFood e não é uma réplica dele. Os artigos do iFood deram a pergunta de custo e o padrão "barato primeiro, caro sob demanda" (a mesma família do RouteLLM e do FrugalGPT); o critério de roteamento deles não é público. Este repositório é um benchmark controlado e reproduzível do padrão, com números medidos — inclusive os que derrubam a própria premissa.

## A tarefa e o dataset

Classificação de nome de item de catálogo brasileiro em código **NCM** (*Nomenclatura Comum do Mercosul* — a taxonomia fiscal brasileira em árvore: capítulo de 2 dígitos → posição de 4 → subposição de 6 → item de 8). Exemplo real do dataset: `"ALCOOL LIQUIDO START NAT 70% - 5VL"` → `22071090`.

- Dataset público [`yurifacanha/NCM-RAFT-samall`](https://huggingface.co/datasets/yurifacanha/NCM-RAFT-samall) (105.336 linhas, PT-BR, formato conversacional). Nomes curtos e ambíguos com cauda real (*long tail*) — o tipo de conteúdo que o blog do iFood descreve.
- Taxonomia derivada do próprio dataset e commitada: `config/ncm_taxonomy.csv` (114 códigos, 19 capítulos).
- Conjunto congelado de avaliação: 800 itens, ~65% cabeça (códigos frequentes) / ~35% cauda (códigos raros), seed 42. Sonda do pool (2026-11-14): extração 0,9968; 69.499 pares únicos; 60 nomes conflitantes excluídos (0,09%).
- Procedência declarada: dataset de autoria única, card vazio — risco de ruído de rótulo mitigado pela revisão humana da amostra (HITL) e pela exclusão de nomes conflitantes.

### Os 3 cenários comparados

1. **System 1 puro (fast-path)**: `/v1/chat/completions` com `max_tokens=12`, sem raciocínio — o modelo emite o código de 8 dígitos direto. Confiança = média geométrica das probabilidades dos tokens gerados; código fora da taxonomia → confiança 0.0 e escalamento imediato (o "caso novo → fallback" do blog).
2. **System 2 puro (slow-path)**: raciocínio *Chain-of-Thought* + JSON estrito (`max_tokens=256`), com a lista de capítulos como contexto da taxonomia.
3. **Cascata adaptativa**: aceita o System 1 quando a confiança ≥ α (default 0.85); escala para o System 2 apenas os itens estatisticamente ambíguos.

## Gates (critérios de sucesso do pivot)

- **SC-A — gap da cauda**: acurácia do S2 puro − S1 puro ≥ 10 pontos percentuais no subconjunto cauda (a premissa de que raciocínio paga onde é ambíguo).
- **SC-B — recuperação**: a cascata recupera ≥ 50% do gap da cauda E não fica abaixo do S1 puro no conjunto completo.
- **SC-C — custo**: p95 ≥ 50% menor e decode tokens ≥ 60% menores que o S2 puro (contando o custo do fast-path — tokens are money).
- **Reportados junto**: taxa de fallback no α ótimo vs banda de 30–50% do blog; gate anti-degeneração (aviso se > 95% dos itens tiverem confiança ≥ 0,95 — a lição do Marco 1).

Gate que falha é reportado como **FAIL** — uma medição, não um erro de execução (precedente: o FAIL honesto do APGR no Marco 1, commit `79a8654`).

## Formulações

**Confiança do fast-path** (média geométrica das probabilidades dos tokens gerados — menos punitiva que o produto puro em códigos de 8 dígitos):

$$\text{Confiança}(x) = \exp\!\Big(\frac{1}{n}\sum_{i=1}^{n} \ln p_i\Big)$$

**Entropia de Shannon** sobre a distribuição top-logprobs do primeiro token gerado (onde o modelo decide o início do código):

$$H(P) = - \sum_{k} P(y_k \mid x)\,\log_2 P(y_k \mid x)$$

**Roteador da cascata** (formulação RouteLLM):

$$R_\alpha(q) = \begin{cases} \text{System 1}, & \text{Confiança}(q) \geq \alpha \\ \text{System 2}, & \text{caso contrário} \end{cases}$$

**APGR** (*Average Performance Gap Recovered*, informativo pós-pivot):

$$\mathrm{APGR}(\alpha) = \frac{\mathrm{Acc}(\text{cascata}) - \mathrm{Acc}(\text{S1})}{\mathrm{Acc}(\text{S2}) - \mathrm{Acc}(\text{S1})}$$

## Componentes

| Componente | Módulo | Responsabilidade |
|---|---|---|
| A — Dataset Pipeline | `src/data/loader.py` | parser do dataset NCM, taxonomia, amostragem congelada cabeça/cauda |
| B — Serving | `src/engine/server.py` | ciclo de vida do servidor vLLM (`vLLMServerManager`) |
| C — System 1 | `src/engine/system1.py` | geração curta do código; confiança por logprobs |
| D — Sinais de incerteza | `src/engine/router.py` | entropia de Shannon sobre os top-logprobs do 1º token |
| E — Roteador | `src/engine/router.py` | decisão $R_\alpha$: confiança ≥ α aceita o S1, senão escala |
| F — System 2 | `src/engine/system2.py` | CoT + JSON; cadeia de fallback JSON → regex → sentinela |
| G — Harness | `src/metrics/summary.py` + `evaluator.py` | 3 cenários, gates, sweep de α por replay, artefatos em `results/` |
| H — Juiz (futuro) | branch `006-judge-audit` | LLM-as-a-Judge audita amostra de `results/raw_benchmark_data.json` |
| I — Insights (futuro) | branch `007-insights-loop` | discordâncias do juiz viram insights; reinjeção no prompt do S2 com delta medido |

As specs por componente ficam em `docs/` (ver estrutura abaixo); o glossário de termos do projeto é o `CONTEXT.md`.

## Estado e roteiro

- ✅ **Marco 1 — primeira medição** (histórico, commit `79a8654`): pipeline completo no ASSIN 2 (NLI em PT-BR). Resultado: p95 −98,3% e tokens −99,6% vs S2 puro, mas **APGR FAIL honesto** — S2 puro 0,865 < S1 puro 0,920: sem cauda ambígua no dataset, a cascata degenera para "sempre S1" (99,5% do tráfego no caminho barato). Essa medição motivou o pivot.
- ✅ **Marco 2 — modularização e repositório**: código em módulos `src/`, 75 testes, CI (ruff + mypy + pytest), ADR, glossário.
- 🔄 **Marco 3 — pivot de dataset para NCM** (PR [#6](https://github.com/Grazinascito/adaptive-cascade-routing/pull/6), branch `005-dataset-pivot-ncm`; spec em `docs/dataset-pivot-ncm-spec.md`): troca da tarefa para classificação NCM com cauda ambígua, gates SC-A/B/C. Código e testes prontos e verdes; falta a rodada Colab (medida completa) e os artefatos.
- **Marco 4 — ciclo de agentes de auditoria** (inspirado no ciclo Sugar → Revisor → Pepper): branch `006-judge-audit` (Componente H, juiz LLM — análogo de laboratório do Revisor), branch `007-insights-loop` (Componente I, loop de insights — a etapa *aprender*), branch `008-hitl-review` (opcional: casos não resolvidos exportados para revisão humana). Fora de escopo deliberado: análogo do Pepper — o dataset NCM tem ground truth, e a mineração dos erros da cascata fica coberta pelo Componente I.
- **Marco 5 — Hugging Face Space** (Gradio): comparação interativa de latência S1 vs S2.
- **Marco 6 — artigo técnico no LinkedIn** com os números medidos no NCM (gap da cauda, recuperação, economia de tokens/latência, taxa de fallback), fundamentado nas referências abaixo.

## Estrutura do repositório

```
adaptive-cascade-routing/
├── .github/workflows/ci.yml         # CI: ruff + mypy + pytest (Python 3.12)
├── CONTEXT.md                       # Glossário do projeto (PT-BR)
├── README.md                        # Este documento
├── pyproject.toml                   # Deps (uv), extra "serving" (vLLM), config ruff/mypy/pytest
├── requirements.txt                 # Deps core exportadas do uv.lock (para pip)
├── requirements-colab.txt           # Core + vLLM (para o Colab)
├── uv.lock                          # Lockfile commitado (reprodutibilidade)
├── config/
│   ├── benchmark_config.yaml        # Contrato congelado (dataset, α, seed, gates)
│   └── ncm_taxonomy.csv             # Taxonomia NCM derivada (114 códigos, 19 capítulos)
├── docs/
│   ├── dataset-pivot-ncm-spec.md    # Spec do pivot (Stories 1–6, gates SC-A/B/C)
│   ├── dataset-loader-spec.md       # Spec do Componente A
│   ├── engine-router-spec.md        # Spec dos Componentes B–E
│   ├── system2-evaluator-spec.md    # Spec dos Componentes F–G
│   └── adr/                         # Registros de decisão (ADR 0001: deps uv/vLLM)
├── src/
│   ├── data/loader.py               # Parser NCM, taxonomia, amostragem congelada
│   ├── engine/
│   │   ├── server.py                # vLLMServerManager
│   │   ├── system1.py               # Geração curta + confiança por logprobs
│   │   ├── system2.py               # CoT + JSON
│   │   └── router.py                # Entropia de Shannon + roteador R_α
│   └── metrics/
│       ├── summary.py               # BenchmarkResult/SummaryReport, métricas puras, gates
│       └── evaluator.py             # Harness, sweep de α, artefatos, CLI
├── notebook/
│   └── adaptive_cascade_benchmark.ipynb  # Colab: sonda + validação B/C/F/G + benchmark
├── results/                         # Artefatos medidos (Marco 1 commitado; NCM pende da rodada)
└── tests/                           # Espelha src/; mocks no limite da rede (httpx.MockTransport)
```

## Como rodar

### Local (sem GPU — testes e lint)

```bash
git clone https://github.com/Grazinascito/adaptive-cascade-routing.git
cd adaptive-cascade-routing
uv sync --dev
uv run pytest        # 75 testes; nenhum depende de rede ou GPU
```

Dependências gerenciadas com [uv](https://docs.astral.sh/uv/). O vLLM vive no extra `serving` e só é instalado em máquina CUDA (`uv sync --extra serving`) — ver [ADR 0001](docs/adr/0001-deps-uv-vllm-separado-colab.md).

### Colab (GPU T4 — benchmark completo)

Abra o [notebook no Colab](https://colab.research.google.com/github/Grazinascito/adaptive-cascade-routing/blob/005-dataset-pivot-ncm/notebook/adaptive_cascade_benchmark.ipynb). A célula de Setup clona o repo e instala `requirements-colab.txt`. A sonda carrega o pool, imprime a estatística e congela a granularidade do rótulo (`LABEL_DIGITS`); as células seguintes validam B (servidor + prefix cache), C (System 1), F (System 2) e G (benchmark completo + gates). Duração esperada: ~25–45 min.

### Servidor vLLM manual + CLI

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-AWQ --quantization awq \
    --enable-prefix-caching --gpu-memory-utilization 0.90 --port 8000

python -m src.metrics.evaluator --config config/benchmark_config.yaml   # 3 cenários + sweep
python -m src.metrics.evaluator --config config/benchmark_config.yaml --limit 12  # dry-run
```

## Referências

1. iFood Engineering Blog: *Tokens are money: a strategic guide to cost optimization in LLMs* (2024).
2. iFood Engineering Blog: *Sugar, Revisor, and Pepper: the agentic cycle that maintains iFood's catalog scale and consistency* (2024).
3. Ong, L., et al. (UC Berkeley & LMSYS): *RouteLLM: Learning to Route LLMs with Preference Data*. arXiv:2406.18665 (2024).
4. Nadir: *FrugalGPT Cascade Routing: A Step-by-Step Implementation* (2024).
5. vLLM Documentation: *OpenAI-Compatible Server — Chat Completions logprobs* (v0.32).
6. Hugging Face: `yurifacanha/NCM-RAFT-samall` — NCM classification dataset, PT-BR (2024).
7. NILC USP / Hugging Face: *ASSIN 2* — dataset da primeira iteração (histórico, commit `79a8654`).
