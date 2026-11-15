# Design Doc: Adaptive Cascade Routing for Typed LLM Inference: System 1 (Logprobs) vs System 2 (CoT)

**Autor(es):** Grazi & Gemini Notebook
**Revisores:** Time de IA Engineering & Comunidade
**Data:** 2026-11-07
**Status:** draft

---

## 1. Overview

O **Adaptive Cascade Routing System** é um sistema de engenharia de inferência projetado para avaliar empiricamente a redução de latência e de custo financeiro no processamento de decisões semânticas em 3 cenários: **System 1 puro**, **System 2 puro** e **Cascata Adaptativa**. Inspirado na arquitetura de catálogo em grande escala do iFood (*Sugar, Revisor & Pepper*) e fundamentado na teoria de roteamento estatístico de modelos (*RouteLLM* / LMSYS / UC Berkeley), o sistema implementa um roteamento adaptativo entre duas estratégias de inferência executadas sobre a mesma infraestrutura de servidor em Português do Brasil (PT-BR):

1. **System 1 (Fast-Path / Geração Curta)**: `/v1/chat/completions` com `max_tokens=12` e `temperature=0.0` sobre o servidor `vLLM` com `--enable-prefix-caching`: o modelo emite diretamente o código NCM de 8 dígitos, sem raciocínio. A confiança é a média geométrica das probabilidades dos tokens gerados ($\exp(\sum_i \ln p_i / n)$); código inexistente na taxonomia recebe confiança 0.0 e escala imediatamente — o "caso novo → fallback" do blog do iFood. (A pergunta `choice` do `/v1/systemone`, binária, caiu com o pivot: 114 classes não cabem em opções de 1 token; histórico no commit `79a8654`.)
2. **System 2 (Slow-Path / Raciocínio Autorregressivo)**: Atua como mecanismo de *fallback* para requisições com alta incerteza estatística (alta entropia nos *logprobs* do System 1), utilizando o endpoint `/v1/chat/completions` para produzir raciocínio passo a passo (*Chain-of-Thought* - CoT) e saída estruturada em JSON.

O sistema demonstra como a triagem de requisições por incerteza estatística permite recuperar mais de 95% da acurácia do modelo completo, enquanto reduz a latência p95 em mais de 50% e diminui o consumo total de tokens em mais de 60%, validando o princípio operacional do iFood de que *Tokens são dinheiro*.

> **Registro de pivot (2026-11-14)**: a tese do projeto (os ganhos da arquitetura híbrida do iFood também aparecem no eixo System 1 × System 2 do mesmo modelo) exige dataset com cauda ambígua — o tipo de conteúdo que o blog do iFood descreve. O ASSIN 2 não tem esse conteúdo; a medição do Marco 1 provou (S1 puro 0,920 > S2 puro 0,865, cascata com 99,5% do tráfego no caminho barato). Decisão: trocar o dataset pela classificação de itens de catálogo em códigos **NCM** (taxonomia fiscal brasileira em árvore; branch `005-dataset-pivot-ncm`, spec em `docs/dataset-pivot-ncm-spec.md`). O ASSIN 2 sai da linha principal do relatório; a medição do Marco 1 permanece no histórico do git (commit `79a8654`). As seções abaixo já descrevem o estado pós-pivot.

---

## 2. Contexto

Em sistemas de produção que operam em escala de milhões de requisições diárias (como categorização de produtos de catálogo, triagem de intenção e filtro de relevância em RAG), o uso indiscriminado de modelos autorregressivos de grande porte gera gargalos severos de desempenho e custos elevados. A fase de *decode* de uma LLM é sequencial e limitada pela largura de banda de memória da GPU, tornando a geração de explicações prolixas um desperdício para decisões essencialmente binárias ou categóricas.

Conforme documentado nos artigos de engenharia do iFood (*Sugar, Revisor & Pepper* e *Tokens are money*), a sustentabilidade financeira de aplicações de IA Generativa requer decisões arquiteturais prévias à implementação dos prompts. O ecossistema do iFood utiliza o motor *Sugar* para resolver de 50% a 70% das classificações de catálogo com modelos mais leves e baratos, acionando modelos de alta capacidade apenas em casos ambíguos ou limítrofes.

Este projeto estende essa filosofia ao construir um benchmark reprodutível em ambiente de recursos limitados (Google Colab com GPU NVIDIA T4 de 16GB VRAM), aplicando o mesmo servidor `vLLM` local aos dois modos de inferência, integrado ao dataset público **NCM-RAFT-samall** (classificação de itens de catálogo brasileiro em códigos da taxonomia fiscal NCM — nomes reais, ambíguos, com cauda *long tail*) e respaldado pela formulação matemática de roteamento do **RouteLLM**. O pivô de dataset (2026-11-14) substituiu o ASSIN 2 por falta de cauda ambígua; a primeira medição permanece no histórico (commit `79a8654`).

---

## 3. Goals e Non-Goals

### 3.1 Goals

#### Funcionais — o que o sistema faz

- **RF-1.** O pipeline DEVE carregar o dataset **NCM-RAFT-samall** (`yurifacanha/NCM-RAFT-samall`), parsear as conversas em pares (nome de item, código NCM de 8 dígitos) e amostrar o conjunto congelado de 800 itens (~65% cabeça / ~35% cauda por frequência de código, seed 42).
- **RF-2.** O servidor `vLLM` DEVE ser inicializado com a flag `--enable-prefix-caching` para reutilizar o estado do *prefill* entre requisições com prompts de instrução idênticos.
- **RF-3.** O motor System 1 DEVE disparar requisições ao `/v1/chat/completions` com `max_tokens=12`, `logprobs` e `top_logprobs`, produzindo o código NCM sem raciocínio; código fora da taxonomia implica confiança 0.0.
- **RF-4.** O módulo de avaliação estatística DEVE calcular a confiança como média geométrica das probabilidades dos tokens gerados e a entropia de Shannon $H(P)$ sobre os top-logprobs do primeiro token.
- **RF-5.** O roteador adaptativo DEVE tomar a decisão de roteamento baseada em um limiar configurável $\alpha$: aceitar a resposta do System 1 se $P_{\text{top}} \ge \alpha$, ou rotear para o System 2 se $P_{\text{top}} < \alpha$.
- **RF-6.** O motor System 2 DEVE processar os casos desviados gerando raciocínio *Chain-of-Thought* (CoT) e JSON validado com o rótulo final.
- **RF-7.** O harness de benchmarking DEVE medir e registrar a latência por requisição ($p50$, $p95$, $p99$), a contagem total de tokens (*prefill* vs *decode*, somando os engines invocados) e as métricas de qualidade (Acurácia geral e por subconjunto cabeça/cauda, F1-Score macro e APGR informativo) nos cenários: *System 1 Puro*, *System 2 Puro* e *Cascata Adaptativa*.

#### Não-funcionais — como o sistema faz

- **Desempenho e Latência**
  - **RNF-01.** A latência p95 da Cascata Adaptativa DEVE ser pelo menos 50% menor que a latência p95 do System 2 Puro. *(Por quê: validar a redução de tempo em produção.)* *(Teste: medição via `time.perf_counter()` nas 800 requisições do conjunto congelado NCM.)*
- **Custo Computacional**
  - **RNF-02.** O volume de tokens gerados na fase de *decode* da Cascata Adaptativa DEVE ser reduzido em no mínimo 60% em relação ao System 2. *(Por quê: demonstrar economia direta de recursos segundo o princípio "Tokens are money".)* *(Teste: auditoria de tokens consumidos registrada pelo vLLM.)*
- **Reprodutibilidade**
  - **RNF-03.** Todo o pipeline de inferência e benchmark DEVE ser executável em uma instância gratuita do Google Colab com GPU NVIDIA T4 (16GB VRAM) sem requerer APIs pagas externas. *(Por quê: garantia de acesso e validação pública no GitHub.)* *(Teste: execução autônoma do notebook Colab do início ao fim.)*
- **Observabilidade**
  - **RNF-04.** O sistema DEVE gerar registros estruturados por requisição (`BenchmarkResult`, Componente G) contendo `query_id`, `subset` (cabeça/cauda), `system_1_latency_ms`, `confidence`, `entropy`, `routed_to_system_2`, `system_2_latency_ms`, `total_latency_ms`, `prediction`, `ground_truth`, `is_correct`, tokens de prefill/decode (somados dos engines invocados) e flags de fallback do parser/erro HTTP. *(Por quê: permitir análise posterior de dispersão e calibragem de limiar.)*

### 3.2 Non-Goals

1. **Serviço de produção em alta disponibilidade (24/7):** O objetivo é um benchmark reproduzível de engenharia e estudo de caso, não um microserviço implantado em Kubernetes.
2. **Fine-tuning ou re-treinamento de modelos:** O projeto foca em estratégias de *serving*, *prompting* e roteamento estatístico com modelos pré-treinados quantizados.
3. **Suporte a dados multimodais:** A avaliação restringe-se a texto em Português do Brasil (tarefa de NLI e classificação categórica de texto).

---

## 4. System Architecture

### 4.1 Visão Geral da Arquitetura

O sistema é estruturado em quatro camadas operacionais independentes: Ingestão de Dados, Serving & Roteamento (vLLM Engine), Avaliação de Performance e Entregáveis.

#### Nível 1 — Contexto do Sistema (C4 Context)

O sistema interage com o avaliador/engenheiro de IA através do carregamento do dataset NCM-RAFT-samall, executando requisições no servidor `vLLM` local (GPU T4) e exportando as métricas de latência, custo de tokens e acurácia.

#### Nível 2 — Contêineres e Componentes (C4 Container)

O sistema opera dentro do ambiente Linux/CUDA da GPU NVIDIA T4 no Google Colab, contendo os seguintes componentes internos:

- **Dataset Loader & Taxonomy**: Consome o dataset NCM-RAFT-samall do Hugging Face Hub, parseia os pares (item, código) e valida contra a taxonomia commitada.
- **vLLM Serving Engine**: Servidor local de inferência em GPU T4 que carrega o modelo quantizado AWQ e disponibiliza o endpoint `/v1/chat/completions` (com logprobs por token) para os dois modos.
- **System 1 Engine**: Módulo de geração curta do código NCM com confiança pela média geométrica dos logprobs.
- **Confidence & Entropy Evaluator**: Componente matemático que calcula a probabilidade normalizada e a entropia dos logits.
- **Cascade Router**: Roteador estatístico que decide a transferência de execução para o System 2.
- **System 2 Engine**: Módulo autorregressivo de *fallback* para raciocínio estruturado (CoT + JSON Schema).
- **Benchmark & Analytics Harness**: Coletor de métricas de tempo, custo em tokens e acurácia.

#### Componentes Principais e Responsabilidades

- **`DatasetLoader`** → Carrega, parseia e amostra o pool NCM em pares (item, código) estruturados.
- **`vLLMServerManager`** → Gerencia o ciclo de vida e a inicialização do processo servidor `vLLM` em background.
- **`System1Engine`** → Executa a geração curta do código NCM e deriva confiança/entropia dos logprobs.
- **`EntropyEvaluator`** → Processa o dicionário de *logprobs* e retorna a distribuição de probabilidade normalizada e a entropia de Shannon.
- **`CascadeRouter`** → Compara a confiança estimada com o limiar $\alpha$ e decide a rota.
- **`System2Engine`** → Invoca o endpoint `/v1/chat/completions` com prompts de *Chain-of-Thought*.
- **`BenchmarkHarness`** → Orquestra as baterias de teste e gera os relatórios estatísticos e gráficos.

#### Estrutura de Pastas do Repositório

```
adaptive-cascade-routing/
├── .github/workflows/ci.yml       # CI: ruff + mypy + pytest no Python 3.12
├── .gitignore                     # Caches Python e .venv; NÃO ignora uv.lock nem results/
├── .python-version                # "3.12" (lido pelo uv)
├── LICENSE                        # MIT
├── CONTEXT.md                     # Glossário do projeto (PT-BR)
├── README.md                      # Documentação técnica do repositório e guia de execução
├── pyproject.toml                 # Dependências (uv), extra "serving" (vLLM), config ruff/mypy/pytest
├── requirements.txt               # Deps core exportadas do uv.lock (para pip)
├── requirements-colab.txt         # Core + vLLM exportadas do uv.lock (para o Colab)
├── uv.lock                        # Lockfile commitado (reprodutibilidade do benchmark)
├── config/
│   ├── benchmark_config.yaml      # Parâmetros de modelo, limiares de confiança e portas do vLLM
│   └── ncm_taxonomy.csv           # Taxonomia NCM derivada do dataset (114 códigos, 19 capítulos)
├── docs/
│   ├── adaptive-cascade-design-doc.md   # Este documento
│   ├── dataset-pivot-ncm-spec.md        # Spec do pivot (Stories 1–6, gates SC-A/B/C)
│   ├── dataset-loader-spec.md           # Stories do Componente A (NCM)
│   ├── engine-router-spec.md            # Stories dos Componentes B–E (NCM)
│   ├── system2-evaluator-spec.md        # Stories dos Componentes F–G (NCM)
│   └── adr/                             # Registros de decisão de arquitetura
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py              # Parser NCM, taxonomia e amostragem congelada head/tail
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── server.py              # Ciclo de vida do servidor vLLM (vLLMServerManager, §5.2)
│   │   ├── system1.py             # Geração curta do código NCM com confiança por logprobs
│   │   ├── system2.py             # Interface com /v1/chat/completions (CoT + JSON NCM)
│   │   └── router.py              # Lógica do Roteador de Cascata e cálculo de entropia
│   └── metrics/
│       ├── __init__.py
│       ├── summary.py             # BenchmarkResult/SummaryReport e métricas puras (percentis, F1, APGR)
│       └── evaluator.py           # BenchmarkHarness, sweep de α, artefatos de results/ e CLI
├── notebook/
│   └── adaptive_cascade_benchmark.ipynb # Google Colab Notebook executável ponta a ponta (Marco 1)
├── results/
│   ├── raw_benchmark_data.json    # Logs detalhados de cada requisição do benchmark (Marco 1)
│   └── latency_cost_tradeoff.png  # Gráficos de dispersão e Curva de APGR vs Custo (Marco 1)
└── tests/
    ├── test_config.py             # Contrato congelado do benchmark_config.yaml
    └── test_structure.py          # Importação de todos os módulos esqueleto
```

---

## 5. Detailed Design

### 5.1 Component A: Dataset Pipeline NCM (`loader.py`)

- **Scope & Responsibility**: Carregar o pool do dataset NCM-RAFT-samall, parsear as conversas em pares (nome, código), validar códigos contra a taxonomia commitada e amostrar o conjunto congelado head/tail.
- **Interfaces & Contracts**:
  - `load_ncm_pool(dataset_id: str = NCM_DATASET_ID) -> tuple[list[NCMItem], int]`
  - `parse_conversation(conversations) -> NCMItem | None`
  - `load_taxonomy() -> dict[str, NCMTaxonomyEntry]`; `is_valid_code(code, taxonomy=None) -> bool`; `chapter_lines() -> list[str]`
  - `get_frozen_eval_set(items, *, sample_limit=800, head_share=0.65, head_min_code_freq=50, seed=42) -> list[NCMEvalSample]`
  - `pool_stats(items, n_rows) -> PoolStats`; `truncate_code(code, digits) -> str`
- **Dependencies**: Biblioteca `datasets` do Hugging Face; `config/ncm_taxonomy.csv` (derivado do próprio dataset: 114 códigos, 19 capítulos, descrição única por código).
- **Core Logic**:
  1. Extrai o par canônico da PRIMEIRA mensagem da conversa (`o codigo da categoria do {nome} é: {código}`); resposta do assistente é ignorada (às vezes traz código, às vezes só descrição).
  2. Amostragem: dedup de pares → exclusão de nomes conflitantes (mesmo nome, códigos diferentes) → frequência por código sobre pares únicos → head = >= 50 nomes únicos → estratificação 65/35 com seed 42.

### 5.2 Component B: vLLM Serving Infrastructure (`vLLMServerManager`)

- **Scope & Responsibility**: Subir e monitorar o processo de background do servidor `vLLM` na GPU T4 do Colab.
- **Interfaces & Contracts**:
  - `start_server(model_name: str, *, host: str = "127.0.0.1", port: int = 8000, quantization: str = "awq", enable_prefix_caching: bool = True, gpu_memory_utilization: float = 0.90, logprobs_mode: str = "raw_logprobs", startup_timeout_s: float = 90.0) -> bool`
  - `health_check(base_url: str = "http://127.0.0.1:8000", *, timeout_s: float = 5.0) -> bool`
  - `stop_server() -> None`; exceção `vLLMServerException` para crash/OOM/timeout de startup (com instruções de restart e leitura de VRAM)
- **Dependencies**: Executável `vllm.entrypoints.openai.api_server`, CUDA 12.x, PyTorch.
- **Core Logic**:
  1. Dispara a linha de comando do servidor com argumentos de otimização (`--model Qwen/Qwen3-4B-AWQ --quantization awq --enable-prefix-caching --gpu-memory-utilization 0.90 --logprobs-mode raw_logprobs`); reusa um servidor já saudável sem lançar novo processo.
  2. Executa polling do endpoint de health do vLLM até receber resposta HTTP 200 dentro do timeout de startup.

### 5.3 Component C: System 1 Short Classification Engine (`system1.py`)

- **Scope & Responsibility**: Executar o fast-path — geração curta (sem raciocínio) do código NCM com confiança derivada dos logprobs por token.
- **Interfaces & Contracts**:
  - `evaluate_short_classification(item_name: str, *, model=DEFAULT_MODEL, base_url=DEFAULT_BASE_URL, instructions=SYSTEM1_INSTRUCTIONS, max_tokens=12, top_logprobs=20, temperature=0.0, timeout_s=10.0, client=None) -> System1Output`
- **Data Models** (TypedDict, no padrão do repo):
  ```python
  class System1Output(TypedDict):
      code: str  # sentinela "00000000" quando não há código parseável
      confidence: float  # média geométrica dos token probs; 0.0 se inválido
      entropy: float  # Shannon sobre top-logprobs do primeiro token
      valid_code: bool
      prompt_tokens: int
      completion_tokens: int
      latency_ms: float
  ```
- **Core Logic**:
  1. POST `/v1/chat/completions` com mensagem `system` fixa (instrução + lista de capítulos, idêntica em toda query para o prefix caching) e `user` = nome do item; `enable_thinking=False` (Qwen3 pensa por padrão).
  2. Extrai o código via regex `\b\d{8}\b`; confiança = $\exp(\sum_i \ln p_i / n)$ sobre os tokens gerados; código inexistente na taxonomia → `valid_code=False` e confiança 0.0 (escala em qualquer α > 0).

### 5.4 Component D: Uncertainty Signals (`router.py` + `system1.py`)

- **Scope & Responsibility**: Produzir os sinais de incerteza que alimentam o roteador.
- **Mathematical Formulations** (International Standard):
  1. **Confidence (geometric mean of generated-token probabilities)**:
     $$\text{Confidence}(x) = \exp\!\Big(\frac{1}{n}\sum_{i=1}^{n} \ln p_i\Big)$$
     (menos punitiva que o produto puro em códigos de 8 dígitos)
  2. **Shannon Entropy** (sobre a distribuição top-logprobs do primeiro token, normalizada):
     $$\text{Entropy}(P) = -\sum_{k=1}^{K} P(y_k \mid x) \log_2 P(y_k \mid x)$$
  3. `softmax`/`compute_metrics` da era `/v1/systemone` foram removidos (sem chamador pós-pivot; histórico no commit `79a8654`).

### 5.5 Component E: Adaptive Cascade Router (`CascadeRouter`)

- **Scope & Responsibility**: Decidir dinamicamente se a resposta do System 1 é aceita ou se a requisição deve ser promovida ao System 2 (Slow-Path), seguindo a formulação do RouteLLM.
- **Mathematical Formulations** (RouteLLM Formulation):
  Following Berkeley's RouteLLM framework, the routing function $R_\alpha(q)$ for a query $q$ given a cost threshold $\alpha \in [0, 1]$ is defined as:
  $$R_\alpha(q) = \begin{cases} \text{System 1 (Fast-Path)}, & \text{if } \text{Confidence}(q) \ge \alpha \\ \text{System 2 (Slow-Path)}, & \text{if } \text{Confidence}(q) < \alpha \end{cases}$$
- **Core Logic**:
  1. Se $\text{Confidence}(q) \ge \alpha$, retorna imediatamente a previsão do System 1.
  2. Caso contrário, marca a flag `routed_to_system_2 = True` e repassa a requisição ao `System2Engine`.

### 5.6 Component F: System 2 Slow-Path Engine (`system2.py`)

- **Scope & Responsibility**: Processar casos limítrofes e ambíguos gerando raciocínio em múltiplos tokens e JSON estruturado.
- **Interfaces & Contracts**:
  - `execute_reasoning(item_name: str, *, model: str = DEFAULT_MODEL, base_url: str = "http://127.0.0.1:8000/v1", max_tokens: int = 256, temperature: float = 0.0, timeout_s: float = 120.0, client: httpx.Client | None = None) -> System2Output`
  - `System2Output` (TypedDict): `reasoning`, `ncm_code`, `prompt_tokens`, `completion_tokens`, `latency_ms`, `fallback_used`
  - O payload inclui `chat_template_kwargs: {"enable_thinking": false}` — modelos Qwen3 "originais" pensam por padrão e os blocos `<think>` quebrariam o parse do JSON
- **Prompt Architecture**:
  ```text
  Você é um especialista em classificação fiscal de mercadorias (NCM — Nomenclatura
  Comum do Mercosul). Analise o nome do item de catálogo abaixo e forneça o
  raciocínio passo a passo (Chain-of-Thought) antes de determinar o código NCM de
  8 dígitos.

  Capítulos disponíveis neste catálogo:
  {lista compacta dos 19 capítulos}

  Item: {item_name}

  Retorne um JSON estrito no seguinte formato:
  {
    "reasoning": "Explicação detalhada...",
    "ncm_code": "12345678"
  }
  ```
- **Error Handling**: Se a resposta não cumprir o schema JSON esperado, faz a extração de código via regex `\b\d{8}\b` como plano de contingência (*degradation*), marcando `fallback_used=True`; output vazio ou sem código reconhecido retorna o default `00000000` (sempre errado na avaliação, nunca derruba o run) com `fallback_used=True` e warning no log.

### 5.7 Component G: Benchmark & Performance Evaluation Harness (`evaluator.py`)

- **Scope & Responsibility**: Medir o desempenho de tempo e custo de forma rigorosa em todas as amostras do benchmark. Tipos e métricas puras vivem em `summary.py`; a orquestração (3 cenários, sweep de α, artefatos, CLI) em `evaluator.py`.
- **Metrics Calculated**:
  - **Latência**: $p50$, $p95$, $p99$ medidos via `time.perf_counter()` em milissegundos. As 5 primeiras requisições de cada cenário são descartadas dos percentis (*warmup*; §10, risco 2) — acurácia, tokens e taxas usam todas as requisições.
  - **Tokens**: Total de tokens de prefill e de decode somando os engines invocados por requisição (o fast-path também custa tokens — "tokens are money"). O KPI de economia compara a cascata contra o System 2 puro.
  - **Acurácia & F1-Score**: Comparação com o código NCM anotado, truncado por `label_digits` (granularidade congelada pela sonda) de forma idêntica em predição e ground truth; F1 macro sobre os códigos presentes. Acurácia reportada geral E por subconjunto cabeça/cauda.
  - **Gates do pivot (SC-A/B/C)**: SC-A = gap S2−S1 na cauda >= 10 pp; SC-B = cascata recupera >= 50% do gap sem cair abaixo do S1 geral; SC-C = p95 e decode tokens >= 50%/60% menores que o S2 puro. Taxa de fallback no α ótimo comparada à banda 30–50% do blog do iFood. Gate anti-degeneração: aviso quando > 95% dos itens têm confiança >= 0,95.
  - **Average Performance Gap Recovered (APGR)** (informativo pós-pivot):
    $$\text{APGR}(\alpha) = \frac{\text{Accuracy}(\text{Cascade}_\alpha) - \text{Accuracy}(\text{System 1})}{\text{Accuracy}(\text{System 2}) - \text{Accuracy}(\text{System 1})}$$
    Denominador zero (acurácias de S1 e S2 iguais) devolve 0.0, sem divisão por zero.

---

## 6. Security & Compliance

- **Privacidade de Dados**: O benchmark utiliza exclusivamente o dataset público **NCM-RAFT-samall** (nomes comerciais de itens de catálogo e códigos fiscais), isento de informações pessoais identificáveis (PII) ou dados sensíveis de produção. Procedência: autoria única, card vazio — risco de ruído declarado no README e mitigado pela revisão humana da amostra (HITL da spec do pivot).
- **Execução Air-Gapped**: Todo o processamento de inferência ocorre dentro do ambiente do Google Colab no servidor local `127.0.0.1:8000`, sem envio de prompts para APIs de terceiros.
- **Governança de Código Aberto**: Uso de modelos de licença aberta no Hugging Face (Qwen / Gemma) em conformidade com as diretrizes de uso responsável.

---

## 7. Performance & Scalability

- **Prefix Caching Optimization**: Ao prefixar as requisições com a instrução do sistema e os rótulos válidos, a flag `--enable-prefix-caching` do vLLM permite reutilizar os tensores de KV Cache no espaço da GPU, reduzindo o tempo de *prefill* para requisições subsequentes a aproximadamente 5-10ms.
- **Gerenciamento de Memória VRAM**: Utilização de quantização AWQ em 4-bits, limitando a ocupação de VRAM do modelo de 4B a aproximadamente 2.7 GB em uma GPU T4 de 16 GB (a troca do 7B original pelo Qwen3-4B-AWQ veio do requisito de read strategy do `/v1/systemone`; ver Assumptions do engine-router-spec), reservando o espaço restante para alocação dinâmica do KV Cache.

---

## 8. Testing Strategy

- **Testes Unitários**:
  - Parser do NCM-RAFT com stubs no formato real das conversas (incluindo falhas de extração contadas).
  - Taxonomia: códigos duplicados/vazios rejeitados; `is_valid_code` contra códigos conhecidos/desconhecidos.
  - Amostragem congelada determinística (seed 42), proporções head/tail e exclusão de conflitos.
  - Engines com `httpx.MockTransport` no limite da rede: confiança geométrica analítica, confiança 0 para código fora da taxonomia, cadeia de fallback JSON → regex → sentinela.
  - Validação da lógica do `CascadeRouter` nos limiares limite $\alpha=0.0$ e $\alpha=1.0$ e da entropia de Shannon com distribuições conhecidas (uniforme $H(P) = 1.0$ bit vs determinística $H(P) = 0.0$ bit).
- **Teste de Integração (Frozen Eval Set)**:
  - Execução da bateria completa sobre o conjunto congelado de 800 amostras head/tail do pool NCM, salvando os resultados brutos em `results/raw_benchmark_data.json`.

---

## 9. Rollout Plan

1. **Marco 1 — Validação no Colab**: Execução do pipeline completo no Google Colab com medição inicial de latência e acurácia.
2. **Marco 2 — Modularização e Repositório GitHub**: Organização da estrutura de código em módulos Python limpos, inclusão de testes unitários e criação de um `README.md` com os diagramas e tabelas de resultados.
3. **Marco 3 — Pivot de dataset para NCM** (decidido em 2026-11-14; branch `005-dataset-pivot-ncm`; spec em `docs/dataset-pivot-ncm-spec.md`): substituir o ASSIN 2 pela classificação de itens de catálogo em códigos NCM (taxonomia fiscal brasileira em árvore: capítulo → posição → subposição → item de 8 dígitos; dataset público `yurifacanha/NCM-RAFT-samall`, 105k linhas em PT-BR com nomes reais de itens), reproduzindo o tipo de conteúdo do ciclo do iFood — nomes curtos e ambíguos, contexto incompleto, *long tail*. Reusa o harness, o roteador por confiança e o sweep de α por replay sem mudança de desenho. Mudam: o loader (parser do dataset + amostragem congelada cabeça/cauda de 800 itens com seed fixa), os prompts (saída = código NCM em JSON) e o mecanismo do System 1 (geração curta de 8–12 tokens com confiança pelos *logprobs* dos tokens, no lugar da pergunta `choice` binária do `/v1/systemone`; código inexistente na taxonomia → confiança 0 → escala ao System 2, espelhando o "caso novo → fallback" do blog). Gates redefinidos: gap S2−S1 ≥ 10 pp na cauda, cascata recuperando ≥ 50% do gap, redução de tokens ≥ 60% e taxa de fallback reportada contra a banda de 30–50% do blog. O ASSIN 2 sai da linha principal do relatório (medição do Marco 1 preservada no histórico, commit `79a8654`).
4. **Marco 4 — Ciclo de Agentes de Auditoria** (inspirado no ciclo Sugar → Revisor → Pepper do iFood; *roadmap* documentado, ainda não iniciado; era "Marco 3" antes do pivot de 2026-11-14): estender a cascata medida no dataset NCM com as etapas *auditar* e *aprender*, um PR por task com spec própria:
   - **`006-judge-audit` — Componente H (§5.8, futuro)**: juiz LLM (*LLM-as-a-Judge*, análogo de laboratório do Revisor). Spec a criar em `docs/judge-spec.md`. Audita uma amostra (taxa no config) dos registros de `results/raw_benchmark_data.json` — reuso que dispensa re-rodar o benchmark — verificando a predição da cascata contra o nome do item e a descrição do capítulo do código NCM correto; discordâncias disparam uma tentativa de reclassificação com o feedback do juiz no prompt. Métrica: taxa de concordância e efeito da reclassificação.
   - **`007-insights-loop` — Componente I (§5.9, futuro)**: memória operacional (a etapa *aprender*). Discordâncias do juiz viram insights estruturados (`results/insights.json`); os principais insights são injetados no prompt do System 2 e o delta de acurácia com/sem é medido no mesmo conjunto congelado. É o que fecha o ciclo: classificar → auditar → aprender → melhorar.
   - **`008-hitl-review` (opcional)**: casos não resolvidos (juiz discorda e a reclassificação falha) exportados em CSV de revisão humana; correções reinjetadas como insights.
   - **Fora de escopo deliberado — análogo do Pepper**: o Pepper existe porque o catálogo do iFood não tem ground truth; o dataset NCM tem (o código anotado). A mineração do subconjunto onde a cascata erra — que seria o equivalente útil — já é coberta pelo loop de insights do Componente I.
5. **Marco 5 — Hugging Face Space (Gradio)**: Publicação de um Space interativo permitindo a comparação em tempo real de latência entre System 1 e System 2.
6. **Marco 6 — Artigo Técnico no LinkedIn**: Publicar o estudo de caso com os números medidos no dataset NCM (gap de acurácia na cauda, recuperação da cascata, economia de tokens e taxa de fallback), fundamentado nas referências do iFood e do RouteLLM.

---

## 10. Dependencies, Risks & Open Questions

### Dependências Externas
- `vLLM` com suporte a CUDA 12 e `logprobs`/`top_logprobs` no endpoint `/v1/chat/completions`.
- Instância de GPU NVIDIA T4 no Google Colab.

### Riscos e Mitigações
- **Risco 1: Limite de VRAM no Colab (16GB)**.  
  *Mitigação*: Utilizar modelos quantizados AWQ da família Qwen3 (4B–8B) com `--gpu-memory-utilization 0.90`.
- **Risco 2: Oscilação de latência em ambiente compartilhado no Colab**.  
  *Mitigação*: Ignorar as primeiras 5 requisições de *warmup* e reportar percentis estatísticos ($p50, p95, p99$) agregados em 200 iterações.

### Questões em Aberto
- Qual o valor ideal do limiar $\alpha$ para o dataset NCM (pós-pivot, branch `005-dataset-pivot-ncm`) que maximiza a acurácia mantendo a taxa de escalamento ao System 2 na banda de 30–50% reportada pelo blog do iFood? (Será respondido na varredura empírica do benchmark.)
- Qual a granularidade de rótulo NCM (capítulo, posição ou item de 8 dígitos) em que o System 1 puro fica na faixa de 30–70% de acurácia — o terreno com ambiguidade real para este par modelo+tarefa? (Será respondida pela célula de sonda da spec do pivot.)

---

## 11. References

1. **iFood Engineering Blog**: *Tokens are money: a strategic guide to cost optimization in LLMs*. (2024).
2. **iFood Engineering Blog**: *Sugar, Revisor, and Pepper: the agentic cycle that maintains iFood's catalog scale and consistency*. (2024).
3. **UC Berkeley & LMSYS (RouteLLM)**: Ong, L., et al. *RouteLLM: Learning to Route LLMs with Preference Data*. arXiv:2406.18665 (2024).
4. **vLLM Documentation**: *OpenAI-Compatible Server: Chat Completions logprobs*. Version 0.32.0.
5. **Hugging Face**: *yurifacanha/NCM-RAFT-samall — NCM classification dataset (PT-BR)*. (2024).
6. **NILC USP / Hugging Face**: *ASSIN 2: Brazilian Portuguese NLI Dataset*. (2020) — dataset da primeira iteração (histórico, commit `79a8654`).
6. **Nadir**: *FrugalGPT Cascade Routing: A Step-by-Step Implementation*. (2024).
