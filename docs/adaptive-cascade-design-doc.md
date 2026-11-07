# Design Doc: Adaptive Cascade Routing for Typed LLM Inference: System 1 (Logprobs) vs System 2 (CoT)

**Autor(es):** Grazi & Gemini Notebook
**Revisores:** Time de IA Engineering & Comunidade
**Data:** 2026-11-07
**Status:** draft

---

## 1. Overview

O **Adaptive Cascade Routing System** é um sistema de engenharia de inferência projetado para avaliar empiricamente a redução de latência e de custo financeiro no processamento de decisões semânticas em 3 cenários: **System 1 puro**, **System 2 puro** e **Cascata Adaptativa**. Inspirado na arquitetura de catálogo em grande escala do iFood (*Sugar, Revisor & Pepper*) e fundamentado na teoria de roteamento estatístico de modelos (*RouteLLM* / LMSYS / UC Berkeley), o sistema implementa um roteamento adaptativo entre duas estratégias de inferência executadas sobre a mesma infraestrutura de servidor em Português do Brasil (PT-BR):

1. **System 1 (Fast-Path / Decisão Tipada)**: Utiliza a rota especializada `/v1/systemone` da engine `vLLM` com `--enable-prefix-caching`, avaliando a distribuição de probabilidades de tokens (*logprobs*) e gerando exatamente 1 token de resposta (`max_tokens=1`). Essa abordagem elimina a fase autorregressiva de *decode*, alcançando latências da ordem de dezenas de milissegundos.
2. **System 2 (Slow-Path / Raciocínio Autorregressivo)**: Atua como mecanismo de *fallback* para requisições com alta incerteza estatística (alta entropia nos *logprobs* do System 1), utilizando o endpoint `/v1/chat/completions` para produzir raciocínio passo a passo (*Chain-of-Thought* - CoT) e saída estruturada em JSON.

O sistema demonstra como a triagem de requisições por incerteza estatística permite recuperar mais de 95% da acurácia do modelo completo, enquanto reduz a latência p95 em mais de 50% e diminui o consumo total de tokens em mais de 60%, validando o princípio operacional do iFood de que *Tokens são dinheiro*.

---

## 2. Contexto

Em sistemas de produção que operam em escala de milhões de requisições diárias (como categorização de produtos de catálogo, triagem de intenção e filtro de relevância em RAG), o uso indiscriminado de modelos autorregressivos de grande porte gera gargalos severos de desempenho e custos elevados. A fase de *decode* de uma LLM é sequencial e limitada pela largura de banda de memória da GPU, tornando a geração de explicações prolixas um desperdício para decisões essencialmente binárias ou categóricas.

Conforme documentado nos artigos de engenharia do iFood (*Sugar, Revisor & Pepper* e *Tokens are money*), a sustentabilidade financeira de aplicações de IA Generativa requer decisões arquiteturais prévias à implementação dos prompts. O ecossistema do iFood utiliza o motor *Sugar* para resolver de 50% a 70% das classificações de catálogo com modelos mais leves e baratos, acionando modelos de alta capacidade apenas em casos ambíguos ou limítrofes.

Este projeto estende essa filosofia ao construir um benchmark reprodutível em ambiente de recursos limitados (Google Colab com GPU NVIDIA T4 de 16GB VRAM), aplicando o endpoint `/v1/systemone` do `vLLM` integrado ao dataset público **ASSIN 2** (NLI em PT-BR) e respaldado pela formulação matemática de roteamento do **RouteLLM**.

---

## 3. Goals e Non-Goals

### 3.1 Goals

#### Funcionais — o que o sistema faz

- **RF-1.** O pipeline DEVE carregar o dataset **ASSIN 2** (`nilc-nlp/assin2`) e normalizar os pares de frases (Premissa e Hipótese) para uma tarefa de inferência textual binária (`Entailment` vs `Contradiction`).
- **RF-2.** O servidor `vLLM` DEVE ser inicializado com a flag `--enable-prefix-caching` para reutilizar o estado do *prefill* entre requisições com prompts de instrução idênticos.
- **RF-3.** O motor System 1 DEVE disparar requisições ao endpoint `/v1/systemone` solicitando `max_tokens=1` e `logprobs=True`, extraindo as distribuições de probabilidade dos tokens candidatos.
- **RF-4.** O módulo de avaliação estatística DEVE calcular a probabilidade da resposta de topo $P_{\text{top}}$ e a entropia de Shannon $H(P)$ sobre os rótulos permitidos.
- **RF-5.** O roteador adaptativo DEVE tomar a decisão de roteamento baseada em um limiar configurável $\alpha$: aceitar a resposta do System 1 se $P_{\text{top}} \ge \alpha$, ou rotear para o System 2 se $P_{\text{top}} < \alpha$.
- **RF-6.** O motor System 2 DEVE processar os casos desviados gerando raciocínio *Chain-of-Thought* (CoT) e JSON validado com o rótulo final.
- **RF-7.** O harness de benchmarking DEVE medir e registrar a latência por requisição ($p50$, $p95$, $p99$), a contagem total de tokens (*prefill* vs *decode*) e as métricas de qualidade (Acurácia, F1-Score e *Average Performance Gap Recovered* - APGR) nos cenários: *System 1 Puro*, *System 2 Puro* e *Cascata Adaptativa*.

#### Não-funcionais — como o sistema faz

- **Desempenho e Latência**
  - **RNF-01.** A latência p95 da Cascata Adaptativa DEVE ser pelo menos 50% menor que a latência p95 do System 2 Puro. *(Por quê: validar a redução de tempo em produção.)* *(Teste: medição via `time.perf_counter()` em 200 requisições do conjunto de teste ASSIN 2.)*
- **Custo Computacional**
  - **RNF-02.** O volume de tokens gerados na fase de *decode* da Cascata Adaptativa DEVE ser reduzido em no mínimo 60% em relação ao System 2. *(Por quê: demonstrar economia direta de recursos segundo o princípio "Tokens are money".)* *(Teste: auditoria de tokens consumidos registrada pelo vLLM.)*
- **Reprodutibilidade**
  - **RNF-03.** Todo o pipeline de inferência e benchmark DEVE ser executável em uma instância gratuita do Google Colab com GPU NVIDIA T4 (16GB VRAM) sem requerer APIs pagas externas. *(Por quê: garantia de acesso e validação pública no GitHub.)* *(Teste: execução autônoma do notebook Colab do início ao fim.)*
- **Observabilidade**
  - **RNF-04.** O sistema DEVE gerar registros estruturados por requisição contendo `query_id`, `system_1_latency`, `system_1_confidence`, `system_1_entropy`, `routed_to_system_2`, `system_2_latency`, `total_latency` e `final_prediction`. *(Por quê: permitir análise posterior de dispersão e calibragem de limiar.)*

### 3.2 Non-Goals

1. **Serviço de produção em alta disponibilidade (24/7):** O objetivo é um benchmark reproduzível de engenharia e estudo de caso, não um microserviço implantado em Kubernetes.
2. **Fine-tuning ou re-treinamento de modelos:** O projeto foca em estratégias de *serving*, *prompting* e roteamento estatístico com modelos pré-treinados quantizados.
3. **Suporte a dados multimodais:** A avaliação restringe-se a texto em Português do Brasil (tarefa de NLI e classificação categórica de texto).

---

## 4. System Architecture

### 4.1 Visão Geral da Arquitetura

O sistema é estruturado em quatro camadas operacionais independentes: Ingestão de Dados, Serving & Roteamento (vLLM Engine), Avaliação de Performance e Entregáveis.

#### Nível 1 — Contexto do Sistema (C4 Context)

O sistema interage com o avaliador/engenheiro de IA através do carregamento do dataset ASSIN 2, executando requisições no servidor `vLLM` local (GPU T4) e exportando as métricas de latência, custo de tokens e acurácia.

#### Nível 2 — Contêineres e Componentes (C4 Container)

O sistema opera dentro do ambiente Linux/CUDA da GPU NVIDIA T4 no Google Colab, contendo os seguintes componentes internos:

- **Dataset Loader & Formatter**: Consome o dataset ASSIN 2 do Hugging Face Hub e constrói o prompt tipado para o vLLM.
- **vLLM Serving Engine**: Servidor local de inferência em GPU T4 que carrega o modelo quantizado AWQ e disponibiliza os endpoints `/v1/systemone` e `/v1/chat/completions`.
- **System 1 Engine**: Módulo que faz chamadas de 1 token ao `/v1/systemone`, extraindo os *logprobs* dos rótulos válidos.
- **Confidence & Entropy Evaluator**: Componente matemático que calcula a probabilidade normalizada e a entropia dos logits.
- **Cascade Router**: Roteador estatístico que decide a transferência de execução para o System 2.
- **System 2 Engine**: Módulo autorregressivo de *fallback* para raciocínio estruturado (CoT + JSON Schema).
- **Benchmark & Analytics Harness**: Coletor de métricas de tempo, custo em tokens e acurácia.

#### Componentes Principais e Responsabilidades

- **`DatasetLoader`** → Carrega, filtra e formata o dataset ASSIN 2 em pares de entrada estruturados.
- **`vLLMServerManager`** → Gerencia o ciclo de vida e a inicialização do processo servidor `vLLM` em background.
- **`System1Engine`** → Executa chamadas tipadas no endpoint `/v1/systemone` com `max_tokens=1`.
- **`EntropyEvaluator`** → Processa o dicionário de *logprobs* e retorna a distribuição de probabilidade normalizada e a entropia de Shannon.
- **`CascadeRouter`** → Compara a confiança estimada com o limiar $\alpha$ e decide a rota.
- **`System2Engine`** → Invoca o endpoint `/v1/chat/completions` com prompts de *Chain-of-Thought*.
- **`BenchmarkHarness`** → Orquestra as baterias de teste e gera os relatórios estatísticos e gráficos.

#### Estrutura de Pastas do Repositório

```
adaptive-cascade-routing-ptbr/
├── README.md                      # Documentação técnica do repositório e guia de execução
├── requirements.txt               # Dependências Python (vLLM, PyTorch, datasets, pandas, matplotlib)
├── config/
│   └── benchmark_config.yaml      # Parâmetros de modelo, limiares de confiança e portas do vLLM
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py              # Ingestão e formatação do dataset ASSIN 2
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── system1.py             # Interface com /v1/systemone e extração de logprobs
│   │   ├── system2.py             # Interface com /v1/chat/completions (CoT + JSON)
│   │   └── router.py              # Lógica do Roteador de Cascata e cálculo de entropia
│   └── metrics/
│       ├── __init__.py
│       └── evaluator.py           # Coleta de latência (p50/p95/p99), tokens e acurácia
├── notebook/
│   └── adaptive_cascade_benchmark.ipynb # Google Colab Notebook executável ponta a ponta
└── results/
    ├── raw_benchmark_data.json    # Logs detalhados de cada requisição do benchmark
    └── latency_cost_tradeoff.png  # Gráficos de dispersão e Curva de APGR vs Custo
```

---

## 5. Detailed Design

### 5.1 Component A: Dataset Pipeline & Prompt Formatter (`loader.py`)

- **Scope & Responsibility**: Carregar as amostras de teste do dataset ASSIN 2, extrair os campos `premise`, `hypothesis` e `entailment_judgment`, e formatar o estado base (*prompt*) otimizado para reutilização de prefixo no KV cache do vLLM.
- **Interfaces & Contracts**:
  - `load_assin2_test_set(split: str = "test", sample_limit: int = 200) -> List[Dict]`
  - `format_system1_prompt(premise: str, hypothesis: str) -> Dict[str, Any]`
- **Dependencies**: Biblioteca `datasets` do Hugging Face.
- **Core Logic**:
  1. Converte o rótulo original do ASSIN 2 (`NONE` -> `Contradiction`, `ENTAILMENT` -> `Entailment`).
  2. Constrói a mensagem do sistema e o contexto da premissa como prefixo estático para alavancar `--enable-prefix-caching`.

### 5.2 Component B: vLLM Serving Infrastructure (`vLLMServerManager`)

- **Scope & Responsibility**: Subir e monitorar o processo de background do servidor `vLLM` na GPU T4 do Colab.
- **Interfaces & Contracts**:
  - `start_server(model_name: str, host: str = "127.0.0.1", port: int = 8000) -> bool`
  - `health_check() -> bool`
- **Dependencies**: Executável `vllm.entrypoints.openai.api_server`, CUDA 12.x, PyTorch.
- **Core Logic**:
  1. Dispara a linha de comando do servidor com argumentos de otimização (`--model Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq --enable-prefix-caching`).
  2. Executa retentativas em loop contra o endpoint do vLLM até receber resposta HTTP 200.

### 5.3 Component C: System 1 Typed Decision Engine (`system1.py`)

- **Scope & Responsibility**: Realizar a chamada de baixíssima latência ao endpoint especializado `/v1/systemone` do vLLM e extrair o dicionário de probabilidades dos tokens de resposta.
- **Interfaces & Contracts**:
  - `evaluate_typed_decision(state: str, question: str, choices: List[str]) -> System1Output`
- **Data Models**:
  ```python
  from dataclasses import dataclass
  from typing import Dict

  @dataclass
  class System1Output:
      top_choice: str
      top_probability: float
      logprobs: Dict[str, float]
      probabilities: Dict[str, float]
      entropy: float
      latency_ms: float
  ```
- **Core Logic**:
  1. Envia a requisição HTTP POST para o endpoint `/v1/systemone` configurando `max_tokens=1` e especificando a lista de `choices=["Entailment", "Contradiction"]`.
  2. Extrai os log-probabilities $\ln(P)$ retornados pelo motor vLLM no primeiro token gerado.

### 5.4 Component D: Confidence & Entropy Evaluator (`router.py`)

- **Scope & Responsibility**: Calcular a distribuição de probabilidades normalizada via Softmax e calcular a Entropia de Shannon sobre os rótulos de interesse.
- **Mathematical Formulations** (International Standard):
  1. **Logprobs to Probability via Softmax**:
     Given logits or log-probabilities $l_i = \ln(P(y_i \mid x))$ for choices $i \in \{1, \dots, K\}$:
     $$P(y_k \mid x) = \frac{\exp(l_k)}{\sum_{j=1}^{K} \exp(l_j)}$$
  2. **Shannon Entropy**:
     $$\text{Entropy}(P) = -\sum_{k=1}^{K} P(y_k \mid x) \log_2 P(y_k \mid x)$$
  3. **Top Confidence**:
     $$\text{Confidence}(x) = \max_{k} P(y_k \mid x)$$

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
  - `execute_reasoning(premise: str, hypothesis: str) -> System2Output`
- **Prompt Architecture**:
  ```text
  Analise o par de frases abaixo e forneça o raciocínio passo a passo (Chain-of-Thought) antes de determinar o rótulo final.

  Premissa: {premise}
  Hipótese: {hypothesis}

  Retorne um JSON estrito no seguinte formato:
  {
    "reasoning": "Explicação detalhada...",
    "label": "Entailment" | "Contradiction"
  }
  ```
- **Error Handling**: Se a resposta não cumprir o schema JSON esperado, faz a extração de rótulo via regex como plano de contingência (*degradation*).

### 5.7 Component G: Benchmark & Performance Evaluation Harness (`evaluator.py`)

- **Scope & Responsibility**: Medir o desempenho de tempo e custo de forma rigorosa em todas as amostras do benchmark.
- **Metrics Calculated**:
  - **Latência**: $p50$, $p95$, $p99$ medidos via `time.perf_counter()` em milissegundos.
  - **Tokens**: Total de tokens de prefill e de decode (obtidos das métricas de uso da API do vLLM).
  - **Acurácia & F1-Score**: Comparação com os rótulos de *ground truth* do ASSIN 2.
  - **Average Performance Gap Recovered (APGR)**:
    $$\text{APGR}(\alpha) = \frac{\text{Accuracy}(\text{Cascade}_\alpha) - \text{Accuracy}(\text{System 1})}{\text{Accuracy}(\text{System 2}) - \text{Accuracy}(\text{System 1})}$$

---

## 6. Security & Compliance

- **Privacidade de Dados**: O benchmark utiliza exclusivamente o dataset público **ASSIN 2**, isento de informações pessoais identificáveis (PII) ou dados sensíveis de produção.
- **Execução Air-Gapped**: Todo o processamento de inferência ocorre dentro do ambiente do Google Colab no servidor local `127.0.0.1:8000`, sem envio de prompts para APIs de terceiros.
- **Governança de Código Aberto**: Uso de modelos de licença aberta no Hugging Face (Qwen / Gemma) em conformidade com as diretrizes de uso responsável.

---

## 7. Performance & Scalability

- **Prefix Caching Optimization**: Ao prefixar as requisições com a instrução do sistema e os rótulos válidos, a flag `--enable-prefix-caching` do vLLM permite reutilizar os tensores de KV Cache no espaço da GPU, reduzindo o tempo de *prefill* para requisições subsequentes a aproximadamente 5-10ms.
- **Gerenciamento de Memória VRAM**: Utilização de quantização AWQ em 4-bits, limitando a ocupação de VRAM do modelo de 7B a aproximadamente 5.5 GB em uma GPU T4 de 16 GB, reservando o espaço restante para alocação dinâmica do KV Cache.

---

## 8. Testing Strategy

- **Testes Unitários**:
  - Validação da função `format_system1_prompt` com entradas sintéticas.
  - Validação da fórmula de Softmax e cálculo de Entropia com distribuições conhecidas (ex: distribuição uniforme $H(P) = 1.0$ bit vs distribuição determinística $H(P) = 0.0$ bit).
  - Validação da lógica do `CascadeRouter` nos limiares limite $\alpha=0.0$ e $\alpha=1.0$.
- **Teste de Integração (Frozen Eval Set)**:
  - Execução da bateria completa sobre o conjunto congelado de 200 amostras do ASSIN 2 Test Split, salvando os resultados brutos em `results/raw_benchmark_data.json`.

---

## 9. Rollout Plan

1. **Marco 1 — Validação no Colab**: Execução do pipeline completo no Google Colab com medição inicial de latência e acurácia.
2. **Marco 2 — Modularização e Repositório GitHub**: Organização da estrutura de código em módulos Python limpos, inclusão de testes unitários e criação de um `README.md` com os diagramas e tabelas de resultados.
3. **Marco 3 — Hugging Face Space (Gradio)**: Publicação de um Space interativo permitindo a comparação em tempo real de latência entre System 1 e System 2.
4. **Marco 4 — Artigo Técnico no LinkedIn**: Publicação do estudo de caso articulando a redução de latência no p95 em +50% e a economia de tokens de +60%, fundamentado nas referências do iFood e do RouteLLM.

---

## 10. Dependencies, Risks & Open Questions

### Dependências Externas
- `vLLM` com suporte a CUDA 12 e endpoint `/v1/systemone`.
- Instância de GPU NVIDIA T4 no Google Colab.

### Riscos e Mitigações
- **Risco 1: Limite de VRAM no Colab (16GB)**.  
  *Mitigação*: Utilizar modelos quantizados AWQ de 7B ou 9B com `--gpu-memory-utilization 0.90`.
- **Risco 2: Oscilação de latência em ambiente compartilhado no Colab**.  
  *Mitigação*: Ignorar as primeiras 5 requisições de *warmup* e reportar percentis estatísticos ($p50, p95, p99$) agregados em 200 iterações.

### Questões em Aberto
- Qual o valor ideal do limiar $\alpha$ para o dataset ASSIN 2 que maximiza a acurácia enquanto mantém a invocação do System 2 abaixo de 15% das requisições? (Será respondido na varredura empírica do benchmark).

---

## 11. References

1. **iFood Engineering Blog**: *Tokens are money: a strategic guide to cost optimization in LLMs*. (2024).
2. **iFood Engineering Blog**: *Sugar, Revisor, and Pepper: the agentic cycle that maintains iFood's catalog scale and consistency*. (2024).
3. **UC Berkeley & LMSYS (RouteLLM)**: Ong, L., et al. *RouteLLM: Learning to Route LLMs with Preference Data*. arXiv:2406.18665 (2024).
4. **vLLM Documentation**: *Structured Decisions and /v1/systemone Endpoint*. Version 0.32.0.
5. **NILC USP / Hugging Face**: *ASSIN 2: Brazilian Portuguese Natural Language Inference Dataset*. (2020).
6. **Nadir**: *FrugalGPT Cascade Routing: A Step-by-Step Implementation*. (2024).
