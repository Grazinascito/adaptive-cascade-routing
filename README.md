# Adaptive Cascade Routing for Typed LLM Inference: System 1 (Logprobs) vs System 2 (CoT)

Empirical inference engineering benchmark evaluating latency, token cost, and accuracy trade-offs across three execution scenarios for Brazilian Portuguese (PT-BR) Natural Language Inference (NLI): **System 1 Pure**, **System 2 Pure**, and **Adaptive Cascade Routing**.

## Executive Overview

In large-scale production environments—such as product catalog categorization at iFood or intent classification in RAG pipelines—unconditionally routing every categorical decision through long-form, autoregressive Large Language Models introduces severe memory bandwidth bottlenecks and unnecessary financial expenditure.

Inspired by iFood's engineering architecture (**Sugar, Revisor & Pepper**) and grounded in dynamic model routing theory (**RouteLLM** by UC Berkeley / LMSYS and **FrugalGPT**), this repository implements an adaptive cascade system running on top of a single local vLLM serving instance (NVIDIA T4 GPU on Google Colab).

### The 3 Evaluated Scenarios

The benchmark executes an apples-to-apples comparison across a frozen 200-sample test set from the ASSIN 2 dataset (nilc-nlp/assin2):

1. **System 1 Pure (Fast-Path / Typed Decision)**: Evaluates input states via vLLM's specialized /v1/systemone endpoint with prefix caching enabled. Inspects token log-probabilities (logprobs) to emit exactly 1 token (max_tokens=1), eliminating autoregressive decoding latency (~40-60ms p95).
2. **System 2 Pure (Slow-Path / Autoregressive CoT)**: Routes queries through /v1/chat/completions, enforcing Chain-of-Thought (CoT) reasoning and strict JSON schema output (max_tokens=256), serving as the baseline for maximum reasoning accuracy.
3. **Adaptive Cascade (Entropy/Confidence Router)**: Calculates top-choice probability $P_{\text{top}}$ and Shannon Entropy $H(P)$ over logprobs. Accepts System 1 predictions when $P_{\text{top}} \geq \alpha$ ($\alpha \approx 0.85$); escalates to System 2 only when statistical uncertainty is high.

## System Architecture

- **Dataset Layer**: Frozen 200-sample test set from ASSIN 2 (seed=42).
- **Serving Engine Layer**: Single local vLLM server with --enable-prefix-caching.
- **System 1 Engine**: Fast-path evaluation via /v1/systemone generating 1 token.
- **Router Layer**: Evaluates top probability $P_{\text{top}}$ vs threshold $\alpha$.
- **System 2 Engine**: Fallback execution via /v1/chat/completions autoregression producing Chain-of-Thought reasoning and strict JSON outputs.
- **Metrics Harness**: Captures p50, p95, p99 latencies, token consumption, and APGR accuracy.

## Mathematical Formulations

Following international standards and Berkeley's RouteLLM formulation:

### 1. Logprobs to Softmax Probability

Given log-probabilities $l_k$ for valid candidate labels:

$$
P(y_k \mid x) = \frac{\exp(l_k)}{\sum_{j} \exp(l_j)}
$$

### 2. Shannon Entropy

$$
\mathrm{Entropy}(P) = - \sum_{k} P(y_k \mid x) \, \log_2\!\big(P(y_k \mid x)\big)
$$

### 3. Cascade Routing Function R_alpha(q) (RouteLLM Formulation)

$$
R_\alpha(q) =
\begin{cases}
\text{System 1 (Fast-Path)}, & \max(P) \geq \alpha \\
\text{System 2 (Slow-Path)}, & \text{caso contrário}
\end{cases}
$$

### 4. Average Performance Gap Recovered (APGR)

$$
\mathrm{APGR}(\alpha) = \frac{\mathrm{Acc}(\text{Cascade}) - \mathrm{Acc}(\text{System 1})}{\mathrm{Acc}(\text{System 2}) - \mathrm{Acc}(\text{System 1})}
$$

## Repository Structure

```
adaptive-cascade-routing/
├── README.md                      # Project documentation and benchmark overview
├── requirements.txt               # Python dependencies (vLLM, PyTorch, datasets, pandas)
├── config/
│   └── benchmark_config.yaml      # Model parameters, router thresholds, and vLLM ports
├── src/
│   ├── data/
│   │   ├── loader.py              # ASSIN 2 dataset loader and prompt formatter
│   ├── engine/
│   │   ├── system1.py             # Interface with /v1/systemone (logprobs extraction)
│   │   ├── system2.py             # Interface with /v1/chat/completions (CoT + JSON)
│   │   └── router.py              # Softmax, Shannon entropy, and RouteLLM cascade logic
│   └── metrics/
│       └── evaluator.py           # Benchmark harness (p50/p95/p99 latency, tokens, APGR)
├── notebook/
│   └── adaptive_cascade_benchmark.ipynb # Google Colab end-to-end executable notebook
└── results/
    ├── raw_benchmark_data.json    # Request-level execution logs
    └── latency_cost_tradeoff.png  # APGR vs token savings trade-off visualization
```

## Quickstart & Setup

### Prerequisites

- Python 3.12+
- Google Colab with GPU NVIDIA T4 (16GB VRAM) or local CUDA 12.x environment

### Installation

```bash
git clone https://github.com/<your-username>/adaptive-cascade-routing.git
cd adaptive-cascade-routing
pip install -r requirements.txt
```

### Running the vLLM Server & Benchmark

1. **Start the local vLLM server** (with prefix caching enabled):

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq \
    --enable-prefix-caching \
    --port 8000
```

2. **Execute the full benchmark suite**:

```bash
python -m src.metrics.evaluator --config config/benchmark_config.yaml
```

## Target Success Criteria (KPIs)

- **p95 Latency Reduction**: $\geq 50\%$ reduction in p95 response time for Adaptive Cascade vs System 2 Pure.
- **Token Cost Savings**: $\geq 60\%$ reduction in total decode tokens consumed.
- **APGR Accuracy Recovery**: $\geq 95\%$ of System 2 accuracy retained via adaptive escalation.

## References & Acknowledgments

1. iFood Engineering Blog: Tokens are money: a strategic guide to cost optimization in LLMs (2024).
2. iFood Engineering Blog: Sugar, Revisor, and Pepper: the agentic cycle that maintains iFood's catalog scale and consistency (2024).
3. UC Berkeley & LMSYS (RouteLLM): RouteLLM: Learning to Route LLMs with Preference Data (2024).
4. vLLM Documentation: Structured Decisions and /v1/systemone Endpoint (2024).
5. NILC USP / Hugging Face: ASSIN 2: Brazilian Portuguese Natural Language Inference Dataset (2020).
