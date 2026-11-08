# Adaptive Cascade Routing for Typed LLM Inference: System 1 (Logprobs) vs System 2 (CoT)

[![CI](https://github.com/Grazinascito/adaptive-cascade-routing/actions/workflows/ci.yml/badge.svg)](https://github.com/Grazinascito/adaptive-cascade-routing/actions/workflows/ci.yml)

Empirical inference engineering benchmark evaluating latency, token cost, and accuracy trade-offs across three execution scenarios for Brazilian Portuguese (PT-BR) Natural Language Inference (NLI): **System 1 Pure**, **System 2 Pure**, and **Adaptive Cascade Routing**.

## Executive Overview

In large-scale production environments—such as product catalog categorization at iFood or intent classification in RAG pipelines—unconditionally routing every categorical decision through long-form, autoregressive Large Language Models introduces severe memory bandwidth bottlenecks and unnecessary financial expenditure.

Inspired by iFood's engineering architecture (**Sugar, Revisor & Pepper**) and grounded in dynamic model routing theory (**RouteLLM** by UC Berkeley / LMSYS and **FrugalGPT**), this repository implements an adaptive cascade system running on top of a single local vLLM serving instance (NVIDIA T4 GPU on Google Colab).

### The 3 Evaluated Scenarios

The benchmark executes an apples-to-apples comparison across a frozen 200-sample test set from the ASSIN 2 dataset (nilc-nlp/assin2):

1. **System 1 Pure (Fast-Path / Typed Decision)**: Evaluates input states via vLLM's specialized /v1/systemone endpoint (Structured Decisions API) with prefix caching enabled. Declares a `choice` question whose options map to one-token labels; the server answers with per-option probabilities (server-side softmax), eliminating autoregressive decoding latency (~40-60ms p95).
2. **System 2 Pure (Slow-Path / Autoregressive CoT)**: Routes queries through /v1/chat/completions, enforcing Chain-of-Thought (CoT) reasoning and strict JSON schema output (max_tokens=256), serving as the baseline for maximum reasoning accuracy.
3. **Adaptive Cascade (Entropy/Confidence Router)**: Calculates top-choice probability $P_{\text{top}}$ and Shannon Entropy $H(P)$ over logprobs. Accepts System 1 predictions when $P_{\text{top}} \geq \alpha$ ($\alpha \approx 0.85$); escalates to System 2 only when statistical uncertainty is high.

## System Architecture

- **Dataset Layer**: Frozen 200-sample test set from ASSIN 2 (seed=42).
- **Serving Engine Layer**: Single local vLLM server with --enable-prefix-caching and --logprobs-mode raw_logprobs.
- **System 1 Engine**: Fast-path evaluation via /v1/systemone choice questions (per-option probabilities).
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
├── .github/workflows/ci.yml         # CI: ruff + mypy + pytest on Python 3.12
├── .gitignore                       # Python caches, .venv; keeps uv.lock and results/
├── .python-version                  # "3.12" (read by uv)
├── LICENSE                          # MIT
├── CONTEXT.md                       # Project glossary (PT-BR)
├── README.md                        # Project documentation and benchmark overview
├── pyproject.toml                   # Dependencies (uv), "serving" extra (vLLM), ruff/mypy/pytest config
├── requirements.txt                 # Core deps exported from uv.lock (for pip)
├── requirements-colab.txt           # Core + vLLM exported from uv.lock (for Colab)
├── uv.lock                          # Committed lockfile (benchmark reproducibility)
├── config/
│   └── benchmark_config.yaml        # Frozen benchmark contract (model, α, seed, endpoints, KPIs)
├── docs/
│   ├── adaptive-cascade-design-doc.md   # Full design doc (PT-BR)
│   ├── dataset-loader-spec.md           # Component A stories & testing
│   ├── engine-router-spec.md            # Components B–E stories & testing
│   ├── system2-evaluator-spec.md        # Components F–G stories & testing
│   └── adr/                             # Architecture decision records
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py                # ASSIN 2 dataset loader and prompt formatter
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── server.py                # vLLM server lifecycle (vLLMServerManager)
│   │   ├── system1.py               # Interface with /v1/systemone (choice questions; per-option probabilities)
│   │   ├── system2.py               # Interface with /v1/chat/completions (CoT + JSON)
│   │   └── router.py                # Softmax, Shannon entropy, and RouteLLM cascade logic
│   └── metrics/
│       ├── __init__.py
│       └── evaluator.py             # Benchmark harness (p50/p95/p99 latency, tokens, APGR)
├── notebook/
│   └── adaptive_cascade_benchmark.ipynb # Google Colab end-to-end executable notebook (Marco 1)
├── results/
│   ├── raw_benchmark_data.json      # Request-level execution logs (Marco 1)
│   └── latency_cost_tradeoff.png    # APGR vs token savings trade-off visualization (Marco 1)
└── tests/
    ├── test_config.py               # Frozen benchmark contract (config keys and values)
    └── test_structure.py            # All skeleton modules import without error
```

## Quickstart & Setup

### Prerequisites

- Python 3.12+
- Local development: any OS (client code only; vLLM is never installed locally)
- Serving & benchmark execution: Google Colab with GPU NVIDIA T4 (16GB VRAM) or a local CUDA 12.x environment

### Local development (uv)

```bash
git clone https://github.com/Grazinascito/adaptive-cascade-routing.git
cd adaptive-cascade-routing
uv sync --dev
uv run pytest
```

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + committed
`uv.lock`); vLLM lives in the `serving` extra and is only installed on CUDA machines
(`uv sync --extra serving`). See [docs/adr/0001-deps-uv-vllm-separado-colab.md](docs/adr/0001-deps-uv-vllm-separado-colab.md).

### Colab / CUDA environment (pip, includes vLLM)

```bash
pip install -r requirements-colab.txt
```

On Google Colab (GPU NVIDIA T4), open [notebook/adaptive_cascade_benchmark.ipynb](notebook/adaptive_cascade_benchmark.ipynb) — its Setup cell clones this repo and installs the dependencies above; the remaining cells validate Components B (server + prefix cache), C (System 1 typed decisions) and F (System 2 CoT+JSON).

### Running the vLLM Server & Benchmark

1. **Start the local vLLM server** (with prefix caching enabled; `--logprobs-mode raw_logprobs` is required by the `/v1/systemone` Structured Decisions API):

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507-AWQ \
    --quantization awq \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.90 \
    --logprobs-mode raw_logprobs \
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
