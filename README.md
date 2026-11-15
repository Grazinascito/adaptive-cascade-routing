# Adaptive Cascade Routing for Typed LLM Inference: System 1 (Logprobs) vs System 2 (CoT)

[![CI](https://github.com/Grazinascito/adaptive-cascade-routing/actions/workflows/ci.yml/badge.svg)](https://github.com/Grazinascito/adaptive-cascade-routing/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Grazinascito/adaptive-cascade-routing/blob/005-dataset-pivot-ncm/notebook/adaptive_cascade_benchmark.ipynb)
[![Dataset](https://img.shields.io/badge/🤗_Dataset-NCM--RAFT--samall-FFD21F)](https://huggingface.co/datasets/yurifacanha/NCM-RAFT-samall)
[![Model](https://img.shields.io/badge/🤗_Model-Qwen3--4B--AWQ-FFD21F)](https://huggingface.co/Qwen/Qwen3-4B-AWQ)

Empirical inference engineering benchmark evaluating latency, token cost, and accuracy trade-offs across three execution scenarios for Brazilian catalog item classification into **NCM codes** (Mercosur Common Nomenclature — the Brazilian fiscal taxonomy tree): **System 1 Pure**, **System 2 Pure**, and **Adaptive Cascade Routing**.

## Executive Overview

The project thesis: the cost gains of iFood's hybrid architecture (engineering blog: *Sugar, Revisor & Pepper*) are not exclusive to the small-model × large-model axis — they also appear on the System 1 × System 2 axis **of the same model**, provided the task has the ambiguous long-tail content the blog describes. The NCM dataset provides exactly that: real catalog item names ("ALCOOL LIQUIDO START NAT 70% - 5VL") mapping to codes in a 4-level taxonomy tree, with frequent codes (head) and rare, ambiguous ones (tail).

Grounded in dynamic model routing theory (**RouteLLM** by UC Berkeley / LMSYS and **FrugalGPT**), this repository implements the adaptive cascade on a single local vLLM serving instance (NVIDIA T4 GPU on Google Colab). Historical note: the first measured iteration of this harness ran on the ASSIN 2 NLI dataset (commit `79a8654`); it showed that without an ambiguous tail the cascade degenerates to "always System 1" — the measurement that motivated this pivot.

### The 3 Evaluated Scenarios

The benchmark executes an apples-to-apples comparison across a frozen 800-sample set (65% head / 35% tail by code frequency, seed=42):

1. **System 1 Pure (Fast-Path / Short Generation)**: `/v1/chat/completions` with `max_tokens=12`, no reasoning — the model emits the 8-digit code directly. Confidence is the geometric mean of the generated tokens' probabilities (`exp(Σ ln p / n)`); a code outside the taxonomy gets confidence 0.0 and escalates (the "new case → fallback" pattern of the blog).
2. **System 2 Pure (Slow-Path / Autoregressive CoT)**: Chain-of-Thought reasoning plus strict JSON output (`max_tokens=256`), with the taxonomy chapter list as prompt context — the baseline for maximum reasoning accuracy.
3. **Adaptive Cascade (Confidence Router)**: accepts System 1 predictions when confidence ≥ α (α ≈ 0.85); escalates only statistically ambiguous items to System 2, following RouteLLM's formulation.

## System Architecture

- **Dataset Layer**: `yurifacanha/NCM-RAFT-samall` (105k rows, PT-BR, conversational format) parsed into (item name, NCM code) pairs; taxonomy of 114 codes / 19 chapters committed at `config/ncm_taxonomy.csv`.
- **Serving Engine Layer**: single local vLLM server with `--enable-prefix-caching`.
- **System 1 Engine**: short generation with per-token logprobs (confidence via geometric mean; entropy over the first token's top-logprobs).
- **Router Layer**: confidence vs threshold α (RouteLLM formulation); System 2 handler injected.
- **System 2 Engine**: CoT + JSON via `/v1/chat/completions`, regex/default fallback chain on malformed output.
- **Metrics Harness**: p50/p95/p99 latencies, prefill/decode tokens (both engines), accuracy overall and per subset (head/tail), macro F1, APGR (informational), and the pivot gates below.

## Mathematical Formulations

### 1. Confidence of the fast path (geometric mean of token probabilities)

$$
\text{Confidence}(x) = \exp\!\Big(\frac{1}{n}\sum_{i=1}^{n} \ln p_i\Big)
$$

where $p_i$ are the probabilities of the $n$ generated tokens (the code digits). Less punitive than the raw product for 8-digit codes.

### 2. Shannon Entropy (first generated token)

$$
\mathrm{Entropy}(P) = - \sum_{k} P(y_k \mid x) \, \log_2 P(y_k \mid x)
$$

computed over the normalized top-logprobs distribution of the first token — where the model "decides" how the code starts.

### 3. Cascade Routing Function R_alpha(q) (RouteLLM Formulation)

$$
R_\alpha(q) =
\begin{cases}
\text{System 1 (Fast-Path)}, & \text{Confidence}(q) \geq \alpha \\
\text{System 2 (Slow-Path)}, & \text{caso contrário}
\end{cases}
$$

### 4. Average Performance Gap Recovered (APGR, informational)

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
├── requirements-colab.txt           # Core + vLLM exported from uv.lock (for the Colab)
├── uv.lock                          # Committed lockfile (benchmark reproducibility)
├── config/
│   ├── benchmark_config.yaml        # Frozen benchmark contract (dataset, α, seed, gates)
│   └── ncm_taxonomy.csv             # NCM taxonomy derived from the dataset (114 codes)
├── docs/
│   ├── adaptive-cascade-design-doc.md   # Full design doc (PT-BR)
│   ├── dataset-pivot-ncm-spec.md        # Pivot spec: Stories 1–6 (this task)
│   ├── dataset-loader-spec.md           # Component A stories & testing (NCM)
│   ├── engine-router-spec.md            # Components B–E stories & testing (NCM)
│   ├── system2-evaluator-spec.md        # Components F–G stories & testing (NCM)
│   └── adr/                             # Architecture decision records
├── src/
│   ├── data/
│   │   └── loader.py                # NCM parser, taxonomy, frozen head/tail sampling
│   ├── engine/
│   │   ├── server.py                # vLLM server lifecycle (vLLMServerManager)
│   │   ├── system1.py               # Short classification with logprobs confidence
│   │   ├── system2.py               # CoT + JSON NCM classification
│   │   └── router.py                # Shannon entropy + RouteLLM cascade decision
│   └── metrics/
│       ├── summary.py               # BenchmarkResult/SummaryReport + pure metrics (subsets, gates)
│       └── evaluator.py             # Benchmark harness, alpha sweep, results/ artifacts, CLI
├── notebook/
│   └── adaptive_cascade_benchmark.ipynb # Colab notebook: probe + B/C/F/G validation
└── tests/                           # Mirrors src/ layout; httpx.MockTransport at the network edge
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

On Google Colab (GPU NVIDIA T4), open [notebook/adaptive_cascade_benchmark.ipynb](notebook/adaptive_cascade_benchmark.ipynb) — its Setup cell clones this repo and installs the dependencies above. The probe cell loads the pool, prints extraction statistics and decides the label granularity (`LABEL_DIGITS`); the remaining cells validate Components B (server + prefix cache), C (System 1 short classification), F (System 2 CoT+JSON) and G (full benchmark + gates).

### Running the vLLM Server & Benchmark

1. **Start the local vLLM server** (with prefix caching enabled):

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-AWQ \
    --quantization awq \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.90 \
    --port 8000
```

2. **Execute the full benchmark suite** (3 scenarios over the frozen 800-sample set at the config's default alpha, plus the alpha sweep replay; writes `results/raw_benchmark_data.json` and `results/latency_cost_tradeoff.png`, and prints each gate with a PASS/FAIL verdict):

```bash
python -m src.metrics.evaluator --config config/benchmark_config.yaml
```

Use `--limit 12` for a dry-run.

## Target Success Criteria (pivot gates)

- **SC-A (tail gap)**: System 2 Pure − System 1 Pure accuracy ≥ 10 percentage points on the tail subset — the premise that reasoning pays where items are ambiguous.
- **SC-B (recovery)**: the cascade recovers ≥ 50% of the tail gap and never falls below System 1 Pure overall accuracy.
- **SC-C (cost)**: p95 latency ≥ 50% lower and decode tokens ≥ 60% lower than System 2 Pure (token counts include the fast path — "tokens are money").
- **Reported alongside**: the fallback rate at the optimal α, compared against the 30–50% band reported by the iFood blog.

A gate that fails is reported as FAIL — a measurement, not an error (the honest-FAIL precedent was set by the ASSIN 2 iteration: commit `79a8654`).

## References & Acknowledgments

1. iFood Engineering Blog: Tokens are money: a strategic guide to cost optimization in LLMs (2024).
2. iFood Engineering Blog: Sugar, Revisor, and Pepper: the agentic cycle that maintains iFood's catalog scale and consistency (2024).
3. UC Berkeley & LMSYS (RouteLLM): RouteLLM: Learning to Route LLMs with Preference Data (2024).
4. vLLM Documentation: OpenAI-Compatible Server & Chat Completions logprobs (2024).
5. Hugging Face: yurifacanha/NCM-RAFT-samall — NCM classification dataset, PT-BR (2024).
