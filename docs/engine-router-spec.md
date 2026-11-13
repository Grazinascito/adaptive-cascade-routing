# Feature Branch: `spec/component-b-c-d-e-engine-router`

**Created**: 2026-11-07
**Status**: Draft
**Input**: Detailed Design Sections 5.2 (Component B: vLLM Server Manager), 5.3 (Component C: System 1 Engine), 5.4 (Component D: Confidence & Entropy Evaluator), and 5.5 (Component E: Adaptive Cascade Router) from Design Doc `docs/adaptive-cascade-design-doc.md`.

---

## Component Overview

- **Component B (§5.2): vLLM Serving Infrastructure (`vLLMServerManager`)**: Launches and monitors the local vLLM background engine with `--enable-prefix-caching`.
- **Component C (§5.3): System 1 Typed Decision Engine (`system1.py`)**: Executes single-token typed decisions against `/v1/systemone`, declaring a `choice` question whose `criteria` options are the NLI labels; the server answers with per-option `probabilities` (softmax over one-token labels).
- **Component D (§5.4): Confidence & Entropy Evaluator (`router.py`)**: `compute_metrics(logprobs, choices)` (composed of `softmax` and `shannon_entropy`) computes Softmax probabilities, top confidence, and Shannon Entropy over candidate choices.
- **Component E (§5.5): Adaptive Cascade Router (`router.py`)**: `route(query, confidence, alpha, system2)` evaluates query confidence against cost threshold $\alpha$ to route between System 1 Fast-Path and System 2 Slow-Path, dispatching to an injected System 2 handler.

`EntropyEvaluator` and `CascadeRouter` are the component names used by the design doc (§5.4/§5.5) and by the `router.py` module docstring. The implemented API follows the `loader.py` convention — data shapes as TypedDicts, behavior in module-level functions.

---

## Stories & Testing *(mandatory)*

### Story 1 - vLLM Server Setup and Prefix Caching Initialization (Component B §5.2, Type: enabler, Priority: P1)

Configure, launch, and monitor the local `vLLM` inference server on an air-gapped GPU environment (NVIDIA T4 16GB VRAM on Linux/CUDA), enabling KV cache reuse via `--enable-prefix-caching` to minimize prefill latency across repeated system prompt contexts.

**Why this priority**: Highest risk reduction — without a stable vLLM server process serving `/v1/systemone`, neither System 1 single-token evaluations nor prefix-cached fast-path execution can take place.

**Independent Test**: Health-check suite executing a request to local port 8000 returning `200 OK`, followed by warmup requests verifying prefix cache reuse through the prefill latency drop of scenario 2 (queries 2–10 >= 70% faster than query 1) — the gate executed in the Colab validation. The vLLM `/metrics` counters (`avg_prompt_throughput`, KV cache reuse) remain an optional diagnostic when Component G extends the notebook.

**Acceptance Scenarios**:
1. **Given** a GPU T4 environment with PyTorch and vLLM installed, **When** `vLLMServerManager.start_server(model_name="Qwen/Qwen3-4B-AWQ", quantization="awq", enable_prefix_caching=True)` is called, **Then** the server process binds to port 8000 within 90 seconds and allocates <= 90% GPU VRAM for model weights and KV cache.
2. **Given** a running vLLM server, **When** 10 sequential queries sharing an identical system prompt prefix are submitted, **Then** prefill latency for queries 2–10 drops by >= 70% compared to query 1 due to prefix cache hits.

---

### Story 2 - Single-Token Typed Decision Inference via `/v1/systemone` (Component C §5.3, Type: functionality, Priority: P1)

Implement the `System1Engine` interface to execute single-token typed decisions against vLLM's `/v1/systemone` Structured Decisions endpoint. The request declares one `choice` question with the candidate labels as `criteria` options (`{"Entailment": null, "Contradiction": null}`) and the fixed system prompt in the top-level `instructions` field (identical across queries, for prefix caching); the server maps each option to a one-token label, applies softmax server-side, and returns per-option `probabilities` in the answer. There are no `max_tokens`/`logprobs`/`choices` request fields — unknown fields are rejected with HTTP 400 — so the engine derives logprobs $\ln(P)$ from the returned probabilities and feeds them to `compute_metrics` (Component D).

**Why this priority**: Essential functionality — forms the Fast-Path mechanism that eliminates autoregressive decode iterations, achieving sub-60ms response times for high-confidence items.

**Independent Test**: Deterministic execution against 50 validation samples from ASSIN 2, verifying that 100% of responses contain `probabilities` for the target choices and execute with latency $p95 < 60\text{ms}$.

**Acceptance Scenarios**:
1. **Given** a formatted NLI prompt (`state`), **When** `System1Engine.evaluate_typed_decision(state, choices=["Entailment", "Contradiction"])` is invoked, **Then** the engine receives an HTTP response containing per-option `probabilities` for both candidate labels (from which $\ln(P)$ logprobs are derived) without any autoregressive decoding.
2. **Given** invalid or out-of-vocabulary candidate choices, **When** `evaluate_typed_decision` is called, **Then** the engine raises a `ValueError` with clear diagnostic context before dispatching the HTTP call.

---

### Story 3 - Softmax Normalization, Confidence Scoring, and Shannon Entropy Evaluation (Component D §5.4, Type: functionality, Priority: P2)

Implement the mathematical evaluation functions in `router.py` — `softmax(logprobs, choices)`, `shannon_entropy(probabilities)`, and `compute_metrics(logprobs, choices)` — to convert raw logprobs into a normalized probability distribution $P$, top confidence $\text{Confidence}(x)$, and Shannon Entropy $\text{Entropy}(P)$. `compute_metrics` returns a `DecisionMetrics` record (`top_choice`, `confidence`, `entropy`, `probabilities`).

**Why this priority**: Core mathematical engine — provides the uncertainty signals required by the Adaptive Cascade Router to distinguish unambiguous items from ambiguous edge cases.

**Independent Test**: Unit test suite verifying mathematical correctness against synthetic logprob inputs with known analytical solutions (e.g., uniform distribution yields maximum entropy $H(P) = 1.0$ bit; deterministic distribution yields $H(P) = 0.0$ bit).

**Acceptance Scenarios**:
1. **Given** logprobs `{"Entailment": ln(0.9), "Contradiction": ln(0.1)}` and `choices=["Entailment", "Contradiction"]`, **When** `compute_metrics(logprobs, choices)` is executed, **Then** it returns $\text{Confidence} = 0.9000 \pm 10^{-4}$ and $\text{Entropy} \approx 0.4690$ bits.
2. **Given** equal logprobs `{"Entailment": ln(0.5), "Contradiction": ln(0.5)}`, **When** `compute_metrics(logprobs, choices)` is executed, **Then** it returns $\text{Confidence} = 0.5000$ and maximum Shannon Entropy $\text{Entropy} = 1.0000$ bit, with `top_choice = "Entailment"` (ties break by the first label in `choices` order).

---

### Story 4 - Adaptive Cascade Routing and Fallback Decision (Component E §5.5, Type: hypothesis, Priority: P2)

Implement `route(query, confidence, alpha, system2)` in `router.py`, following Berkeley's RouteLLM framework $R_\alpha(q)$, to route high-confidence queries ($\text{Confidence} \ge \alpha$) to the System 1 fast-path without invoking System 2, and low-confidence queries ($\text{Confidence} < \alpha$) to a System 2 fallback handler received by injection (`System2Handler = Callable[[RoutingQuery], Label]`). `route` returns a `RouteOutcome` record (`route`, `routed_to_system_2`, `system2_prediction`).

**Why this priority**: The cornerstone of the project — tests the hypothesis that routing queries with $\text{Confidence} < \alpha$ to System 2 recovers >= 95% of System 2 standalone accuracy while keeping System 2 invocation rate below 20%.

**Independent Test**: Execution over the frozen 200-sample ASSIN 2 test set across thresholds $\alpha \in \{0.50, 0.70, 0.85, 0.95\}$, verifying monotonic trade-offs between System 2 call rate, $p95$ latency, and overall F1-Score.

**Acceptance Scenarios**:
1. **Given** a query with $\text{Confidence} = 0.92$ and threshold $\alpha = 0.85$, **When** `route(query, confidence=0.92, alpha=0.85, system2=handler)` is called, **Then** it returns `route = "SYSTEM_1"`, `routed_to_system_2 = False`, `system2_prediction = None`, and completes without invoking the System 2 handler.
2. **Given** a query with $\text{Confidence} = 0.61$ and threshold $\alpha = 0.85$, **When** `route(query, confidence=0.61, alpha=0.85, system2=handler)` is called, **Then** it returns `route = "SYSTEM_2"`, `routed_to_system_2 = True`, and dispatches the `RoutingQuery` payload (`premise` + `hypothesis`) to the injected handler, returning its `Label` as `system2_prediction`.

---

### Edge Cases

- **Model Logprob Omission**: What happens if vLLM fails to return logprobs for one of the candidate tokens in the choices list?  
  *Handling*: `softmax` assigns the module constant `LOGPROB_FLOOR = -100.0` ($P \approx 0$) to candidates missing from the logprobs dict and normalizes over `choices`. Logprob keys outside `choices` are ignored, and an input carrying no logprob for any choice ("no signal") raises `ValueError`.
- **Extreme Uniform Uncertainty ($P_{\text{top}} = 0.5000$)**: How does the system handle exact 50/50 ties between `Entailment` and `Contradiction`?  
  *Handling*: `compute_metrics` breaks probability ties deterministically by the first label in `choices` order; any query with $\text{Confidence} < \alpha$ automatically triggers System 2 fallback, so at $\alpha = 0.85$ exact ties are routed to System 2 100% of the time.
- **vLLM Process Crash / OOM**: How does `vLLMServerManager` handle GPU Out-Of-Memory errors during heavy KV cache allocation?  
  *Handling*: Catch connection timeouts/refusals, log structured failure event with GPU VRAM stats, and throw `vLLMServerException` with restart instructions.
- **Malformed System 2 JSON Output**: What happens if System 2 fallback returns non-JSON or unparseable text?  
  *Handling*: Fall back to regular-expression label extraction (`r"Entailment|Contradiction"`) to recover the prediction without crashing the evaluation run.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST start and manage a local `vLLM` server instance listening on port 8000 with AWQ 4-bit quantization, `--enable-prefix-caching`, and `--logprobs-mode raw_logprobs` (the flag the `/v1/systemone` API requires to serve per-option probabilities) (Component B §5.2).
- **FR-002**: System MUST expose `System1Engine.evaluate_typed_decision()` to send HTTP POST requests to `/v1/systemone` declaring a `choice` question whose `criteria` options are `["Entailment", "Contradiction"]` (Component C §5.3). Per-option `probabilities` in the response are converted to logprobs $\ln(P)$ and fed to `compute_metrics` (FR-003).
- **FR-003**: System MUST compute Softmax-normalized probabilities $P(y_k \mid x) = \frac{\exp(l_k)}{\sum_{j=1}^K \exp(l_j)}$ from raw candidate logprobs (Component D §5.4).
- **FR-004**: System MUST compute Shannon Entropy $\text{Entropy}(P) = -\sum_{k=1}^K P(y_k \mid x) \log_2 P(y_k \mid x)$ and Top Confidence $\text{Confidence}(x) = \max_k P(y_k \mid x)$ for every query (Component D §5.4).
- **FR-005**: System MUST implement the RouteLLM decision function $R_\alpha(q)$ (Component E §5.5):
  $$R_\alpha(q) = \begin{cases} \text{System 1 (Fast-Path)}, & \text{if } \text{Confidence}(q) \ge \alpha \\ \text{System 2 (Slow-Path)}, & \text{if } \text{Confidence}(q) < \alpha \end{cases}$$
- **FR-006**: System MUST return a structured record for every request — `BenchmarkResult` (Component G, `src/metrics/summary.py`), composed by the benchmark harness from `RouteOutcome` plus the engine outputs — logging `query_id`, `system_1_latency_ms`, `confidence`, `entropy`, `routed_to_system_2`, `system_2_latency_ms`, `total_latency_ms`, `prediction`, and `ground_truth` (plus prefill/decode tokens and the parser/error fallback flags; canonical field list in the system2-evaluator spec Key Entities).

### Non-Functional Requirements *(AI-critical)*

- **NFR-001 (Latency Fast-Path)**: System 1 execution latency MUST achieve $p95 < 60\text{ms}$ per request under prefix-cached execution on NVIDIA T4.
- **NFR-002 (Memory Overhead)**: Total GPU VRAM consumption for vLLM model weights + KV Cache MUST NOT exceed 14.4 GB (90% of T4 16GB capacity).
- **NFR-003 (Deterministic Routing)**: The Cascade Router MUST execute in $O(K)$ time where $K$ is the number of choices ($K=2$), adding $< 1\text{ms}$ overhead to the decision path.
- **NFR-004 (Air-Gapped Operation)**: All inference HTTP requests MUST be routed exclusively to local localhost port 8000 with zero external network dependencies.

### Key Entities

- **`System1Output`**: TypedDict holding top choice (`Label`), top probability, the derived logprobs dict ($\ln$ of the server probabilities), normalized probabilities dict, Shannon entropy, and latency in ms.
- **`System2Output`**: TypedDict holding Chain-of-Thought reasoning string, extracted label, prompt/completion token counts, latency in ms, and the `fallback_used` flag (canonical field list in `system2-evaluator-spec` Key Entities).
- **`DecisionMetrics`**: TypedDict returned by `compute_metrics` holding `top_choice`, top `confidence`, Shannon `entropy` in bits, and the normalized `probabilities` dict.
- **`RouteOutcome`**: TypedDict returned by `route` holding the `route` decision (`"SYSTEM_1"` or `"SYSTEM_2"`), the `routed_to_system_2` flag, and the `system2_prediction` (`None` on the fast-path).
- **`BenchmarkResult`** (was specified as `CascadeResult`): Unified record of a single benchmark item execution containing all System 1 metrics, routing flags, System 2 metrics (if invoked), and final decision vs ground truth. Implemented in Component G (`src/metrics/summary.py`); the router itself returns `RouteOutcome` and the harness composes the unified record.

---

## Success Criteria *(mandatory)*

- **SC-001 (System 1 Speed)**: System 1 Fast-Path achieves $p95 \le 60\text{ms}$ latency across the 200-sample test set.
- **SC-002 (Cascade Latency Reduction)**: Adaptive Cascade (at calibrated $\alpha = 0.85$) reduces total $p95$ latency by $\ge 50\%$ compared to System 2 Standalone.
- **SC-003 (Accuracy Recovery - APGR)**: Adaptive Cascade recovers $\ge 95\%$ of System 2 standalone accuracy ($\text{APGR} \ge 0.95$) on the frozen 200-sample ASSIN 2 test set.
- **SC-004 (Decode Token Reduction)**: Adaptive Cascade reduces total generated decode tokens by $\ge 60\%$ compared to System 2 Standalone.

---

## Assumptions

- Assumes vLLM version >= 0.32.0 with the `/v1/systemone` Structured Decisions API installed in the execution environment and served with `--logprobs-mode raw_logprobs`.
- Assumes the underlying GPU model (`Qwen/Qwen3-4B-AWQ`, the official 4-bit AWQ of Qwen3-4B — the 2507 Instruct series has no official AWQ on the Hugging Face Hub) fits within 16GB VRAM alongside the KV cache allocated by vLLM. The original `Qwen/Qwen2.5-7B-Instruct-AWQ` returned `501 Not Implemented` on `/v1/systemone`: the Structured Decisions API requires an architecture with a registered read strategy (`Qwen3ForCausalLM` has one; Qwen2.5 does not), and no flag enables it. Qwen3 'original' models think by default in the chat template, so System 2 disables it per request via `chat_template_kwargs: {"enable_thinking": false}` (`/v1/systemone` keeps thinking off by itself).
- Assumes the frozen 200-sample test set from ASSIN 2 contains balanced representations of `Entailment` and `Contradiction` labels.
