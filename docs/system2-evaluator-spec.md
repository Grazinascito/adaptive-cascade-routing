# Feature Branch: `spec/component-f-g-system2-evaluator`

**Created**: 2026-11-07
**Status**: Draft
**Input**: Detailed Design Sections 5.6 (Component F: System 2 Slow-Path Engine - `system2.py`) and 5.7 (Component G: Benchmark & Performance Evaluation Harness - `evaluator.py`) from Design Doc `docs/adaptive-cascade-design-doc.md`.

---

## Component Overview

- **Component F (§5.6): System 2 Slow-Path Engine (`system2.py`)**: Responsible for handling borderline/ambiguous NLI cases by executing Chain-of-Thought (CoT) reasoning and enforcing structured JSON outputs via `/v1/chat/completions`.
- **Component G (§5.7): Benchmark & Performance Evaluation Harness (`evaluator.py`)**: Responsible for executing tests across all 3 scenarios (System 1 Pure, System 2 Pure, Adaptive Cascade), recording per-request latency ($p50$, $p95$, $p99$), counting prefill/decode tokens, evaluating accuracy/F1-score, and computing Average Performance Gap Recovered (APGR).

---

## Stories & Testing *(mandatory)*

### Story 1 - System 2 Chain-of-Thought & JSON Schema Execution (Component F §5.6, Type: functionality, Priority: P1)

Implement the slow-path fallback module (`System2Engine`) that accepts ambiguous/low-confidence queries from the router, prompts the LLM for step-by-step reasoning (Chain-of-Thought), and parses structured JSON output containing the final label (`Entailment` or `Contradiction`).

**Why this priority**: Ensures the cascade router has a functional, high-accuracy fallback mechanism when System 1 uncertainty is high.

**Independent Test**: Execute 20 sample NLI pairs through `execute_reasoning()`. Pass threshold: >= 95% valid JSON outputs parsed without fallback regex, with correct label schema.

**Acceptance Scenarios**:
1. **Given** a premise and hypothesis flagged with high entropy by the router, **When** `execute_reasoning(premise, hypothesis)` is called via `/v1/chat/completions`, **Then** the engine returns a `System2Output` TypedDict containing `reasoning`, `label`, `prompt_tokens` + `completion_tokens`, `latency_ms`, and `fallback_used` (field list per Key Entities below).
2. **Given** a non-compliant JSON response from the LLM, **When** JSON decoding fails, **Then** the system gracefully falls back to regex label extraction (`Contradiction` or `Entailment`) and logs a warning without crashing.

---

### Story 2 - Benchmark Harness & Multi-Scenario Metrics Collector (Component G §5.7, Type: enabler, Priority: P1)

Implement `BenchmarkHarness` to orchestrate execution over the frozen 200-sample test set for all 3 scenarios (System 1 Pure, System 2 Pure, Adaptive Cascade) and log detailed latency, token consumption, and accuracy metrics.

**Why this priority**: Essential for proving the 'Tokens are money' hypothesis and measuring latency/cost reductions with statistical rigor.

**Independent Test**: Run a benchmark dry-run over 10 frozen samples across all 3 scenarios. Verify that `raw_benchmark_data.json` is generated with complete metrics for every query ID.

**Acceptance Scenarios**:
1. **Given** a frozen dataset of 200 ASSIN 2 samples, **When** `run_full_benchmark(confidence_threshold=alpha)` is executed (samples come from the harness constructor), **Then** it records per-request `system_1_latency_ms`, `system_2_latency_ms`, `total_latency_ms`, `tokens_prefill`, `tokens_decode`, `prediction`, and `ground_truth` in a `BenchmarkResult` (full field list per Key Entities below; dry-run writes `results/raw_benchmark_data_dryrun.json`).
2. **Given** the recorded benchmark results, **When** `compute_summary_metrics()` is called, **Then** it outputs $p50$, $p95$, $p99$ latency percentiles, total token counts, F1-Score, Accuracy, and APGR (Average Performance Gap Recovered). The first `warmup_discard` requests (default 5, design doc §10 risk 2) are excluded from the latency percentiles only; accuracy/tokens/rates use every request.

---

### Story 3 - APGR & Trade-off Curve Generator (Component G §5.7, Type: hypothesis, Priority: P2)

Sweep the confidence threshold alpha in [0.0, 1.0] across the benchmark results to evaluate the trade-off curve between latency reduction, token savings, and APGR accuracy recovery. Implemented as a replay: `sweep_alpha(run)` recomputes the routing decision per alpha over the measurements recorded by `run_full_benchmark` — no new HTTP calls (`temperature=0.0` makes the System 2 output deterministic within a session) — and writes `results/latency_cost_tradeoff.png`.

**Why this priority**: Validates RouteLLM's mathematical formulation of optimal decision thresholds.

**Independent Test**: Sweep alpha across 5 values (0.50, 0.70, 0.85, 0.90, 0.95). Pass threshold: verify monotonic relationship between alpha and System 2 call rate.

**Acceptance Scenarios**:
1. **Given** benchmark execution logs across varying alpha, **When** APGR is calculated via $\text{APGR}(\alpha) = \frac{\text{Accuracy}(\text{Cascade}_\alpha) - \text{Accuracy}(\text{System 1})}{\text{Accuracy}(\text{System 2}) - \text{Accuracy}(\text{System 1})}$, **Then** the system identifies the optimal alpha where $\text{APGR} \ge 0.95$ with minimum System 2 invocation rate.

---

### Edge Cases

- **Model Timeout / Failure in System 2**: If System 2 times out or returns HTTP 500, return System 1's top prediction as a graceful fallback and log an error.
- **Extreme Ambiguity ($P_{\text{top}} = 0.50$)**: Router must cleanly escalate to System 2 without infinite loops or division-by-zero errors in APGR calculation when System 1 and System 2 accuracies are equal.
- **Empty or Truncated Output**: System 2 parser handles empty response strings by defaulting to Contradiction with a fallback flag set.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System2Engine (Component F §5.6) MUST construct the prompt from fixed system-level CoT instructions plus the user prompt (premise/hypothesis pair), requiring step-by-step Chain-of-Thought reasoning prior to the final classification. Both parts travel in a single `user` message — the chat contract validated in the Colab gate (>= 95% valid JSON on the 20-pair run); splitting them into separate `system`/`user` messages is a post-merge prefix-caching optimization candidate.
- **FR-002**: System2Engine (Component F §5.6) MUST enforce strict JSON formatting (reasoning and label: Entailment or Contradiction).
- **FR-003**: BenchmarkHarness (Component G §5.7) MUST execute all 3 scenarios (System 1 Pure, System 2 Pure, Adaptive Cascade) on the exact same frozen 200-sample test set to ensure unbiased comparisons.
- **FR-004**: BenchmarkHarness (Component G §5.7) MUST measure latency per request using high-precision timers (`time.perf_counter`).
- **FR-005**: BenchmarkHarness (Component G §5.7) MUST record prefill tokens and decode tokens for each request.
- **FR-006**: BenchmarkHarness (Component G §5.7) MUST calculate accuracy, macro F1-score, $p50/p95/p99$ latencies, and APGR metric for each scenario.

### Non-Functional Requirements *(AI-critical)*

- **NFR-001**: System 2 execution MUST remain bounded (`max_tokens` <= 256) to prevent run-away generation cost during fallback.
- **NFR-002**: Benchmark logging MUST persist raw data to `results/raw_benchmark_data.json` for auditability and post-hoc analysis.

### Key Entities

- **`System2Output`**: TypedDict containing reasoning, label, prompt_tokens, completion_tokens, latency_ms, fallback_used.
- **`BenchmarkResult`**: TypedDict (superset of RNF-04 of the design doc and FR-006 of the engine-router spec) containing `query_id`, `scenario` (`system1_pure` | `system2_pure` | `cascade`), `alpha`, `replayed`, `system_1_latency_ms`, `system_2_latency_ms`, `total_latency_ms`, `confidence`, `entropy`, `routed_to_system_2`, `tokens_prefill`, `tokens_decode`, `prediction`, `ground_truth`, `is_correct`, `system2_fallback_used` (parser level), `system2_error_fallback` (HTTP failure of System 2 in the cascade), and `json_valid` (basis of the SC-001 re-measurement). Tokens of System 1 are always 0 — the `/v1/systemone` response has no `usage`.
- **`SummaryReport`**: TypedDict with the aggregates of one scenario: $p50/p95/p99$ latencies (`latency_percentiles_ms`), total prefill/decode tokens, Accuracy, macro F1-Score, System 2 call rate, APGR, and `json_success_rate`.
- **`KPIReport`**: TypedDict with the SC-002/SC-003/SC-004 verdicts as (passed, measured value) pairs.

## Success Criteria *(mandatory)*

- **SC-001**: System 2 JSON parsing success rate >= 95% on the test set.
- **SC-002**: Benchmark Harness produces valid $p50, p95, p99$ latency percentiles and token counts for 100% of the 200 test samples across all 3 scenarios.
- **SC-003**: Adaptive Cascade achieves $p95$ latency reduction >= 50% vs System 2 Pure while recovering >= 95% APGR accuracy.

## Assumptions

- vLLM chat completions endpoint is healthy and accessible locally.
- Frozen eval set of 200 samples is loaded deterministically from `loader.py`.
