# Spec: Component A — Dataset Pipeline & Prompt Formatter (`src/data/loader.py`)

**Feature Branch**: `001-component-a-dataset-loader`  
**Created**: 2026-11-07  
**Status**: Draft  
**Input**: Description: "Dataset ingestion, label normalization, prompt formatting for vLLM Prefix Caching, and frozen test set sampling for ASSIN 2 PT-BR benchmark"

---

## Stories & Testing *(mandatory)*

### Story 1 - ASSIN 2 Dataset Ingestion and Label Normalization (Type: ENABLER, Priority: P1)

Implement data loading from Hugging Face Datasets (`nilc-nlp/assin2`) and map textual entailment labels into binary classification categories (`Entailment` vs `Contradiction`).

**Why this priority**: Highest risk reduction — without verified, deterministic, and clean PT-BR dataset pairs, downstream System 1 and System 2 inference engines cannot execute or be evaluated against ground truth.

**Independent Test**: Unit test with `pytest` checking dataset loading, schema validation, and 100% label mapping correctness across all loaded samples.

**Acceptance Scenarios**:

1. **Given** the Hugging Face `nilc-nlp/assin2` dataset, **When** `load_assin2_raw_split(split="test")` is invoked, **Then** it MUST return a Pandas DataFrame or list of dicts containing non-empty `premise`, `hypothesis`, and mapped `label` strings.
2. **Given** an ASSIN 2 sample with `entailment_judgment = "ENTAILMENT"`, **When** processed by `normalize_label()`, **Then** the normalized label MUST equal `"Entailment"`.
3. **Given** an ASSIN 2 sample with `entailment_judgment = "NONE"`, **When** processed by `normalize_label()`, **Then** the normalized label MUST equal `"Contradiction"`.

---

### Story 2 - vLLM Prefix Caching Prompt Formatter (Type: FUNCTIONALITY, Priority: P2)

Construct static system prompt prefixes and user state strings designed to maximize vLLM's `--enable-prefix-caching` KV-cache hit rate across queries.

**Why this priority**: Unlocks the ultra-fast System 1 prefill performance (~5-10ms) by ensuring the system instruction and choices schema remain 100% byte-identical across requests.

**Independent Test**: Unit test verifying exact string formatting and byte-level alignment of the system prompt prefix.

**Acceptance Scenarios**:

1. **Given** a `premise` and `hypothesis` pair, **When** `format_system1_prompt(premise, hypothesis)` is called, **Then** it MUST return a `state` string containing a standardized system header, the premiss text, the hypothesis text, and a rigid formatting structure.
2. **Given** multiple queries processed sequentially, **When** formatted by `format_system1_prompt`, **Then** the system instruction prefix MUST remain identical to enable vLLM KV-cache reuse.

---

### Story 3 - Frozen Test Set Sampler with Deterministic Seed (Type: HYPOTHESIS, Priority: P3)

Provide a deterministic sampling function `get_frozen_eval_set(sample_limit=200, seed=42)` that extracts a balanced 50/50 subset of `Entailment` and `Contradiction` pairs for benchmark evaluation.

**Why this priority**: Ensures reproducible evaluation across System 1, System 2, and Adaptive Cascade runs without class imbalance bias.

**Independent Test**: Executing `get_frozen_eval_set(200, seed=42)` twice yields identical sample ordering and exactly 100 `Entailment` and 100 `Contradiction` pairs.

**Acceptance Scenarios**:

1. **Given** `sample_limit = 200` and `seed = 42`, **When** `get_frozen_eval_set()` is called, **Then** the returned list MUST contain exactly 200 items, with 100 labeled `"Entailment"` and 100 labeled `"Contradiction"`.
2. **Given** repeated invocations with `seed = 42`, **When** dataset hashes are computed, **Then** the hash of the dataset samples MUST be 100% deterministic.

---

### Edge Cases

- **Missing or corrupted dataset split on Hugging Face**: The loader MUST fallback to cached local copies or raise an informative `DatasetDownloadError`.
- **Malformed text or missing fields in sample**: Rows with `None` or empty strings in `premise` or `hypothesis` MUST be filtered out automatically with a warning log.
- **Unrecognized label string**: Any label value other than `"ENTAILMENT"` or `"NONE"` (e.g., numeric values or unexpected strings) MUST raise a `ValueError("Invalid ASSIN2 label")`.
- **Sample count requested exceeds available split size**: If `sample_limit` > split length, system MUST raise `ValueError` and log max available rows.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST load the `nilc-nlp/assin2` dataset split (`test` or `train`) via Hugging Face `datasets` API.
- **FR-002**: System MUST convert `entailment_judgment` values into target binary strings: `"ENTAILMENT"` -> `"Entailment"`, `"NONE"` -> `"Contradiction"`.
- **FR-003**: System MUST provide a function `format_system1_prompt(premise: str, hypothesis: str) -> Dict[str, str]` returning:
  - `system_prefix`: Static system instruction for vLLM prefix caching.
  - `user_state`: Prompt payload containing the premise and hypothesis text.
- **FR-004**: System MUST provide a deterministic sampling function `get_frozen_eval_set(split: str = "test", sample_limit: int = 200, seed: int = 42) -> List[Dict[str, Any]]`.
- **FR-005**: System MUST validate input strings to prevent prompt injection or formatting breaking characters.

### Non-Functional Requirements *(AI-critical)*

- **NFR-001**: Execution time for dataset loading and prompt formatting of 200 samples MUST stay under 1.0 second.
- **NFR-002**: Memory footprint for storing loaded test samples MUST stay under 50 MB RAM.
- **NFR-003**: Determinism: 100% bit-exact reproducibility when sampling with fixed seed `42`.

### Key Entities

- **ASSIN2Sample**:
  - `id`: Unique string or int identifier.
  - `premise`: Raw Portuguese premise text.
  - `hypothesis`: Raw Portuguese hypothesis text.
  - `label`: Mapped binary label (`"Entailment"` or `"Contradiction"`).
  - `original_score`: Float similarity score (1.0 to 5.0) from ASSIN 2 metadata.
- **FormattedPromptPayload**:
  - `system_prefix`: Static prompt prefix for KV-cache reuse.
  - `user_state`: Instance specific text payload.
  - `choices`: Rigid list `["Entailment", "Contradiction"]`.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% accuracy in mapping ASSIN 2 raw labels to `"Entailment"` and `"Contradiction"`.
- **SC-002**: 100% bit-exact determinism across repeated executions of `get_frozen_eval_set(200, seed=42)`.
- **SC-003**: Zero prompt formatting errors or unhandled schema violations on the entire ASSIN 2 test split (2,448 samples).

---

## Assumptions

- The `nilc-nlp/assin2` dataset is publicly accessible on Hugging Face Hub without requiring authentication tokens.
- The task scope is restricted strictly to binary Natural Language Inference (NLI) in PT-BR.
- Python 3.12 environment with `datasets>=2.14.0` and `pandas>=2.0.0` pre-installed.