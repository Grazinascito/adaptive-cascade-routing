# adaptive-cascade-routing

Empirical benchmark for one question: **does the cost saving of a hybrid inference cascade survive when both paths run the same model?**

iFood's catalog classifier (Sugar) routes 50–70% of inferences to a cheap proprietary model and escalates the rest to frontier LLMs, cutting cost by an estimated 60–80%. That architecture couples the saving to *two different models*. This project removes that variable: the cheap path and the expensive path share the same weights (`Qwen/Qwen3-4B-AWQ` on a single vLLM server), and differ only in inference strategy — so whatever the benchmark measures is the strategy, not the model.

| Path | Mechanism | Endpoint | Cost profile |
|---|---|---|---|
| **System 1** (fast) | one-token typed decision: per-option probabilities from the server, client-side softmax, top confidence `P_top` and Shannon entropy | `/v1/systemone` | 1 forward pass, no decode loop |
| **System 2** (slow) | chain-of-thought, strict-JSON answer, `revisor_insight` logged for free (the iFood Revisor's raw material) | `/v1/chat/completions` | autoregressive decode |
| **Cascade** | RouteLLM-style rule `R_α`: accept System 1 when `P_top ≥ α`, escalate otherwise | — | System 1 cost + escalation share |

## The three scenarios

All three run on the same frozen test set, against the same server, with warmup requests discarded. Gates are computed once, from the committed per-item log (`results/raw_benchmark_data.json`).

| Scenario | Claim under test | Gate |
|---|---|---|
| **SC-A** — System 1 alone | cheap and fast, but loses accuracy on the noisy tail | tail accuracy ≥ 10 pp below System 2 |
| **SC-B** — System 2 alone | highest accuracy, highest token cost and latency | reference for both gates below |
| **SC-C** — cascade | keeps System 2's accuracy at a fraction of its token cost | recovers ≥ 50% of the S1→S2 accuracy gap (APGR ≥ 0.5) using ≤ 40% of System 2's decode tokens |

Methodology pinned in the code, not in prose: the α threshold is tuned **only** on a separate validation split (sweep 0.70–0.90); the frozen test set is measured once; a 100-item pilot runs first and halts everything if System 2 does not beat System 1 on the tail — the exact failure mode that made an earlier ASSIN 2 version of this benchmark unmeasurable (APGR denominator negative: the "weak" path beat the "strong" one on clean data).

## Dataset — DARU (real retail, no synthetic rows)

`gilsileydaru/classificao-produtos-varejo-cpg-prbr` — 158,113 product descriptions from the 18 largest Brazilian retailers (ABRAS ranking), paper: Darú et al. 2022 (DOI 10.5433/1679-0375.2022v43n2p189). Names are invoice-style and genuinely noisy — `APRESUNTADO REZENDE PEC KG`, `CARNE SUIN ESPINHACO KG`, `lte cond moca desn tp 395G` — the same regime described in iFood's engineering blog (`HMBR ART C/ CHED`).

- **Label**: `categoria` level of the dataset's taxonomy tree (segmento → categoria → subcategoria → canonical product).
- **12 food/beverage categories**, each with ≥ 1,400 examples, pinned in `config/benchmark_config.yaml` and guarded by tests.
- **Validation 200 / frozen test 600**, stratified 70/30 between *head* (descriptions longer than 20 chars) and *tail* (≤ 20 chars — the abbreviated long tail), seed 42, no overlap, no duplicates. Every item is a sampled real row; nothing is generated.

## Repository layout

```
config/benchmark_config.yaml   frozen contract (categories, splits, α, gates)
src/data/loader.py             DARU parser + stratified frozen splits
src/engine/system1.py          /v1/systemone typed-decision client (contract-restored)
src/engine/router.py           softmax renormalization, Shannon entropy, R_α decision (pure)
src/engine/system2.py          CoT + strict JSON client, fuzzy category matching
src/engine/server.py           vLLM server lifecycle (launch argv, health, stop)
src/benchmark/runner.py        scenarios, pilot gate, α sweep, checkpoints, results file
notebook/adaptive_cascade_benchmark.ipynb   end-to-end Colab run (T4)
tests/                         88 tests: contracts, math, parse edge cases, fake-server smoke
```

Robustness decisions worth knowing: a model answer whose category only *approximately* matches the list (e.g. `SUCCOS E REFRESCOS` for `SUCOS E REFRESCOS` — a real Qwen3-4B misspelling from the first run) is deterministically fuzzy-matched and logged (`s2_category_fuzzy`); an unparsable answer becomes a **counted failure** for that item (`s2_parse_ok`), never a crash; every phase dumps a checkpoint so late failures never discard finished GPU work.

## Reproducing

Local, no GPU needed (plumbing and contracts):

```bash
uv sync
uv run pytest          # 88 tests
uv run ruff check . && uv run mypy
```

Full measurement — one Colab T4 session: open [`notebook/adaptive_cascade_benchmark.ipynb`](notebook/adaptive_cascade_benchmark.ipynb), select a T4 runtime, Run all. The notebook checks the GPU, installs vLLM, clones this branch, downloads DARU via `kagglehub`, starts the server, runs the pilot gate (stopping on failure), then the full run, and saves charts plus a results archive.

## Results

**Measured or it didn't happen:** the numbers below come from the committed run in `results/raw_benchmark_data.json` (2026-11-15, Colab T4, `Qwen/Qwen3-4B-AWQ`, git `85a835f`, full run 13 min, 0 crashes, 0 parse failures, 1 fuzzy category match).

| Metric | SC-A (System 1) | SC-B (System 2) | SC-C (cascade, α=0.90 from validation sweep) |
|---|---|---|---|
| Accuracy (frozen test, n=600) | **8.5%** | **48.2%** | **30.0%** |
| Accuracy head / tail | 8.6% / 8.3% | 44.5% / 56.7% | 29.3% / 31.7% |
| Decode tokens | 0 | 68,120 | 60,838 (89.3% of SC-B) |
| Latency p50 | 450 ms | 3,733 ms | 4,205 ms |
| Escalated to System 2 | — | 100% | 82.7% |

**Gates: 2 of 3 passed.**

| Gate | Required | Measured | Verdict |
|---|---|---|---|
| Tail-accuracy gap (SC-B − SC-A) | ≥ 10 pp | +48.3 pp | **PASS** |
| Accuracy-gap recovery (APGR) | ≥ 0.5 | 0.542 | **PASS** |
| Decode-token share (SC-C / SC-B) | ≤ 0.40 | 0.893 | **FAIL** |

**Diagnosis (from the per-item log):** the cost gate failed because System 1's confidence signal carries no information. Its accuracy is flat at chance level (~8%, 12 classes) across *every* `P_top` bin — including the 106 items where it claimed `P_top ≥ 0.9` (accuracy there: 8.5%). The one-token letter-alias decision is confidently wrong, so the router escalates 83% of traffic (paying System 2 anyway) and accepts mostly-wrong answers on the rest. The cascade *machinery* is validated — the fast-path *instantiation* (single-token alias choice on a 4B model over 12 PT-BR retail categories) provides no usable signal. Candidate fix for the next iteration: score full category names as criteria instead of letter aliases.

Secondary observation: System 2 is *better on the tail than on the head* (56.7% vs 44.5%) — abbreviated invoice-style names give chain-of-thought less distractor text (brands, sizes) to trip over.

A write-up in Brazilian Portuguese follows.

## References

1. iFood — *Sugar, Revisor, and Pepper: the agentic cycle that maintains iFood's catalog scale and consistency* — https://tech.ifood.com.br/data-ai/sugar-revisor-and-pepper-the-agentic-cycle-that-maintains-ifoods-catalog-scale-and-consistency/
2. iFood — *Tokens are money: a strategic guide to cost optimization in LLMs* — https://tech.ifood.com.br/en/data-ai/tokens-are-money-a-strategic-guide-to-cost-optimization-in-llms
3. Ong et al. — *RouteLLM: Learning to Route LLMs with Preference Data* — https://arxiv.org/abs/2406.18665
4. Chen et al. — *FrugalGPT: How to Use Large Language Models While Reducing Cost and Damage* — https://arxiv.org/abs/2305.05176
5. Darú et al. — DARU dataset paper — DOI 10.5433/1679-0375.2022v43n2p189 · data: https://www.kaggle.com/datasets/gilsileydaru/classificao-produtos-varejo-cpg-prbr
6. vLLM — inference server and the `/v1/systemone` Structured Decisions endpoint — https://docs.vllm.ai

Full design decisions and the claim scope (what this benchmark asserts and does not assert about the iFood architecture): [SPEC-DARU.md](SPEC-DARU.md).
