# Experiment 4(a) — Proposed method on validation (EVG-CAS), pre-protected

**Status:** validation complete ⇒ numbers below are the ones the method was frozen on.
**Protected tests are NOT opened in this document.** See `exp_4(b).md`.
**Run dir:** `trust_fin_extra_phase/artifacts/phase4/` (full). Smoke harness: `artifacts/phase4_smoke/`.
**Backbone / adapter / prompts:** inherited frozen from `../artifacts/phase3/manifests/phase3_freeze.json` (E5). **Unchanged.**

---

## 1. The method, in one paragraph

EVG-CAS = **Execution-Verified Generation + Calibrated Abstention**. At inference we sample
`n = 8` candidate programs at `temperature 0.7 / top_p 0.9`, execute each with the
**official FinQA evaluator on the input table already in the prompt**, and answer by majority
vote over the *executable* candidates (ties broken by higher mean token log-probability).
No gold program and no gold answer is consulted. Abstention is then driven by a calibrator
fitted **only on validation** (5-fold cross-fitting for reporting, refit on all of validation
for deployment) with thresholds fixed at the pre-declared coverages {0.80, 0.90}.

This is the *only* class of method available: Phase-3 froze the backbone, adapter, prompt
policy and selection rule, so Phase-4 may only change what happens at inference time.

## 2. Why this method — the Phase-3 evidence it answers

| Phase-3 finding | Phase-4 response |
|---|---|
| 96 % of FinQA errors are reasoning/execution errors; invalid-program rate 0.3 % | re-verify by **executing** sampled candidates → C1 |
| Confidence is only weakly informative (FinQA ROC-AUC 0.577, Spearman 0.125) | cross-fitted **calibration** so a threshold has probability meaning → C2 |
| No abstention mechanism and no gated trust eval anywhere in Phase-3 | risk-coverage curves + frozen operating points → C2/C3 |

## 3. Pre-declared protocol (nothing below was chosen after seeing a test set)

| Item | Value |
|---|---|
| Candidates `n` | 8 (ablation at 1, 2, 4, 8) |
| Sampling | temperature 0.7, top_p 0.9, `do_sample=True` |
| Token budget | unchanged from Phase-3: FinQA 192, FiQA 24, FOMC 16 |
| Selection rules compared | `c0_print`, `c0_exec`, `best_of_n`, `self_consistency`, `exec_verified` (⭐) |
| Calibration | 5-fold `KFold(42)` cross-fitted; methods `raw_confidence`, `platt_conf`, `isotonic_conf`, `platt_conf_full` |
| FiQA correctness | \|pred − gold\| ≤ 0.10 (the frozen Phase-3 tolerance) |
| Operating coverages | 0.80 and 0.90 (frozen); reporting grid {0.60, 0.70, 0.80, 0.90, 0.95} |
| Selection of calibrator | lowest **out-of-fold e-AURC** on validation, per frame |

**Selection-rule renames (honesty):** `c0_print` = candidate-0's own printed ANSWER line;
`c0_exec` = candidate-0's executor answer. The Phase-3 baseline rows are the *frozen*
`E5.csv` predictions, labelled `greedy_print` / `greedy_exec`. This matters: on the smoke
set `c0_exec` already beats `c0_print` by ~10 pp of answer accuracy, so a "baseline" that
uses the executor is a genuinely different, stronger control than the Phase-3 row.

## 4. What the metrics can prove (read before the tables)

- **C2 is tested on probability quality** (Brier / ECE / NLL) and on risk@coverage. A
  single-feature monotone calibrator (Platt, isotonic on confidence alone) refitted on the
  data it is applied to **cannot change the ranking**, so it cannot change AURC. Differences
  between the cross-fitted AURC of `platt_conf` and `raw_confidence` are an artifact of
  holding each fold's map fit on *different* data — they are reported as a deployable
  selection diagnostic, never as evidence of reordering.
- **C3 is tested on ranking**, because `platt_conf_full` adds EVG agreement and executable
  rate and is therefore *not* a monotone map of confidence. C3 holds iff it beats the
  single-feature methods on out-of-fold e-AURC.
- If `pick_best` returns `raw_confidence` for a frame, that is a **negative result for C3 on
  that frame**, and it is what the freeze records.

## 5. Smoke harness (what it proved, and what it could not)

`RUN_MODE=smoke` caps validation at 20 records/dataset and `n = 2`, and writes to
`artifacts/phase4_smoke/` so that a smoke freeze can never satisfy a full run's gate.
Three smoke passes were run end-to-end and fixed the following real bugs before any full
compute was spent:

1. `Path.cwd()` resolves to `.../notebooks` under nbconvert, so every output was landing in
   `notebooks/artifacts/phase4/` → workspace is now found by walking up to `PLAN.md`, with
   `TRUSTFIN4_WORKSPACE` as the authoritative override set by the shard tool.
2. Stage-id normalization upper-cased `P4-E3b` into the unknown `P4-E3B` → ids are now
   matched case-insensitively against the canonical set.
3. A figure call passed `marker="--"` (a linestyle) → matplotlib raised.
4. `NS_EVAL` advertised `n = 4` even when the smoke run only sampled 2 candidates → the
   ablation now only lists `n` values actually sampled.
5. Smoke and full shared artifact paths (a cached 20-record candidate file would have
   satisfied the full run's gates) → smoke now writes to `artifacts/phase4_smoke/`.

What smoke **cannot** say: any effect size. With 20 records a 10 pp accuracy difference is
noise, and the calibrators are fit on 20 points where a 5-fold split leaves 4 per fold —
Platt/isotonic are wildly unstable there and e-AURC differences are meaningless. The smoke
freeze in `artifacts/phase4_smoke/manifests/phase4_freeze.json` exists **only** to prove the
freeze/gate path works; it is never read by a full run.

## 6. Validation results

Validation = 883 FinQA + 117 FiQA records (`finqa_val` / `fiqa_val`, the same split Phase-3
selected E5 on). Protected test is untouched.

### 6.0 Integrity check first (P4-E1)

Phase-4's own evaluation path, pointed at the frozen E5 adapter, reproduces Phase-3 exactly:

| Metric | Phase-4 P4-E1 | Phase-3 (frozen) |
|---|---|---|
| FinQA execution accuracy | **0.6750** | 0.675 |
| FinQA program accuracy | 0.6116 | 0.612 |
| FinQA confidence ROC-AUC (execution) | **0.5769** | 0.5769 |
| FiQA MAE | **0.1301** | 0.130 |

So every downstream comparison is against a baseline that is *literally the same numbers*,
not a re-implementation. One number is worth flagging: **printed-answer accuracy is only
0.3760 while execution accuracy is 0.6750.** The model's own `ANSWER:` line is wrong for
62 % of the records whose program executes to the right value. Any Phase-4 rule that reads
the executor (rather than the model's own answer) starts from a much higher floor — which is
why `c0_exec` and `greedy_exec`, not `c0_print`, are the honest controls.

### 6.1 C1 — does execution-verified selection beat the frozen baseline?

**FinQA, n = 8 candidates, 883 records (`analysis/P4-E3_selection_ablation.csv`):**

| rule | answer acc | execution acc | program acc | invalid prog. | mean agreement |
|---|---|---|---|---|---|
| `c0_print` | 0.3726 | 0.6501 | 0.5753 | 0.0034 | 1.000 |
| `c0_exec` | 0.6501 | 0.6501 | 0.5753 | 0.0034 | 1.000 |
| `best_of_n` | 0.3907 | 0.6659 | 0.5946 | 0.0045 | 0.522 |
| `self_consistency` | 0.3828 | 0.6636 | 0.5946 | 0.0033 | 0.522 |
| **`exec_verified`** ⭐ | **0.6750** | **0.6750** | 0.6036 | **0.0023** | 0.853 |
| *frozen Phase-3 greedy (`greedy_exec`)* | 0.3760 | **0.6750** | 0.6116 | — | — |

**C1 is a negative result on FinQA.** `exec_verified` scores 0.6750 execution accuracy,
*exactly* the frozen greedy baseline (596/883 correct for both). Per-record:

| | n |
|---|---|
| both correct | 588 |
| EVG-only correct | **8** |
| greedy-only correct | **8** |
| both wrong | **279** |

EVG flips 8 records in and 8 out — net zero. Sampling at `temperature 0.7` is not a free
improvement: `c0_exec` (candidate 0, executed) lands at 0.6501, **2.5 pp below** the
temperature-0 greedy baseline, and EVG only claws that back to a tie. The one place EVG does
help on FinQA is program hygiene: invalid-program rate falls from 0.0034 to 0.0023.

**FiQA, n = 8, 117 records (`analysis/P4-E3b_fiqa_sampling.csv`):**

| aggregator | MAE | R² | acc (tol 0.10) |
|---|---|---|---|
| greedy (frozen) | 0.1301 | 0.7301 | 0.5470 |
| `mean_n` | **0.1258** | **0.7462** | 0.5556 |
| `median_n` | 0.1275 | 0.7373 | 0.5556 |
| `conf_weighted_mean` | 0.1261 | 0.7453 | 0.5556 |

**C1 is weakly positive on FiQA:** MAE −0.0043 (−3.3 % relative), R² +0.016, accuracy +0.9 pp.
Small, but consistent across all three aggregation rules.

> **Verdict on C1:** not supported as an accuracy claim on FinQA; weakly supported on FiQA.
> The honest reading is that on a task where the model already writes executable programs
> 99.8 % of the time, execution-verified sampling cannot add accuracy — it can only add a
> *confidence signal*. That is exactly what §6.3 shows it does.

### 6.2 C2 — does calibration make abstention usable?

`analysis/P4-E7_all_calibration.csv`, frame `finqa_exec_verified` (883 records):

| method | Brier ↓ | ECE ↓ | e-AURC ↓ | risk@cov 0.80 ↓ |
|---|---|---|---|---|
| `raw_confidence` | 0.2845 | 0.2555 | 0.2029 | 0.3088 |
| `isotonic_conf` | 0.2139 | 0.0164 | 0.2028 | 0.3173 |
| `platt_conf` | 0.2200 | **0.0096** | 0.2496 | 0.3215 |
| **`platt_conf_full`** ⭐ | **0.1579** | 0.0475 | **0.1232** | **0.2139** |

**C2 holds on probability quality, exactly as pre-declared.** Brier falls 0.2845 → 0.1579
(−44.5 %) and ECE 0.2555 → 0.0475. Note the `platt_conf` row: it drives ECE to 0.0096 but
*harms* ranking (e-AURC 0.2029 → 0.2496). That is not a contradiction — a single-feature
monotone map cannot re-order anything, and the cross-fitting folds add noise. It is the
reason §4 separates the two metric families.

**Abstention quality at the frozen operating points** (`manifests/phase4_freeze.json`):

| coverage target | threshold | achieved cov. | validation selective risk | full risk | error reduction |
|---|---|---|---|---|---|
| 0.80 | 0.4090 | 0.7995 | **0.2139** | 0.3250 | −34.2 % |
| 0.90 | 0.2211 | 0.9003 | **0.2679** | 0.3250 | −17.6 % |

Refusing the least-confident 20 % of FinQA answers cuts the error rate of the answers that
are kept by more than a third, on validation only.

### 6.3 C3 — do EVG features improve the calibrator?

`platt_conf_full` adds **EVG agreement** and **executable-program rate** to the confidence,
so it *can* reorder records and is the only method that can be tested on ranking.

| frame | best method | e-AURC | risk@cov 0.80 |
|---|---|---|---|
| `finqa_greedy_exec` (frozen baseline frame) | `raw_confidence` | 0.1897 | 0.3031 |
| `finqa_exec_verified` + `raw_confidence` | `raw_confidence` | 0.2029 | 0.3088 |
| `finqa_exec_verified` + **`platt_conf_full`** | **`platt_conf_full`** | **0.1232** | **0.2139** |

**C3 is supported on FinQA.** e-AURC falls 0.2029 → 0.1232 (**−39.3 %**) against the same
frame's raw confidence, and −35.1 % against the *frozen baseline frame at its own best
method* (0.1897 → 0.1232). Risk@coverage 0.80 improves 0.3088 → 0.2139. The gain comes from
agreement, which is what makes it a genuine reordering:

| EVG agreement bin | n records | `exec_verified` accuracy |
|---|---|---|
| (0.00, 0.25] | 34 | 0.147 |
| (0.25, 0.50] | 97 | 0.165 |
| (0.50, 0.75] | 125 | 0.512 |
| (0.75, 1.00] | 625 | **0.818** |

Agreement separates a 15 % from an 82 % accuracy regime. The model's own log-probability
cannot do that (ROC-AUC 0.577); the *diversity of its own samples* can.

**C3 is a negative result on FiQA.** The frozen FiQA method is `raw_confidence` on the
*baseline* frame `fiqa_greedy` (e-AURC 0.2372); every calibrator hurts FiQA ranking
(`platt_conf` 0.2841, `platt_conf_full` 0.2847, `isotonic` 0.3353) even though they all
improve Brier/ECE. With only 117 validation records and a regression target, the EVG
features do not pay for themselves — so the frozen FiQA abstention policy is deliberately
**the baseline plus raw confidence**, and that is what P4-E8e will apply on test.

### 6.4 Efficiency — what EVG costs

| pass | s/record | × greedy |
|---|---|---|
| FinQA greedy (frozen) | 1.284 | 1.00× |
| **FinQA EVG n = 8** | **4.81** | **3.75×** |
| FiQA greedy (frozen) | 0.391 | 1.00× |
| **FiQA EVG n = 8** | **1.11** | **2.84×** |

Sampling 8 candidates costs **3.75×**, not 8×, because the 8 candidates share one prefill.
The method is affordable at n = 8 on a 6 GB RTX 4050 (peak 3.1 GB during the pass).

### 6.5 What was frozen

`artifacts/phase4/manifests/phase4_freeze.json`, `frozen_at_utc 2026-10-12T12:27:00Z`,
`run_mode full`:

| | chosen | e-AURC | basis |
|---|---|---|---|
| FinQA | frame `finqa_exec_verified` + `platt_conf_full` | 0.1232 | lowest OOF e-AURC |
| FiQA | frame `fiqa_greedy` + `raw_confidence` | 0.2372 | lowest OOF e-AURC |
| operating points | coverages 0.80 / 0.90 | — | pre-declared |

`uses_gold_answer: false`, `uses_gold_program: false`. The registry and Phase-3 freeze
hashes are recorded inside. Nothing in §6 was computed from a protected record.

## 7. Freeze

`artifacts/phase4/manifests/phase4_freeze.json` is written only by P4-E7, from validation
data only. It records the chosen selection rule, calibrator and method, the operating-point
thresholds, and `uses_gold_answer: false`. `P4-E8` refuses to run unless that file says
`FROZEN` **and** `TRUSTFIN4_PROTECTED=1` is set in the environment — two independent keys.
