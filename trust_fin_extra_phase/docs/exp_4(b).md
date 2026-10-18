# Experiment 4(b) — Proposed method on the protected tests (final)

**Status:** complete. The protected sets were opened **once**, by `P4-E8`, only after
`phase4_freeze.json` said `FROZEN` and `TRUSTFIN4_PROTECTED=1` was set. No hyper-parameter,
threshold or calibrator was changed after that point.
**Run artifacts:** `artifacts/phase4/` (`Phase4_full_TEST.ipynb`, `metrics/P4-E8*.json`,
`analysis/P4-E8*.csv`, `manifests/phase4_completion_report.json`).

| | value |
|---|---|
| freeze | `frozen_at_utc 2026-10-12T12:27:00Z`, `run_mode full` |
| freeze sha256 (before and after test access, unchanged) | `701afd75dbc964dee852d36bdca9be4faad4b14730b532a9a82936bf4a03e079` |
| protected sets opened | `finqa_test` (1147), `fiqa_test` (234), `finben_fomc_test` (496) |
| `fintrust_registry` | not opened (P4-E9 deferred by `PLAN.md` §4) |
| completed | `2026-10-12T14:40:12Z` (≈ 2 h 15 m wall, RTX 4050 6 GB, peak ≈ 3.1 GB) |

---

## 1. FinQA test — C1 (execution-verified generation)

Frozen Phase-3 baseline (`P4-E8a`), 1147 records: printed-answer accuracy 0.3496,
**execution accuracy 0.6574**, program accuracy 0.5894.

EVG (`P4-E8b`), n = 8:

| rule | answer acc | execution acc | program acc | invalid prog. rate | mean agreement |
|---|---|---|---|---|---|
| `c0_print` | 0.3365 | 0.6399 | 0.5763 | 0.0035 | 1.000 |
| `c0_exec` | 0.6399 | 0.6399 | 0.5763 | 0.0035 | 1.000 |
| `best_of_n` | 0.3531 | 0.6556 | 0.5885 | 0.0026 | 0.496 |
| `self_consistency` | 0.3514 | 0.6478 | 0.5798 | 0.0026 | 0.496 |
| **`exec_verified`** ⭐ | **0.6556** | **0.6556** | 0.5876 | **0.0000** | 0.860 |
| *frozen greedy (`greedy_exec`)* | 0.3496 | **0.6574** | 0.5894 | — | — |

Per-record, `exec_verified` vs frozen greedy:

| | n |
|---|---|
| both correct | 734 |
| EVG-only correct | **18** |
| greedy-only correct | **20** |
| both wrong | 375 |

**C1 is rejected on FinQA.** `exec_verified` is 0.6556 vs the frozen baseline's 0.6574 —
**−0.18 pp**, with 18 records gained and 20 lost. This reproduces the validation result
(−0.00 pp there) and confirms the effect is a tie, not a win. The single real gain is
program hygiene: the invalid-program rate drops from 0.0035 to **0.0000**; every answer
`exec_verified` produces is backed by a program the official evaluator accepted.

Agreement, however, is strongly predictive on test:

| EVG agreement bin | n records | `exec_verified` accuracy |
|---|---|---|
| (0.00, 0.25] | 35 | **0.029** |
| (0.25, 0.50] | 122 | 0.238 |
| (0.50, 0.75] | 181 | 0.448 |
| (0.75, 1.00] | 809 | **0.792** |

A 2.9 % accuracy regime and a 79.2 % regime are separated by nothing the model says about
its answer — only by how much its own samples agree. This is the signal §3 turns into a
working abstention policy.

## 2. FiQA test — C1

| aggregator | MAE | R² | Pearson r | acc (tol 0.10) |
|---|---|---|---|---|
| greedy (frozen) | **0.13012** | 0.7723 | 0.8837 | 0.5684 |
| `median_of_n` (EVG) | 0.13058 | 0.7715 | 0.8832 | 0.5556 |

**C1 is rejected on FiQA too.** The small validation gain (MAE 0.1301 → 0.1258) does not
reproduce: on test, median-of-8 is 0.0005 *worse* and accuracy is 1.3 pp lower. The FiQA
freeze had already picked the baseline frame with raw confidence, so nothing downstream
depends on this.

## 3. FinQA test — C2 & C3 (frozen abstention, `P4-E8e`)

The frozen policy was applied **as written** — the validation-fitted calibrator
(`platt_conf_full` on the `finqa_exec_verified` frame, features = selected confidence,
agreement, executable rate) and the validation thresholds. Nothing was refitted.

| coverage target | threshold | test coverage | test selective risk | test selective accuracy | abstained | error reduction |
|---|---|---|---|---|---|---|
| — (answer all) | — | 1.000 | 0.3444 | 0.6556 | 0 | — |
| **0.80** | 0.4090 | 0.7855 | **0.2375** | **0.7625** | 246 | **−31.0 %** |
| 0.90 | 0.2211 | 0.9259 | 0.3004 | 0.6996 | 85 | −12.8 % |

**C2/C3 hold on the protected test.** Refusing 21 % of the test answers raises the accuracy
of the kept answers from 65.6 % to 76.2 % — a **31.0 % relative error reduction**, against
the 34.2 % predicted on validation. The transfer is close (test coverage 0.786 vs 0.800
target; test risk 0.238 vs validation 0.214), which is what a calibrated, pre-registered
threshold is supposed to do. On validation the same frame's raw confidence gave
risk@cov0.80 = 0.3088; the EVG-feature calibrator gives 0.2139 (validation) → 0.2375 (test).

## 4. FiQA test — abstention

| coverage target | test coverage | test selective risk | error reduction |
|---|---|---|---|
| 0.80 | 0.7650 | 0.4190 | −2.9 % |
| 0.90 | 0.8846 | 0.4203 | −2.6 % |

As the FiQA freeze predicted, abstention on FiQA is near-worthless (−2.9 %). Reported as a
negative result; raw confidence on this regression task does not rank errors.

## 5. FinBen-FOMC test — zero-shot OOD (adapter vs base)

Never tuned, no FOMC in training.

| system | accuracy | macro-F1 | parse success | F1 dovish | F1 neutral | F1 hawkish |
|---|---|---|---|---|---|---|
| base `Qwen/Qwen3-1.7B` | 0.4536 | 0.2147 | 0.899 | 0.163 | 0.650 | 0.046 |
| **E5 adapter (frozen)** | **0.4839** | **0.3035** | **0.986** | 0.333 | 0.621 | 0.260 |

The financial adapter improves OOD stance classification: **+3.0 pp** accuracy and
**+8.9 pp** macro-F1 (the balanced metric that matters, since the set is
neutral 245 / dovish 129 / hawkish 122). The gain is concentrated in the non-neutral
classes (dovish F1 0.16 → 0.33, hawkish 0.05 → 0.26) — exactly the classes a
neutral-majority model would otherwise ignore — and parse success rises 0.90 → 0.99.

## 6. Efficiency (measured), on the protected pass

| pass | records | wall | s/record |
|---|---|---|---|
| FinQA greedy 1147 | 1147 | ≈ 28 min (incl. model load) | 1.46 |
| **FinQA EVG n = 8** | 1147 × 8 = 9176 candidates | ≈ 90 min | **4.71** |
| FiQA EVG n = 8 | 234 × 8 | ≈ 7 min | 1.79 |

EVG costs ≈ **3.7×** greedy per FinQA record (4.71 vs 1.28 s on validation), not 8×,
because the 8 candidates share one prefill. Peak VRAM ≈ 3.1 GB on a 6 GB RTX 4050.

## 7. Verdict against the pre-declared claims

| claim | prediction | outcome |
|---|---|---|
| **C1** — execution-verified generation improves the answer | accuracy ↑ | **Rejected.** FinQA test −0.18 pp (tie), FiQA test −0.0005 MAE. Only program hygiene improves (invalid-prog. 0.0035 → 0.0000). |
| **C2** — calibration gives thresholds probability meaning | Brier/ECE ↓, risk@coverage usable | **Supported (FinQA).** Brier −44.5 %, ECE −81 % vs raw; the frozen threshold transfers (predicted −34 % error, realised −31 %). |
| **C3** — EVG features improve the calibrator | e-AURC ↓ where non-monotone features exist | **Supported (FinQA), rejected (FiQA).** FinQA e-AURC 0.2029 → 0.1232 (−39.3 %) with agreement+executability; FiQA all calibrators hurt ranking. |

**The honest headline.** On a task where the frozen model already writes an executable
program 99.7 % of the time, execution-verified sampling cannot buy accuracy — it *ties* the
greedy baseline. What it does buy is a trust signal: the agreement among its own samples is
monotone and steep (2.9 % → 79.2 % accuracy across agreement bins), it cannot be read off
the model's log-probability (ROC-AUC 0.577), and calibrating on it produces a pre-registered
abstention policy that cuts FinQA test error by **31 %** at 80 % coverage while reproducing
its validation behaviour. Generation-level verification did not make the model *right*; it
made the model *know when it is wrong*.

## 8. Limitations (for the write-up, stated up front)

1. **C1 is a negative result.** It must be reported as such, not buried. The method's value
   is entirely in abstention, not in a better answer.
2. **FiQA transfers neither C1 nor the calibration benefit** — 117 validation / 234 test
   records and a regression target are too little for the EVG features to pay off.
3. **Single backbone, single seed.** All sampling uses one frozen adapter; between-seed
   variance of the agreement signal is not measured. n = 8 is pre-declared, not tuned.
4. **Cost.** The abstention gain costs 3.7× the greedy latency; the accuracy gain is zero.
   Deploying EVG is justified only where a 31 % reduction in answered-error is worth 3.7×.
5. **FOMC is zero-shot**, never trained or tuned, so its absolute numbers are a floor, not a
   tuned ceiling.

## 9. Annex A — Statistical rigor (added with the eval-rigor pass)

The C1 verdict, the transfer numbers, and the difficulty story above are re-stated with
proper tests and confidence intervals in `docs/eval_rigor.md` (computed offline from the
frozen protected-test artifacts by `scripts/eval_rigor.py`, B=10,000 bootstrap resamples,
seed 42). Highlights:

- **C1 is a verified tie.** Paired McNemar (exact) p = 0.871; bootstrap 95% CI on the
greedy-vs-exec-verified delta is [-0.0122, +0.0087], P(delta > 0) = 0.343. The 0.18-p.p.
gap is numerically and statistically indistinguishable from zero.
- **Agreement bins carry Wilson CIs** (0.029 [0.005, 0.145] … 0.792 [0.763, 0.819]); bins 2-4
do not overlap.
- **Transfer is statistically significant.** Selective-risk error reduction at theta=0.4090:
31.0% with 95% CI [22.8%, 39.3%] — the interval excludes 0, and realised -31.0% vs
validation-predicted -34.2% is inside it. FiQA: CI [-2.9% region] ≈ 0, consistent with nil.
- **FiQA C1:** greedy-vs-median MAE delta CI [-0.0020, +0.0028] contains 0.
- **Error taxonomy:** the only channel EVG leaves open is the *printed answer string*
(355/1147 = 31.0% of greedy records compute the right value but print the wrong string;
0 for exec-verified). Roadmap R1 in eval_rigor.md targets exactly this channel.
- **Difficulty:** abstention rate and relative risk cut rise with arithmetic depth
(0-op: 4.4%/25%; 3-6-op: 31.3%/41.7%).
- **Comparability:** our validation set *is* the official FinQA dev partition (883/883 ids
match); our protected test is a carve of the official 1,800-item test with the other 653
records never opened. Head-to-head numbers against FinQANet/Phogat et-al. rows must remain
split-labeled; see the positioning table in `docs/eval_rigor.md`.
