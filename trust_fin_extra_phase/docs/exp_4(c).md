# Phase-4.2 — Evaluation-Improvement Pass (review-driven)

**Status:** analysis complete (2026-10-12). Produced by `scripts/eval_matched_coverage.py`
from the **frozen** Phase-4 prediction artifacts only; output in
`artifacts/phase4_2/analysis/`; reproducible via `notebooks/Phase4_2_Evaluation_Improvement.ipynb`.

**Why this pass exists.** External review of the Phase-4 draft asked (1) to compare
abstention methods **at matched coverage**, (2) to run **component ablations**, (3) to give
**confidence intervals** for selective risk, (4) to **reconcile the error taxonomy**, and
(5) to decide whether agreement-based calibration generalizes beyond FinQA. This pass answers
1–5 with statistics on the already-collected protected-test predictions. It is a **post-hoc
analysis**: matched-coverage thresholds are descriptive (for method comparison) and are *not*
new pre-registered operating points; nothing was refit, re-run, or opened beyond the one
existing protected access.

## 1. Matched-coverage selective prediction (FinQA, protected test, n=1,147)

All strategies answer the **same number of questions k** (k = ⌊coverage·n⌋), ranked by their
own score, evaluated on the **execution-correctness channel** (official FinQA evaluator).

| strategy (score) | cov 0.50 | cov 0.70 | cov 0.80 | cov 0.90 | cov 0.95 |
|---|---|---|---|---|---|
| greedy + **confidence** (baseline) | 0.674 | 0.675 | 0.685 | 0.677 | 0.668 |
| best-of-8 + confidence | 0.660 | 0.638 | 0.654 | 0.659 | 0.663 |
| exec-verified + **agreement** | **0.818** | **0.794** | **0.758** | 0.715 | 0.687 |
| exec-verified + agreement·executability | 0.818 | 0.794 | 0.758 | 0.714 | 0.686 |
| exec-verified + **frozen p̂** (Platt-full) | **0.818** | 0.794 | **0.754** | 0.710 | 0.689 |

Selective risk at **coverage 0.80** (with Wilson 95% CI):

| strategy | selective risk | CI | vs greedy-full (0.3426) |
|---|---|---|---|
| greedy + confidence | 0.315 | [0.286, 0.346] | −8.0% |
| best-of-8 + confidence | 0.346 | [0.316, 0.377] | ≈ 0% |
| exec-verified + agreement | **0.242** | [0.216, 0.271] | **−29.7%** |
| exec-verified + frozen p̂ | 0.246 | [0.220, 0.275] | −28.1% |

**Verdict vs the reviewer's core question.** At matched coverage, agreement-based abstention
cuts error ~29% relative to the greedy-full answer rate and ~23% relative to
confidence-based abstention at the *same* coverage. Confidence-only abstention is nearly
worthless (≤8%), best-of-n selection even mildly anti-productive. The method's operational
value survives when compared against its simpler alternatives — this is the strongest
comparative claim in the package.

### Paired significance at matched coverage 0.80 (bootstrap, B=3,000, seed 42)

| difference (A → B) | mean | 95% CI | P(B better) |
|---|---|---|---|
| greedy-conf → **exec-verified p̂** | −0.070 risk | [−0.089, −0.050] | 1.000 |
| greedy-conf → **agreement-only** | −0.074 risk | [−0.093, −0.055] | 1.000 |
| agreement-only → exec-verified p̂ | +0.003 risk | [−0.003, +0.010] | 0.097 |

The FinQA selective gain is **statistically robust** against the confidence baseline. The
p̂-calibrator-on-top adds nothing beyond agreement (consistent with its role as a monotone
rescore of the agreement ranking) — the *feature* is the agreement, the calibration gives it
probability meaning and a pre-registered threshold.

### Ranking quality of each score (ROC-AUC vs correctness, descriptive on test)

| score | AUC |
|---|---|
| raw model confidence (greedy) | 0.578 |
| best-of-8 confidence | 0.535 |
| executable-rate (exec-verified) | 0.510 |
| **agreement (exec-verified)** | **0.758** |
| agreement·executability | 0.757 |
| **frozen p̂ (Platt-full)** | **0.763** |

Executability carries no ranking signal on its own; agreement carries almost all of it.

## 2. Component ablation (what each signal contributes)

| configuration | agreement | executability | confidence | cov-0.80 risk |
|---|---|---|---|---|
| A | − | − | yes | 0.315 |
| B | yes | − | − | **0.242** |
| C | − | yes | yes | ≈ A (executability uninformative) |
| D | yes | yes | − | 0.242 |
| E (full) | yes | yes | yes (p̂) | 0.246 |

Ablation verdict: **agreement is necessary and sufficient**; confidence adds ranking power
only *after* agreement; executability contributes nothing (it is ~saturated after
exec-verification). This isolates the mechanism precisely and should be the paper's ablation
table.

## 3. Error-taxonomy reconciliation (mutually exclusive; partitions sum to 1,147)

Truth table over parse × execute × print for **greedy**:

| cell | n | note |
|---|---|---|
| parse failure | 0 | |
| invalid (unexecutable) program | 3 | |
| exec **correct** & print correct | 399 | |
| exec correct & print wrong (**format mismatch**) | 355 | 31.0% of records — EVG's *only* unfixed channel |
| exec wrong & print wrong | 388 | |
| exec wrong & print correct | 2 | printed string equals gold while executed value differs |

Execution accuracy = (399+355)/1147 = **0.6574**; printed-answer accuracy = (399+2)/1147 =
**0.3496**. The two metrics differ *solely* by the 355-record format channel (plus 2
reverse cases). For **exec-verified**: invalid 0, executes-wrong 395, correct 752 → 0.6556;
answer/program mismatch 0 by construction.

**Prose correction for the paper:** the phrase "perfect agreement" in the results section
describes the **highest** agreement bin and must be kept distinct from the **lowest** bin
(0–0.25 agreement → 2.9% correct). Suggested wording is in `docs/paper_revisions.md`.

## 4. FiQA regression (n=234): matched-coverage selective MAE

| strategy (score) | cov 0.50 MAE | cov 0.80 MAE | Δ vs greedy full (0.1301) | AUC vs correct@0.10 |
|---|---|---|---|---|
| greedy + confidence | 0.1009 | 0.1176 | −9.6% | 0.521 |
| median-of-8 + confidence | 0.1063 | 0.1134 | −12.8% | 0.535 |
| median-of-8 + **agreement (neg. sample std)** | **0.0919** | **0.1094** | **−15.9%** | **0.597** |

Agreement again ranks best and improves selective MAE at matched coverage on the regression
task, though the absolute effect is small and the sample is tiny (234). This is descriptive —
the frozen FiQA policy (raw confidence) remains a published negative transfer — but it
answers the reviewer's "does agreement generalize beyond FinQA?" with a qualified yes that
warrants a larger study.

## 5. FOMC
Per-record logits are not stored in the frozen artifacts, so matched-coverage FOMC is
deferred to P4-E9 (as already planned). Existing zero-shot macro-F1 0.215 → 0.304 stands.

## 6. Honest claim-set after this pass

- **Strong, robust:** at matched coverage, agreement-based abstention beats confidence-based
  abstention and best-of-n on FinQA (paired bootstrap excludes zero); agreement is the
  necessary feature; the pre-registered frozen-policy transfer (−31.0% error at cov≈0.786,
  CI [−23%, −39%]) is consistent with this matched-coverage view.
- **Supportable:** agreement helps selective ranking on FiQA (descriptive, small n).
- **Still open:** FinTrust seven-dimension evaluation (P4-E9); between-seed variance; latency
  (4.8 s vs 1.3 s, 3.7×) versus benefit; no accuracy gain (C1 stays rejected).

## 7. Reproducibility
`scripts/eval_matched_coverage.py` (seed 42, B=3,000) reads `artifacts/phase4/`
(read-only) and writes `artifacts/phase4_2/analysis/`; the notebook re-runs it and renders
plots. Provenance manifest (input SHA-256 + params + headline results):
`artifacts/phase4_2/manifest.json`.

## 8. Novelty check (added this pass)
See `docs/novelty_check.md` — an audited comparison vs CISC (2025), Second Guess (arXiv
2605.25394), SConU (2025), LEC (arXiv 2512.01556), ASPIRE (arXiv 2310.11689), FinMAN (2025),
GBFR (2026), FinChain (arXiv 2506.02515). No located work combines executed-program agreement
as a calibrated abstention feature with a frozen threshold + protected transfer on a 1.7B
financial SLM. The corrected paper's Related-work section was updated accordingly
(`paper/TrustFin_Conference_Paper.{tex,md}`).