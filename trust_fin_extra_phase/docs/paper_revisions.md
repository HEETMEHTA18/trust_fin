# Paper Corrections — Phase-4 draft → after the evaluation-improvement pass

Applies to the draft currently at `../paper/TrustFin_Conference_Paper.{tex,md}`. A corrected
self-contained copy lives here at `paper/TrustFin_Conference_Paper.{tex,md}` so the handoff
folder carries the update. Evidence for every change: `docs/exp_4(c).md` and
`artifacts/phase4_2/analysis/`.

## C1 — agreement-bin prose (reviewer issue #2)

**Current (ambiguous/contradictory):**
> "the model reports perfect agreement on records that are 2.9% correct and record-high
> agreement on records that are 79.2% correct"

**Corrected:**
> "The model reports total disagreement (agreement ≤ 0.25) on records that are 2.9%
> correct and unanimous agreement (agreement = 1.0, the top bin) on records that are 79.2%
> correct; nothing in its own log-probability separates these regimes."

Rule: any sentence naming a bin must name the bin's agreement range. The lowest bin describes
**disagreement**, never "perfect agreement".

## C1 — error-taxonomy reconciliation (reviewer issue #1)

Replace the loose "355 format errors / 395 wrong" prose with the mutually exclusive 6-cell
partition (from `eval_rigor` + `phase4_2`), all sums = 1,147:

| cell (greedy) | n |
|---|---|
| parse failure | 0 |
| invalid / unexecutable program | 3 |
| exec correct & print correct | 399 |
| exec correct & print wrong (format mismatch) | 355 |
| exec wrong & print wrong | 388 |
| exec wrong & print correct (string matches gold, executed value differs) | 2 |

State explicitly: **execution accuracy ≠ printed-answer accuracy**. For greedy they differ
solely by the 355-record format channel (0.6574 vs 0.3496). For exec-verified: invalid 0,
executes-wrong 395, correct 752, answer/program mismatch 0 by construction. Both channels'
totals reconcile to 1,147. (The draft's Table 6 "greedy executes right, prints wrong 355 /
EVG executes right prints wrong 0" is correct but must be presented with these mid-level
categories, not as "355 vs 395".)

## Results — add the matched-coverage comparison (reviewer priorities 1–3)

New Table (covered in `docs/exp_4(c).md` §1): selective risk at matched coverage, all
evaluated on the execution-correctness channel with the same answer-selection rule:

| strategy | cov 0.5 | cov 0.8 | cov 0.9 |
|---|---|---|---|
| greedy + confidence | 0.674 | 0.685 | 0.677 |
| best-of-8 + confidence | 0.660 | 0.654 | 0.659 |
| exec-verified + agreement | **0.818** | **0.758** | 0.715 |
| exec-verified + frozen p̂ | 0.818 | 0.754 | 0.710 |

Where, e.g. at coverage 0.80: agreement cuts error **−29.7%** vs greedy-full and **−23%** vs
confidence-abstention at the same coverage (paired bootstrap CI [−0.093, −0.055],
P=1.0). Confidence-only abstention is ≤8% — it is the weak baseline the paper must report
to be fair.

## Add the ablation table (reviewer priority 3)

| config | agreement | executability | confidence | cov-0.80 risk |
|---|---|---|---|---|
| A | − | − | ✓ | 0.315 |
| B | ✓ | − | − | **0.242** |
| C | − | ✓ | ✓ | ≈ A |
| D | ✓ | ✓ | − | 0.242 |
| E (full) | ✓ | ✓ | ✓ (p̂) | 0.246 |

Claim: **agreement is necessary and sufficient**; executability adds nothing (AUC 0.51);
the Platt-full calibrator adds no ranking (P=0.097 vs agreement-only) but supplies the
threshold's probability meaning. Scoring AUC table (§1 of exp_4(c)): agreement 0.758 vs
confidence 0.578.

## FiQA — turn the negative into a qualified positive with the caveat (reviewer priority 5)

Keep the frozen-transfer result as a negative (published, honest). Add: at **matched
coverage**, the agreement/median strategy has the best selective MAE (0.80 cov: 0.1094 vs
greedy 0.1176; AUC 0.597 vs 0.521) — descriptive, n=234; motivates a larger study. Refs:
existing confidence-AUC 0.577 on FinQA.

## Claims tightening (reviewer priority 7)

- Reposition title/abstract: **"trustworthy selective prediction for a small financial
  language model"**, not "improved financial reasoning."
- Do **not** claim seven-dimension FinTrust benefit (§ future work / P4-E9); supported claim:
  *agreement-calibrated abstention improves selective reliability on FinQA under the frozen
  protocol, and agreement transfers as a ranking signal to FiQA (descriptive).*
- Add a sentence that C1 "ties" now rests on exact McNemar (p=0.871) + bootstrap CI
  [−1.22, +0.87] pp, and the matched-coverage gain does **not** imply accuracy gain.

## Latency honesty (reviewer priority 6)

Keep 4.8 s vs 1.3 s (3.7×). Add a "practical value" line: 31% answered-error reduction at
cov≈0.80 for 3.7× latency is the trade offered; FINANCIAL deployment call is a cost/benefit
judgment of the user, not claimed as universally worth it.

## Novelty positioning (added this pass)

Related-work section in the corrected paper now leads with a 2024–2026 positioning paragraph
(cited: CISC, Second Guess, SConU, LEC, ASPIRE, FinMAN, GBFR, FinChain). The paper's claim is
kept narrow: "[the] first calibration-grounded, coverage-targeted abstention for a 1.7B
financial SLM whose decision feature is agreement among execution-verified candidate programs,
validated on a frozen pre-registered protocol with matched-coverage baselines." Full audit:
`docs/novelty_check.md`; provenance: `artifacts/phase4_2/manifest.json`; FinTrust follow-up:
`docs/p4_e9_plan.md`.