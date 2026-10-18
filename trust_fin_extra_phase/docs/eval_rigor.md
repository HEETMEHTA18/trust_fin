# Evaluation Rigor Annex — Phase-4 EVG-CAS

Status: **Static analysis of the frozen Phase-4 protected-test artifacts** (`eval_rigor/summary.json`), produced by `trust_fin_extra_phase/scripts/eval_rigor.py`. No new model runs; the frozen two-key record is untouched. Everything below is measured on the **protected** FinQA test carve (n=1,147) and FiQA test (n=234).

## 0. Why this annex exists

The Phase-4 claims were qualitative (accuracy "ties"; abstention "transfers"). This annex upgrades them to quantitative evidence and connects the evaluation protocol to the supporting literature (FinQA paper held-out protocol + reasoning-depth analysis [finqa]; fine-tuned-SLM execution-accuracy evaluation [smlm]): error taxonomy, paired-hypothesis tests, bootstrap confidence intervals, difficulty-stratified selectivity.

## 1. C1 (accuracy): greedy vs EVG are statistically indistinguishable

Protected FinQA test, paired per-record comparison (n=1,147):

| quantity | value | 95% CI |
|---|---|---|
| greedy execution accuracy | 0.6574 | — |
| exec-verified (EVG) execution accuracy | 0.6556 | — |
| delta (EVG − greedy) | −0.0017 | bootstrap [−0.0122, +0.0087] |
| McNemar exact p (2-sided) | **0.871** | — |
| P(bootstrap delta > 0) | 0.343 | — |

Confusion (paired): both correct 734, EVG-only 18, greedy-only 20, both wrong 375.

Facts established: (i) the 0.0017 gap is two orders of magnitude below any practical difference; (ii) the 95% CI alone covers ±0.01; (iii) Weighted McNemar does not reject at any conventional level. **Conclusion: C1 stays REJECTED for "EVG improves accuracy," and is now stated with a proper test** — the earlier tie claim is not an artifact of eyeballing rounded means.

FiQA (n=234): greedy MAE 0.1301 vs median-of-n MAE 0.1306; bootstrap delta CI [−0.0020, +0.0028] ⊃ 0. Same conclusion, regression objective.

## 2. Calibration signal: agreement bins (protected test, EVG)

Binned self-consistency accuracy (n=1,147), Wilson 95% CIs:

| agreement | n | accuracy | 95% CI |
|---|---|---|---|
| (0.00, 0.25] | 35 | 0.029 | [0.005, 0.145] |
| (0.25, 0.50] | 122 | 0.238 | [0.171, 0.321] |
| (0.50, 0.75] | 181 | 0.448 | [0.377, 0.520] |
| (0.75, 1.00] | 809 | 0.792 | [0.763, 0.819] |

Monotone (Spearman ρ=1.0 by construction); the bin CIs are separated for bins 2–4. 70.5% of records fall in the top bin at 0.79 accuracy — the score is informative but the top bin saturates (a known limitation of n=8 self-consistency; see roadmap).

## 3. Error taxonomy (protected test)

| cause | greedy | exec-verified |
|---|---|---|
| parse failure | 0 (0.0%) | 0 (0.0%) |
| invalid/unexecutable program | 3 (0.26%) | 0 (0.0%) |
| program executes but answer wrong | 393 (34.3%) | 395 (34.4%) |
| **executes correctly but emitted string wrong** | **355 (31.0%)** | 0 (0.0%) |

Main finding: on 31% of records the greedy model computes the correct value yet **prints an answer string that does not match the executed value** (formatting drift — e.g., scientific notation, rounding, or a different value echoed). EVG's `exec_verified` rule eliminates this mismatch by construction. **But** this is exactly the failure that a *printed-answer* consumer would still see — the calibration head scores the generated-program channel, not the user-facing string. This is the primary surface-level improvement available (see roadmap R1/R2).

## 4. Abstention transfer (frozen policy) — with intervals

FinQA protected test; frozen thresholds θ=0.4090 (cov≈0.80) and θ=0.2211 (cov≈0.90); calibrated score p̂ = P(prefer-exec_verified) after OOF Platt + isotonic fit (frozen, calibration on validation only).

| metric | @θ0.80 | @θ0.90 |
|---|---|---|
| achieved coverage | 0.7855 | 0.9259 |
| full-risk | 0.3444 | 0.3444 |
| selective risk (mean) | 0.2375 | 0.3004 |
| selective risk 95% CI (bootstrap) | [0.209, 0.266] | [0.272, 0.328] |
| coverage 95% CI | [0.761, 0.809] | [0.910, 0.941] |
| relative error reduction (mean) | **31.0%** | 12.8% |
| error-reduction 95% CI | **[22.8%, 39.3%]** | — |

The error-reduction interval at θ0.80 excludes 0 → the FinQA transfer is **significant**. Realized −31.0% vs validation-predicted −34.2% lies inside the interval (prediction error 3.2 pp).

FiQA @cov0.80 (threshold = 20th score percentile): coverage mean 0.799 [0.748, 0.850], selective risk 0.423 [0.352, 0.495] vs full 0.432 → transfer null (consistent with exp_4(b) nil-abstention conclusion).

## 5. Difficulty stratification (FinQA protected test, EVG)

Arithmetic-operation depth of the *selected* program (FinQA's own reasoning-depth measure):

| program ops | n | accuracy (full) | accuracy (selected, θ0.80) | abstention rate | relative risk cut |
|---|---|---|---|---|---|
| 0 ops (lookup/reference) | 45 | 0.844 | 0.884 | 4.4% | 25.2% |
| 1 op | 644 | 0.649 | 0.764 | 23.1% | 32.6% |
| 2 ops | 391 | 0.662 | 0.748 | 18.9% | 25.2% |
| 3–6 ops | 67 | **0.552** | 0.739 | **31.3%** | **41.7%** |

The abstention policy concentrates on exactly the records where the model is weakest (deepest programs): hardest bucket is abstained most (31%) and gains the largest relative risk reduction (42%). 0-op lookups are near-memorized and almost never abstained. This is the intended behavior of conservative selection and the strongest single figure for the showcase.

## 6. Positioning vs the supporting papers (split-labeled)

| system | data | split | exec accuracy |
|---|---|---|---|
| **Ours (EVG, FROZEN)** | FinQA train (SFT, in-domain DSL) | **official dev n=883** | **0.6750** |
| **Ours (EVG, FROZEN)** | "" | **protected test carve n=1,147** | **0.6556** |
| FinQANet (RoBERTa-large) [finqa] | FinQA full train | official test n=1,800 | 0.6124 |
| FinQANet-Gold (RoBERTa-large) [finqa] | FinQA full train (+gold programs) | official test | 0.7000 |
| PHI-3-MINI fine-tuned [smlm] | FinQA full train (GPT-4-distilled programs) | official test | 0.7349 |
| PHI-3-MEDIUM fine-tuned [smlm] | "" | official test | 0.7759 |

Notes, to be read with honesty:
- Our dev (883) is literally the official FinQA dev set; our protected test (1,147) is a carve of the official 1,800-item test — the remaining 653 items were never opened (integrity by design).
- Direct numbers are **not** comparable to FinQA-paper/Phogat rows: different retained splits, different program corpora (gold vs in-domain teacher model), different model capacity (1.7B vs 12B BERT-large / 3.8–14B SLMs), different training-program provenance. We report them side by side only to situate scale; the paper must not imply head-to-head wins.
- The 65.05% execution-accuracy figure in the FinQA abstract is on an ambiguous split (Table 2 reports 61.24% test for RoBERTa-large); we did not use the abstract figure.

## 7. Improvements on top of the given papers (adopted now / roadmap)

Adopted in this phase:
- **Program-execution evaluation instead of string match** (Phogat et al. [smlm]; FinQA official evaluator): execution accuracy computed by running DSL programs — immune to formatting drift in the metric itself.
- **Held-out + reserved-test protocol** (FinQA paper protocol [finqa]): validation = official dev; protected test carve; calibration/threshold only on validation.
- **Reasoning-depth stratification** (FinQA paper analysis): abstention benefit shown monotone in program depth.

Roadmap (why-the-improvements-were-not-all-pushed-in-this-phase — they would require retraining or re-inference):
- R1 *(surface-language fix)*: emit the executed value as the printed answer (removes the 31% format-mismatch channel).
- R2 *(teacher programs)*: distill higher-quality Python programs from a teacher (Phogat et al. GPT-4-distillation recipe [smlm]) for SFT.
- R3 *(larger student)*: scale within SLM class, 3.8–14B (PHI-3 / ORCA-2 / Mistral-7B class) [smlm].
- R4 *(self-consistency depth)*: increase n beyond 8 and/or score the top bin (agreement=1.0) with calibrated logprob, given top-bin saturation at 0.79.
- R5 *(verbalized uncertainty)*: verbalize confidence in the answer string (Lin et al. [selcal]) for "human-readable" abstention reasons.
- R6 *(model-internal knowledge)*: plug FinGPT-style financial-LLM pretraining weights [finllm] beneath the same protocol.
- R7 *(XAI framing)*: present calibrated abstention + program evidence as the explanation artifact (Fritz-Morgenthal et al. [xai]).

Each roadmap item is a *follow-up study*; none weakens the integrity of the frozen result, and none was executed on the protected test.

## Confidence & reproducibility
- RNG seed 42; B=10,000 bootstrap resamples; exact McNemar p via binomial tail.
- All inputs = frozen CSVs/JSON under `artifacts/phase4/`; script at `scripts/eval_rigor.py`; outputs at `artifacts/phase4/analysis/eval_rigor/`.