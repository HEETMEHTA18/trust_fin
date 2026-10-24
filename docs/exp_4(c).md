# Phase-4C: Ablations, Statistical Evaluation & Methodology Freeze

## Goal
Phase-4C evaluates whether the full TrustFin-SLM selective answering policy (with frozen Phase-4B risk estimator) actually improves *selective reliability* over simpler baselines. It does this on the same validation holdout from Phase-4B - **no new independent test set** is created. All thresholds come from fit-only Phase-4B, nothing is retrained or retuned here.

## Inputs
- Frozen Phase-4B outputs (`artifacts/phase4b_full/`): readiness, freeze, fit-only selection, holdout scores for B1-B4 and FiQA
- Phase-4A package (same ZIP used by Phase-4B) for features/labels and alignment
- Phase-3 deliverable ZIP for original E5 predictions and observed latencies

## Experiments (C1–C10)

| ID | Name | What it does |
|---|---|---|
| C1 | E5 written-answer baseline | Accuracy of raw model written answers on all FinQA and holdout. |
| C2 | Execution-only corrected baseline | Accuracy after executing the generated program (deterministic execution). Shows big gain (e.g., ~37.6% → ~67.5% on validation context). |
| C3 | Structural/grounding hard gates | Apply deterministic gates: `official_executable_feature` AND `program_structural_valid` (structural gate), plus numerical grounding (`any_literal_unmatched` false) as an additional hard filter. |
| C4 | Likelihood-only abstention | Apply frozen B1/B2 thresholds with validity gate (compare to execution-only). |
| C5 | Verifier-aware abstention | Apply frozen B3/B4 thresholds with validity gate. |
| C6 | Full TrustFin-SLM selective policy | Apply the **fit-only winner** from Phase-4B with validity gate. This is the proposed policy being evaluated. |
| C7 | Component ablations | Isolate contributions (likelihood vs verifier vs grounding) and look at error subgroups (e.g., literal-unmatched cases). |
| C8 | Paired significance & bootstrap CIs | McNemar's test on written vs executed (all FinQA). Paired bootstrap for accuracy/risk differences. Coverage-matched comparison (equal k) and FiQA selective MAE. |
| C9 | Efficiency reporting | Reports observed E5 generation latency from Phase-3; explicitly states verifier/calibrator overhead is **NOT_MEASURED** (CPU post-processing in this offline eval). |
| C10 | Methodology freeze | Write `phase4c_readiness.json` and `phase4_methodology_freeze.json`, package review artifacts. |

## Key Principles
- **No holdout leakage**: All thresholds imported from `B8_fit_only_selection.json`. No refitting/rethresholding on holdout.
- **Validation-only evidence**: Uses the same 185/25 holdout as Phase-4B - not a new independent test set. Protected tests untouched.
- **Honest reporting**: If likelihood-only wins on fit-OOF, novelty is marked as not supported. CIs/effects reported regardless of significance.
- **Execution is primary**: Structural executability + validity are enforced as hard gates where applicable.
- **Grounding is limited**: Numerical presence is checked, but semantic correctness is not fully verified.

## Outputs (what you get)
- `manifests/phase4c_readiness.json` — methodology freeze status with gates and cautions
- `manifests/phase4_methodology_freeze.json` — frozen policy (winner, threshold, gates), states tests not yet run
- `tables/C1_C2_answer_reconciliation.csv`, `tables/C3_C6_ablation_holdout.csv`, `tables/C7_component_ablation.csv`, `tables/C8_paired_significance.csv`, `tables/C8_coverage_matched_comparison.csv`, `tables/C8_fiqa_selective_mae.csv`, `tables/C9_efficiency.csv`
- `predictions/C3_C6_acceptance_masks.csv`
- `plots/C3_C6_selective_risk.png`
- `Phase4C_Review_Package.zip`

## What "PASS_METHODOLOGY_FROZEN" means
The methodology (features, splits, selection protocol, thresholds) is frozen and evaluated honestly on validation holdout. It does **not** claim the proposed method is better than baselines - you need to look at the actual selective risk/accuracy and CIs in the tables to judge.
