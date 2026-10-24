# Phase-4B: Verifier-Aware Confidence Calibration & Selective Prediction

## Goal
Phase-4B builds a **verifier-aware risk estimator** to decide *when to trust* the executed program from the frozen E5 model (Qwen3-1.7B joint QLoRA). The core question: can we predict correctness better than raw model likelihood so that we can selectively abstain on uncertain cases?

## Inputs
- Phase-4A review package: `phase4b_interface/phase4b_finqa_features.jsonl`, `phase4b_interface/phase4b_fiqa_features.jsonl`, `labels/phase4b_finqa_labels.jsonl`, `labels/phase4b_fiqa_labels.jsonl`, `phase4b_interface/phase4b_feature_policy.json`, `manifests/phase4a_readiness.json`
- Frozen splits: FinQA 883 total (698 fit / 185 holdout), FiQA 117 total (92 fit / 25 holdout). All splits fixed by Phase-4A.

## Experiments (B1–B8)

| ID | Name | What it does |
|---|---|---|
| B1 | Raw generated-token likelihood | Baseline confidence from model likelihood. Weak discriminator (~0.585 ROC-AUC). |
| B2 | Likelihood-only Platt/logistic | Calibrate raw likelihood with logistic regression (fit on calibration-fit). |
| B3 | Verifier-aware logistic | Adds verifier signals: `prompt_tokens`, `prompt_truncation_observed`, `official_executable_feature`, `program_structural_valid`. |
| B4 | Grounding-aware logistic | Adds numerical grounding features: `any_literal_unmatched`, `unmatched_literal_count`, `literal_presence_fraction`, `max_question_local_overlap` (recomputed by Phase-4A). |
| B5 | Selective prediction | Fit thresholds on fit-split OOF predictions to achieve target coverage (e.g., 80%). Apply same thresholds to holdout; coverage may vary. |
| B6 | FiQA regression/error uncertainty | Predict absolute sentiment error with Ridge using FiQA features; produce selective MAE and a descriptive (non-conformal) interval. |
| B7 | Reliability/AUC/Brier/ECE/AURC | Fit-OOF and holdout diagnostics to compare models. |
| B8 | Phase-4B freeze | Select winner from **fit-split OOF only** (min AURC, tie-break Brier), freeze feature policy, write handoff to Phase-4C. Holdout never used for selection. |

## Key Safeguards
- **No label leakage**: Gold answers, correctness labels (`execution_correct`, `written_correct`, `corrected_correct`), and errors are never used as predictors. Only whitelisted features.
- **Fit-only selection**: Hyperparameters/policy chosen from 5-fold (FinQA) / 4-fold (FiQA) OOF on calibration-fit. Thresholds frozen before looking at holdout.
- **Handoffs separate**: Model artifacts (`*.joblib`), predictions, tables, and manifests written to `artifacts/phase4b_full/`.
- **Deterministic gates**: Enforces record alignment, split sizes, feature whitelist, and presence of all B1–B8 outputs.

## Outputs (what you get)
- `manifests/phase4b_freeze.json` — winner, thresholds, feature sets, source hashes
- `manifests/phase4b_readiness.json` — PASS_FROZEN gates
- `manifests/B8_fit_only_selection.json` — fit-OOF selection details
- `tables/B1_B4_fit_oof_comparison.csv`, `tables/B7_model_diagnostics.csv`, `tables/B5_risk_coverage.csv`, `tables/B6_fiqa_uncertainty.csv`
- `predictions/B1_B4_finqa_holdout_scores.csv`, `predictions/B6_fiqa_holdout_scores.csv`
- `models/B2_likelihood.joblib`, `models/B3_verifier.joblib`, `models/B4_grounding.joblib`, `models/B6_fiqa_error_ridge.joblib`
- `Phase4B_Review_Package.zip`
