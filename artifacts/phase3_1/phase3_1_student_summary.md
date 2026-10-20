# TrustFin-SLM: Phase-3.1 Student Audit

- Mode: **full**
- Core offline audit: **PASS**
- Full Phase-4 research readiness: **CONDITIONAL_HOLD**
- Primary frozen SLM: **E5 Qwen3-1.7B, balanced FinQA+FiQA QLoRA**.
- Frozen prompts: FinQA canonical; FiQA explicit-schema.

## Key FinQA validation observations
- Written answers correct: **332/883** (37.60%).
- Executed programs correct: **596/883** (67.50%).
- Potential reporting correction: **+29.90 percentage points** (paired bootstrap 95% CI 26.95–32.96).
- Execution correct but written answer wrong: **265**.
- CAUTION: program execution cannot detect a valid but semantically incorrect operation.

## FinQA invalid-program bug
- Original negative invalid-program rates are impossible; repaired files are in corrected_metrics/.
- No Phase-3 source prediction file is overwritten.

## Context, grounding, confidence
- Original Phase-3 maximum sequence length: 1024 tokens.
- Reaching the token cap is not itself proof of actual truncation.
- Numeric-literal grounding status: HEURISTIC_AUDIT_COMPLETED.
- Confidence measured with frozen task-specific prompts; no abstention threshold selected.

## Research gates
- **PASS** — original E5 freeze preserved
- **PASS** — all E1–E11 + prompt ablation prediction IDs align
- **PASS** — all invalid-program rates repaired and bounded
- **PASS** — FinQA answer/program reconciliation exported
- **PASS** — frozen task-specific confidence analyzed
- **PASS** — full 1,000-example validation used
- **PASS** — Phase-2.1 validation data fingerprint verified
- **PASS** — original official FinQA re-execution
- **PASS** — E5 checkpoint fingerprinted
- **PENDING** — 7 FinTrust evaluator protocols mapped

## Recommended next steps
1. Keep E5 frozen; use program execution to correct written numerical output.
2. Develop explicit operation/numeric-source/type verification on FinQA validation.
3. Fit verifier-aware selective-risk scores on validation calibration-fit and assess on calibration-holdout.
4. Apply official task-specific FinTrust evaluators only after verified mapping.
5. Lock Phase-4 methodology *before* opening protected test sets.

## Important reproducibility constraint
Local Phase-2.1 validation/registry fingerprints verified and the E5 adapter weights fingerprinted (see manifests/); no protected test data was accessed.