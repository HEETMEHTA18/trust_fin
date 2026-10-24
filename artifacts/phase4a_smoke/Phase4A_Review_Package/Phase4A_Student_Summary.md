# Phase-4A Student Review Summary

- Execution mode: **smoke**
- Core audit: **SMOKE_PASS**
- Validation: FinQA **35**, FiQA **15**
- A1 written-answer accuracy: **62.86%**
- A1 executed-program accuracy: **77.14%**
- A2 gain: **14.29 percentage points**
- A3 structurally valid: **35/35**
- A4 any unmatched numbers: **0/35**
- A4 evidence provenance: **RECOMPUTED_FROM_PHASE2_1_VALIDATION**
- Local new official executor: **NOT_RUN**
- E5 adapter bytes locally rechecked: **NOT_AVAILABLE_LOCALLY**

## Interpretive cautions

- Valid execution corrects text arithmetic but does not certify the operations or source numbers.
- A number appearing in a financial document does not establish that it supports a question.
- Feature diagnostics use **calibration-fit only**; holdout labels are reserved for Phase-4B.
- Feature JSONL files intentionally exclude targets; label JSONL files are separate.

## Next

- Phase-4B primary features: source likelihood, prompt length and truncation; hard reject unexecutable programs.
- Phase-4B grounding extension: literal mismatch ONLY after original validation input recomputation.
- Answer/execution agreement is an ablation-only signal (nearly no error discrimination in the fit subset).
- Phase-4B: compare likelihood-only versus verifier-aware risk estimation on calibration-fit; run a single holdout assessment.
- Phase-4C: compare full method to execution-only baseline; do not use protected tests before methodology freeze.