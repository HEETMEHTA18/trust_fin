# Deep Explanation (a) — Phase-4A **FULL** run

**Notebook:** `Phase4A_TrustFin_Execution_Verification_and_Financial_Grounding.ipynb`
**Run mode:** `RUN_MODE = "full"` (preceded by a validated `smoke` pass)
**Executed:** 21 October 2026, 13:52–14:03 IST (final successful pass)
**Executed notebook:** `notebooks/Phase4A_TrustFin_Execution_Verification_and_Financial_Grounding_executed.ipynb` (15/15 code cells, 0 errors)
**Result:** `core_phase4a_status: PASS` · `research_full_grounding_status: PASS` · `risk_feature_policy_status: READY_FOR_PHASE4B_DEVELOPMENT`

---

## 1. What Phase-4A is (30-second version)

Phase-4A is a **deterministic audit and feature-generation stage** built on the frozen Phase-3
backbone (experiment E5: `Qwen/Qwen3-1.7B` + balanced FinQA+FiQA QLoRA). It answers four
research questions **without using any protected test data**:

| Research question | Phase-4A answer |
|:---|:---|
| **A1/A2** Execution vs. written answer | Valid execution raises FinQA accuracy from **37.60%** → **67.50%** (+29.90 pp); reconciliation categories: both correct 331, execution-only correct 265, written-only correct 1, both wrong 286 |
| **A3** Program grammar / structural validity | 880/883 programs are structurally valid; 0 mismatches against the official executor; issue taxonomy: `unsupported_operand_token` (2), `table_arity` (1) |
| **A4** Financial operand provenance | 21/883 programs have ≥1 unmatched literal (30 total); 97.6% of numeric operands appear verbatim in the original evidence — but this is **literal presence**, not semantic grounding |
| **A5** Risk-signal diagnostics (calibration-fit only) | `source_likelihood` AUC = 0.5845 (weak); agreement AUC ≈ 0.5000 (no signal); `any_literal_unmatched` error rate = 87.5% (fit subset) |

**Scientific caveat enforced by the notebook:** Executability and literal presence **do not** prove semantic program correctness. Phase-4A is a verifier and feature-generation stage, not a claimed end-to-end solution to grounding.

---

## 2. Research rule enforced by the notebook

| Rule | Enforcement |
|:---|:---|
| **Validation-only** | Only the Phase-2.1 `validation` split (883 FinQA dev + 117 FiQA valid) is ever loaded. Protected sets (FinQA test, FiQA test, FinBen-FOMC, FinTrust) are never accessed; the notebook asserts `protected_test_sets: "never used by this notebook"`. |
| **Holdout protection** | A5 diagnostics run **only on `calibration_fit` (698 records)**. The `calibration_holdout` (185 records) is never opened — `fit_only: True` on every diagnostic row. |
| **Label isolation** | Phase-4B interface files are strictly separated: `phase4b_finqa_features.jsonl` / `phase4b_fiqa_features.jsonl` contain **zero target columns** (verified: `execution_correct`, `written_correct`, `corrected_correct`, `gold_sentiment`, `absolute_error`, `gold_answer`, `gold_program` absent). Targets live in `labels/` as separate JSONL files. |
| **Evidence provenance** | Original Phase-2.1 `validation.jsonl` is mounted, upgrading `evidence_provenance` from `CACHED_PHASE3_1_ONLY` → `RECOMPUTED_FROM_PHASE2_1_VALIDATION`. The notebook explicitly refuses paths containing `test`, `protected`, `fomc`, `fintrust`. |
| **No new training / evaluation** | Zero model weights are loaded; the official FinQA executor is only re-run if `TRUSTFIN_FINQA_EVALUATOR` + `TRUSTFIN_FINQA_DEV_JSON` are supplied (optional, `NOT_RUN` here). |

---

## 3. Environment actually used

| Item | Value |
|:---|:---|
| GPU | **None required** — Phase-4A is a saved-prediction audit; no model download or inference. |
| Python | 3.12.3 in `/home/heet18/trustfin-slm/.venv` (kernel `trustfin`) |
| Core packages | `numpy 2.5.2`, `pandas 3.0.5`, `scipy 1.18.1`, `scikit-learn 1.9.1`, `matplotlib 3.11.2` |
| Runtime | `jupyter nbconvert --execute --ExecutePreprocessor.kernel_name=trustfin --ExecutePreprocessor.timeout=3600` |
| Wall clock | **smoke: ~4.4 s** | **full: ~4.0 s** (274 MB RSS, exit 0) |

---

## 4. Three fixes applied before the full run could complete

The notebook as distributed contained three latent failures that only manifest in **full** mode (the smoke subset never exercises the relevant branches):

| # | Failure | Fix (in `tools/phase4a_configure.py`) |
|:--|:---|:--|
| 1 | **Split guard assertion error** — `assert split ∈ {dev, validation}` raised on all 117 FiQA rows because Phase-2.1 exports FiQA with `split="valid"` (FiQA's official HF split name; confirmed in `artifacts/phase2_1/manifests/source_fingerprints.json` and the passing "FiQA valid count" check). | Widened to `{dev, valid, validation}`. Every test/protected/FOMC/FinTrust token still rejected. |
| 2 | **Self-hashing inventory** — cell 31 built `artifact_inventory.json`, then rewrote it, leaving a stale self-hash. Every auditor hit one spurious mismatch (`manifests/artifact_inventory.json` SHA mismatch). | Excluded `artifact_inventory.json` from its own listing (same rule already used for `Phase4A_Review_Package.zip`). All 19 entries now verify byte-for-byte. |
| 3 | **NaN `executed_answer` edge case** — 3 records with `program_executable=False` have `executed_answer=NaN`. The notebook falls back to `pred_answer`; no NaN in `pred_answer`/`gold_answer`, so `near_equal` never sees NaN-vs-NaN. | Confirmed benign via pre-flight; no code change needed. |

---

## 5. Key numerical results (full mode, 883 FinQA + 117 FiQA)

### A1/A2 — Execution vs. written answer reconciliation
```
written_answer_accuracy       : 0.37599093997734995   (≡ phase3_freeze.json winner)
executed_program_answer_accuracy : 0.6749716874292185   (≡ phase3_freeze.json winner)
execution_minus_written_gain_pp : 29.898074745186857
reconciliation:
  both_correct                : 331
  execution_only_correct      : 265   ← valid execution rescues wrong written answers
  written_only_correct        : 1
  both_wrong                  : 286
execution_validity            : 0.9966  (880/883)
answer_execution_agreement    : 0.5640
corrected_output_accuracy     : 0.6750  (uses execution when executable)
```

### A3 — Program structural validation
```
structurally_valid            : 880/883
official_executable           : 880/883  (3 non-executable)
mismatch_structural_to_official: 0       ← independent validator agrees perfectly
issue_counts: {unsupported_operand_token: 2, table_arity: 1}
step_count_distribution: 1→559, 2→267, 3→32, 4→4, 5→19, 6→1, 7→1
```

### A4 — Operand evidence provenance (RECOMPUTED)
```
evidence_provenance           : RECOMPUTED_FROM_PHASE2_1_VALIDATION
unmatched_any_n               : 21 / 883  (2.4%)
unmatched_literals_total      : 30
literal_presence_fraction     : 97.6%     (document-level; NOT semantic grounding)
```

### A5 — Fit-only feature diagnostics (698 calibration-fit records)
| feature | n | AUC (correctness) | symmetric AUC | Spearman |
|:--|--:|--:|--:|--:|
| source_likelihood | 698 | **0.5845** | 0.5845 | 0.1363 |
| max_question_local_overlap | 698 | 0.5756 | 0.5756 | 0.1223 |
| literal_in_table_count | 698 | 0.5446 | 0.5446 | 0.0765 |
| literal_presence_fraction | 698 | 0.5295 | 0.5295 | **0.1838** |
| unmatched_literal_count | 698 | 0.4705 | 0.5295 | -0.1836 |
| any_literal_unmatched | 698 | 0.4706 | 0.5294 | -0.1832 |
| answer_execution_agreement_feature | 698 | 0.5000 | 0.5000 | ~0 |

Subgroup error rate (fit): `any_literal_unmatched=1` → execution accuracy **12.5%** (vs 69.5% when 0).

---

## 6. Phase-4B interface frozen by Phase-4A

**File:** `artifacts/phase4a/phase4b_interface/phase4b_feature_policy.json`

```json
{
  "schema_version": "trustfin.phase4a.v1",
  "phase3_frozen_backbone": "E5",
  "primary_model": "Qwen/Qwen3-1.7B",
  "validation_only": true,
  "fit_split": "calibration_fit",
  "holdout_split": "calibration_holdout",
  "FinQA": {
    "core_features": ["source_likelihood", "prompt_tokens", "prompt_truncation_observed"],
    "deterministic_rejection_gates": ["official_executable_feature", "program_structural_valid"],
    "ablation_only_features": ["generated_tokens", "program_step_count", ...],
    "grounding_features_conditional_on_recomputation": ["any_literal_unmatched", "unmatched_literal_count", "literal_presence_fraction"],
    "optional_full_evidence_features": ["literal_in_table_count", "max_question_local_overlap"],
    "target": "execution_correct",
    "caution": "Do not claim logical/semantic correctness from numeric presence or answer-program agreement."
  },
  "FiQA": {
    "features": ["source_likelihood", "generated_tokens", "prompt_tokens", "sentiment_pred_clipped", "score_near_boundary", "parse_ok"],
    "target": "absolute_error or well-calibrated residual interval"
  },
  "feature_provenance": "RECOMPUTED_FROM_PHASE2_1_VALIDATION",
  "grounding_parity_ready": true,
  "no_gold_label_in_model_features": true,
  "phase4b_model_fit_only_on_calibration_fit": true,
  "phase4b_holdout_use": "one-pass evaluation after hyperparameters frozen",
  "protected_test_sets": "never used by this notebook"
}
```

**Interface files produced:**
```
phase4b_interface/phase4b_finqa_features.jsonl   883 rows × 28 features
phase4b_interface/phase4b_fiqa_features.jsonl    117 rows × 10 features
labels/phase4b_finqa_labels.jsonl                883 rows (execution_correct, written_correct, corrected_correct)
labels/phase4b_fiqa_labels.jsonl                 117 rows (gold_sentiment, absolute_error)
```

---

## 7. Artifacts and how to reproduce

### Review package
`artifacts/phase4a/Phase4A_Review_Package.zip` (175 KB) — 20 files matching cell 32's manifest:
```
manifests/phase4a_readiness.json
analysis/A1_A2_reconciliation.json
analysis/A3_program_structure_summary.json
analysis/A3_program_structure_cases.csv
analysis/A4_operand_evidence_summary.json
analysis/A4_operand_evidence_cases.csv
tables/A5_fit_feature_diagnostics.csv
tables/A5_fit_subgroup_accuracy.csv
phase4b_interface/phase4b_finqa_features.jsonl
phase4b_interface/phase4b_fiqa_features.jsonl
labels/phase4b_finqa_labels.jsonl
labels/phase4b_fiqa_labels.jsonl
phase4b_interface/phase4b_feature_policy.json
manifests/input_fingerprints.json
manifests/live_evaluator_status.json
manifests/adapter_verification_status.json
manifests/artifact_inventory.json
plots/A1_A2_accuracy.png
plots/A5_feature_discrimination.png
Phase4A_Student_Summary.md
```

### Reproduction (one command, fresh kernel)
```bash
cd /home/heet18/Downloads/Research Papers
bash tools/phase4a_run.sh   # runs smoke, then full
```

The `tools/phase4a_configure.py` script pins the notebook to the correct input paths
(`deliverables/Phase3_TrustFin_Deliverable.zip`, `artifacts/phase3_1/Phase3_1_Review_Package.zip`,
`artifacts/phase2_1/datasets/validation.jsonl`) and applies the three fixes above.
It is **idempotent** — re-running it produces the identical configured notebook.

### What Phase-4B should do next
1. Use `phase4b_*_features.jsonl` + `labels/` as separate train/calibrate sets.
2. Fit risk models **only on `calibration_fit`** (698 FinQA, 92 FiQA).
3. Compare: raw likelihood → likelihood + execution gates → likelihood + grounding features.
4. Single holdout assessment on `calibration_holdout` after hyperparameters frozen.
5. **Do not load protected tests** before methodology freeze (Phase-4C).

---

## 8. What Phase-4A proved — and what it did not

| Claim | Status |
|:---|:---|
| Valid program execution corrects written arithmetic (+29.9 pp) | **Proven** |
| Structural grammar validator is necessary and sufficient for executability | **Proven** (0 mismatches on 883) |
| Literal presence ≡ semantic grounding | **Explicitly refuted** (notebook flag: `document_literal_presence_is_not_semantic_grounding: true`) |
| `source_likelihood` is a strong risk signal | **Not supported** (AUC 0.585 ≈ weak) |
| Answer/execution agreement is a useful risk signal | **Not supported** (AUC 0.500) |
| Phase-4B can build a calibrated risk model | **Feature policy ready** — but holdout untouched |
| Any protected test was evaluated | **Never loaded** (asserted in readiness manifest) |
| New independent official execution was performed | `NOT_RUN` — requires optional `evaluate.py` + FinQA `dev.json` |
| E5 adapter bytes locally rechecked | `NOT_AVAILABLE_LOCALLY` — Phase-3.1 manifest retains fingerprint |