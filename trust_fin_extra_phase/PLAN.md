# Phase-4 Plan — Execution-Verified Generation + Calibrated Abstention (EVG-CAS)

**Status:** 4.1 — **validation + protected stages complete (2026-10-12).** C1 rejected, C2/C3
supported on FinQA (see `docs/exp_4(a).md`, `docs/exp_4(b).md`). P4-E9 (FinTrust) deferred.
**Evaluation rigor pass:** paired McNemar + bootstrap CIs, error taxonomy, difficulty
stratification, split-labeled literature positioning (see `docs/eval_rigor.md`,
`scripts/eval_rigor.py`). **Showcase:** related-work corpus (`../docs/related_work.md`,
`../docs/trustfin_references.bib`, `../docs/literature_reading_matrix.md`) and paper draft
(`../paper/TrustFin_Conference_Paper.tex`/`.md`) at draft v1.
**Phase-4.2 (review-driven, 2026-10-12):** matched-coverage selective-prediction comparison,
component ablations, bootstrap CIs, and error-taxonomy reconciliation — computed only from
the frozen artifacts and written to `artifacts/phase4_2/` (see `docs/exp_4(c).md`, notebook
`Phase4_2_Evaluation_Improvement.ipynb`). Corrected self-contained paper draft v2 in
`paper/`.
**Runs in:** `trust_fin_extra_phase/` (this folder)
**Reads (read-only):** outer `../artifacts/phase2_1/`, `../artifacts/phase3/`, `../data/raw/`
**Writes:** `trust_fin_extra_phase/artifacts/phase4/` (full runs only).
Smoke runs write to `trust_fin_extra_phase/artifacts/phase4_smoke/` so a smoke freeze or a
cached smoke candidate file can never satisfy a full run's gates.
**Guides:** guide §15 stage 12 (proposed-method implementation), stage 18 (Experiments/Results for it); §19 checklist.

---

## 1. Why this method (evidence from Phase-3, not vibes)

Phase-3 proved three concrete defects in the frozen backbone **E5**
(`Qwen/Qwen3-1.7B` + balanced joint QLoRA, FinQA exec 0.675 / FiQA MAE 0.130):

| # | Phase-3 finding | Source | Consequence |
|---|---|---|---|
| D1 | **96 % of FinQA errors are execution/reasoning errors**, only 0.3 % invalid programs | `analysis/E12_finqa_failure_summary.csv` | The model *writes programs fine* but picks the wrong operation chain → re-verify by executing. |
| D2 | Confidence is **only weakly informative** (ROC-AUC 0.577, Spearman 0.125) | `analysis/E14_confidence_bins.csv` | Raw confidence cannot be thresholded → needs cross-fitted calibration. |
| D3 | Balancing helped (+6 pp) but nothing else is gated: no abstention, no OOD trust eval | `docs/exp_3(b).md` §7 | Trustworthiness claim is unquantified → selective prediction + FOMC OOD. |

## 2. Novelty claims (guide stage 9 — pre-declared, cannot be re-stated after seeing test)

- **C1 (EVG).** *Execution-verified self-consistency*: sample *n* candidate programs, execute
  each with the **official FinQA executor on the input table**, select the answer by majority
  vote over executable candidates (ties → higher mean token log-probability). No gold program,
  no gold answer is used — only the table, which is already part of the model's input.
- **C2 (CAS).** *Cross-fitted calibrated abstention*: Platt and isotonic calibrators fitted
  **within validation only** (5-fold cross-fitting so the reported validation risk-coverage is
  out-of-fold, not in-sample), then refit on full validation and **frozen before any protected
  test is opened**.
  **Tested on probability quality (Brier / ECE / NLL) and on risk@coverage at the pre-declared
  coverages**, i.e. does calibration give the abstention threshold usable probability meaning.
  It is *not* tested on AURC: a single-feature monotone calibrator refitted on the same scores
  it is applied to cannot change their ranking, so it cannot change AURC.
- **C3 (Joint).** EVG supplies extra features (vote agreement, executable-program rate) that
  materially improve the calibrator over confidence alone — i.e. verification and abstention
  are complementary, not redundant. **This is the only claim that can be tested on ranking**,
  because `platt_conf_full` is not a monotone map of confidence. Test = lower out-of-fold
  `e_AURC` for `platt_conf_full` than for `raw_confidence`/`platt_conf`/`isotonic_conf`.
  A frame where nothing reorders the records is reported as a **negative result**, not a bug.

Anti-novelty guard: the backbone, adapter, prompt policy and selection rule from
`../artifacts/phase3/manifests/phase3_freeze.json` **do not change**. Phase-4 proposes an
*inference-time* method, which is the only class of method available without re-selecting the
backbone.

## 3. Contract inherited from Phase-3 (must not be violated)

| Item | Frozen value |
|---|---|
| Backbone | `E5` — `../artifacts/phase3/checkpoints/E5` |
| Base model | `Qwen/Qwen3-1.7B` |
| FinQA prompt | `canonical` · FiQA prompt `explicit_schema` |
| Selection | validation-only; Phase-4 must not re-select the backbone |
| Calibration | fitted on validation only (`phase4_eval_config.json → calibration_split`) |
| Protected tests | `finqa_test`, `fiqa_test`, `finben_fomc_test`, `fintrust_registry` — **opened last** |
| Official executor | `../data/raw/finqa_official_evaluator/evaluate.py` (SHA-256 verified) |

## 4. Experiments

| ID | Name | Data | Cost |
|---|---|---|---|
| **P4-E1** | Baseline + calibrator benchmark from frozen greedy predictions | validation (loaded from `../artifacts/phase3/predictions/E5.csv`) | none (no GPU) |
| **P4-E2** | EVG sampling n = 8 (n=2, n=4 derived as prefixes of the same 8) | FinQA validation 883 | ~1 pass |
| **P4-E3** | EVG selection ablation: majority vs executable-only vs logprob-rank | FinQA validation | free (derived) |
| **P4-E3b** | FiQA self-consistency (median of 8 sampled scores vs greedy) | FiQA validation 117 | ~1 pass |
| **P4-E4** | Platt cross-fitted (5-fold) on confidence | FinQA + FiQA validation | none |
| **P4-E5** | Isotonic cross-fitted (5-fold) | FinQA + FiQA validation | none |
| **P4-E6** | Combined Platt(confidence + EVG agreement + executable flag) | FinQA validation | none |
| **P4-E7** | Selective-prediction curves → **freeze `phase4_freeze.json`** | validation only | none |
| **P4-E8** | **Protected test final evaluation** (only after status `FROZEN`) | FinQA test 1147, FiQA test 234, FOMC 496 | ~1–2 passes |
| **P4-E8e** | Apply the **frozen** calibrator + frozen thresholds to the protected test | FinQA + FiQA test | none (no refit) |
| **P4-E9** | FinTrust official per-dimension eval (optional, deferred) | FinTrust 7 dimensions | separate |

Smoke mode (`RUN_MODE=smoke`) caps validation at 20/dataset and EVG at n=2, and writes to
`artifacts/phase4_smoke/` (never `artifacts/phase4/`).

## 5. Metrics (fixed by `phase4_eval_config.json`)

- **FinQA:** official execution/program accuracy, answer accuracy, parse success, answer-program consistency.
- **FiQA:** MSE, R², MAE, Pearson r (clip rule reported explicitly: predictions clipped to [-1,1]).
- **Selective prediction:** coverage, selective risk, **AURC**, E-AURC, accuracy@coverage ∈ {0.6, 0.7, 0.8, 0.9, 0.95}, coverage@risk targets.
- **Calibration:** Brier / NLL and ECE-style bins for the binary correctness target.
- **FOMC:** accuracy, macro-F1, per-class F1 (zero-shot, never tuned).
- **Efficiency:** latency, tok/s, VRAM peak — EVG's *n×* cost must be reported, not hidden.
- **Metric→claim map:** C2 ← Brier/ECE/NLL + risk@coverage; C3 ← out-of-fold e_AURC of
  `platt_conf_full` vs the single-feature methods. AURC differences between a monotone
  calibrator and raw confidence are an artifact of the cross-fitting folds, not a reordering,
  and are not claimed as evidence.

## 6. Protocol guardrails (enforced by asserts in the notebook)

1. Protected artifact keys are **not resolvable** until `P4-E8` and only if
   `phase4_freeze.json.status == "FROZEN"` — the loader raises otherwise.
2. No hyper-parameter (n, temperature, calibrator type, threshold) is selected using any
   protected record. n and temperature are pre-declared here: **n = 8, temperature = 0.7,
   top_p = 0.9, 80 % of max_new_tokens** budget unchanged (FinQA 192, FiQA 24).
3. Calibrator fits use `sklearn` with fixed folds (`KFold(5, shuffle=True, random_state=42)`).
4. `../artifacts/**` is opened read-only; every Phase-4 output lands under this folder.
5. Thresholds are chosen at **pre-declared coverage targets** on validation, never by peeking
   at test risk.
6. **Nothing is refitted on test.** P4-E8e applies the calibrator objects and thresholds
   written by P4-E7; the protected stage contains no `fit()` call.

## 7. Directory layout (this folder)

```
trust_fin_extra_phase/
├── PLAN.md                                   # this file (4.0)
├── README.md                                 # how to run + results index
├── notebooks/Phase4_TrustFin_Proposed_Method.ipynb
├── docs/exp_4(a).md                          # validation deep-dive + smoke harness (pre-protected)
├── docs/exp_4(b).md                          # protected-test deep-dive + final claims verdict
├── tools/phase4_shard.py                     # nbconvert launcher/verifier
├── logs/                                     # run logs
├── artifacts/phase4_smoke/                   # RUN_MODE=smoke only (never read by full gates)
└── artifacts/phase4/                         # RUN_MODE=full only
    ├── predictions/   P4-E* per-record outputs
    ├── metrics/       P4-E*.json
    ├── analysis/      ablation + risk-coverage CSVs
    ├── plots/         risk-coverage, EVG gain figures
    └── manifests/     phase4_freeze.json, environment.json, plan.json
```

## 8. Budget

| Stage | GPU | Wall-clock |
|---|---|---|
| P4-E1/E3/E4/E5/E6/E7 (CPU only) | — | minutes |
| P4-E2 + E3b (sampling n=8) | ~4 GB peak | ~1–1.5 h |
| P4-E8 protected test | ~4 GB peak | ~1–2 h |
| **Total** | fits 6 GB RTX 4050 | **~2–4 h**, shardable |

## 9. Definition of "research phase done" (guide §15)

Stages 11/13/14 done by Phase-3. Phase-4 completes **12** and **18**. Remaining after that are
writing-only: 15 Intro, 16 Related Work (needs `papers/` ≥ 10–15 core refs; only 7 PDFs now),
17 Methodology, 19 Abstract+Conclusion, 20 formatting — plus §19 checklist items
"I have results for the proposed method" / "ablation experiments supporting the novelty claims".
