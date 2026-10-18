# TrustFin-SLM

**Trustworthy Small Language Models for Financial Tasks**

A research pipeline for building, validating, and evaluating trustworthy small language models
(SLMs) on financial reasoning (FinQA), financial sentiment (FiQA), FOMC stance classification,
and multi-dimensional trust benchmarks (FinTrust).

---

## Project Phases

| Phase | Notebook(s) | Deep-dive doc | Status | Artifacts |
|---|---|---|---|---|
| **Phase-2** — Data integrity & unified pipeline | `notebooks/Phase2_TrustFin_Data_Integrity_and_Unified_Pipeline.ipynb` | `docs/exp_1.md` | PASS | `artifacts/phase2/` |
| **Phase-2.1** — Research validation & data contract | `notebooks/Phase2_1_TrustFin_Research_Validation_and_Data_Contract.ipynb` | `docs/exp_2.md` | **PASS** (68 gates, 0 failures) | `artifacts/phase2_1/` |
| **Phase-3 (smoke)** — pipeline dry-run | `notebooks/Phase3_smoke_executed.ipynb` | `docs/exp_3(a).md` | PASS | `artifacts/phase3_smoke_16of16.tar.gz` |
| **Phase-3 (full)** — baselines & failure analysis | `Phase3_TrustFin_SLM_Baselines_and_Failure_Analysis.ipynb` | **`docs/exp_3(b).md`** | **PASS · FROZEN** | `artifacts/phase3/` |
| **Phase-4** — EVG-CAS (proposed method) | `trust_fin_extra_phase/notebooks/Phase4_TrustFin_Proposed_Method.ipynb` | **`docs/exp_4(a).md`**, **`docs/exp_4(b).md`** | **COMPLETE · FROZEN** (validation + protected) | `trust_fin_extra_phase/artifacts/phase4/` |
| **Showcase** — paper & related work | `paper/TrustFin_Conference_Paper.tex`/`.md` | `docs/related_work.md`, `docs/literature_reading_matrix.md`, `docs/trustfin_references.bib` | draft v1 | — |

Phase-3 selects the Phase-4 backbone on **validation data only**; the frozen winner is **E5**
(`Qwen/Qwen3-1.7B` + balanced joint QLoRA). Contract: `artifacts/phase3/manifests/phase3_freeze.json`.

Phase-4 headline (pre-registered, protected test): C1 rejected (accuracy ties the baseline),
C2/C3 supported on FinQA — a frozen abstention threshold cuts protected-test answered-error
by **31.0%** at 80% coverage (validation predicted 34.2%). Summary:
`trust_fin_extra_phase/docs/exp_4(b).md` §7. Statistical rigor (paired McNemar, bootstrap CIs,
error taxonomy, difficulty stratification): `trust_fin_extra_phase/docs/eval_rigor.md`.

---

## Repository Layout

```
.
├── README.md                              # this file
├── Phase3_TrustFin_SLM_Baselines_and_Failure_Analysis.ipynb   # Phase-3 master notebook
│
├── docs/                                  # deep-dive write-ups (per phase)
│   ├── exp_1.md                           #   Phase-2  cell-by-cell walkthrough
│   ├── exp_2.md                           #   Phase-2.1 walkthrough
│   ├── exp_3(a).md                        #   Phase-3 smoke run
│   ├── exp_3(b).md                        #   Phase-3 full run
│   ├── trustfin_references.bib            #   29-entry bibliography (6 tagged families)
│   ├── related_work.md                    #   related-work taxonomy + positioning gap
│   ├── literature_reading_matrix.md       #   CORE-read matrix (15 rows) + claimed mappings
│   └── Research_Paper_Reading_and_Conference_Drafting_Guide.docx
│
├── paper/                                 # conference paper draft (LaTeX + markdown mirror)
│   └── TrustFin_Conference_Paper.tex / .md
│
├── phases/                                # per-phase run archives (phase-wise grouping)
│   └── phase3/
│       ├── README.md                      #   Phase-3 run notes
│       ├── RESUME_phase3_2026-10-10.md
│       ├── notebooks/                     #   executed per-part notebooks (8 parts + smoke)
│       └── logs/                          #   nbconvert + supervisor logs
│
├── notebooks/                             # Phase-2 / Phase-2.1 notebooks (+ Phase-3 smoke)
├── tools/                                 # Phase-3 sharding & supervisor drivers
│   ├── phase3_shard.py                    #   per-part nbconvert launcher / verifier
│   └── phase3_supervisor.sh               #   sequential part driver with GPU cool-down
├── deliverables/                          # shareable submission bundles (.zip)
│   ├── Phase2_1_TrustFin_Deliverable.zip
│   └── Phase3_TrustFin_Deliverable.zip
├── artifacts/                             # frozen data contracts & run outputs
│   ├── phase2/                            #   Phase-2 exports
│   ├── phase2_1/                          #   Phase-2.1 contract (FROZEN, SHA-256)
│   └── phase3/                            #   Phase-3 metrics, analysis, plots, checkpoints
├── data/raw/                              # downloaded raw data (re-runnable)
├── papers/                                # reference research papers (PDFs)
└── trust_fin_extra_phase/                 # Phase-4 EVG-CAS (notebooks, artifacts, analysis)
    ├── docs/                              #   exp_4(a).md, exp_4(b).md, eval_rigor.md
    ├── scripts/                           #   eval_rigor.py (statistical rigor on frozen artifacts)
    └── artifacts/phase4/                  #   frozen predictions, metrics, freeze manifest
```
And `trustfin_slm_workspace/` is the symlink shim → `artifacts/`, `data/` (absolute-path contract).
```

> **Note on the frozen contract.** `artifacts/phase2_1/manifests/dataset_registry.json` stores
> **absolute paths** to `artifacts/` and `data/`. Those two directories plus `papers/` are kept in
> place deliberately — moving them would invalidate the frozen SHA-256 contract. Only loose
> Phase-3 run files are grouped into `phases/`.

---

## Datasets

| Dataset | Source | Role | Splits |
|---------|--------|------|--------|
| **FinQA** | `czyssrs/FinQA` | Financial numerical reasoning | train / dev / test |
| **FiQA Sentiment** | `TheFinAI/fiqa-sentiment-classification` | Aspect-aware sentiment regression [-1,1] | train / valid / test |
| **FinBen-FOMC** | `TheFinAI/finben-fomc` | FOMC stance classification | test only (eval-only) |
| **FinTrust** | `HughieHu/FinTrust` | 7-dimension trust benchmark | all (eval-only registry) |

## Training / Evaluation Policy

| Pool | Datasets | Purpose |
|------|----------|---------|
| **Training** | FinQA + FiQA | SFT fine-tuning |
| **Validation** | FinQA dev + FiQA valid | Calibration, threshold selection, model selection |
| **Protected test** | FinQA test + FiQA test + FinBen-FOMC + FinTrust | Final evaluation only |

- FinBen-FOMC and FinTrust **never** enter SFT, calibration, or threshold tuning.
- All exported artifacts are fingerprinted with SHA-256 for reproducibility.
- The official FinQA executor validates every gold program.

## Phase Status

```
PHASE-2.1 STATUS:    PASS (68 gates, 0 failures)   ·  READY FOR PHASE-3 / PHASE-4: True
PHASE-3   STATUS:    PASS  ·  All E1-E16 outputs present: True  ·  Freeze status: FROZEN
```

### Phase-3 headline results (validation only)

| Experiment | Model | Sampling | FinQA exec | FiQA MAE | FiQA R² |
|---|---|---|---|---|---|
| **E5** ⭐ | Qwen3-1.7B | balanced joint | **0.675** | 0.130 | 0.730 |
| E11 | Qwen3-1.7B | natural joint | 0.615 | 0.128 | 0.736 |
| E8 | Phi-4-mini | balanced joint | 0.701 | 0.130 | 0.809 |

→ Frozen backbone: **E5** (selection score 0.7446). Full detail: `docs/exp_3(b).md`.

---

## How to Re-run

```bash
# Phase-2.1 data contract
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=3600 \
  --output "Phase2_1_TrustFin_Research_Validation_and_Data_Contract_run.ipynb" \
  --output-dir="./notebooks/" \
  "./notebooks/Phase2_1_TrustFin_Research_Validation_and_Data_Contract.ipynb"

# Phase-3 full run (sharded; see phases/phase3/README.md and docs/exp_3(b).md)
bash tools/phase3_supervisor.sh
```

## Schema Version

`trustfin-phase2.1-v1.0`

## Downstream Use

Phase-3 and Phase-4 notebooks import `artifacts/phase2_1/python/phase2_1_interface.py` and read
JSONL/JSON artifacts directly. Do not re-download raw datasets independently.

```python
from artifacts.phase2_1.python.phase2_1_interface import load_registry, load_train, load_validation

registry_path = "artifacts/phase2_1/manifests/dataset_registry.json"
train = load_train(registry_path)
validation = load_validation(registry_path)
```
