# TrustFin-SLM

**Trustworthy Small Language Models for Financial Tasks**

A research pipeline for building, validating, and evaluating trustworthy small language models (SLMs) on financial reasoning, sentiment analysis, FOMC stance classification, and multi-dimensional trust benchmarks.

---

## Project Structure

```
.
├── notebooks/                          # Jupyter notebooks (Phase-2.1)
│   ├── Phase2_1_TrustFin_*.ipynb       # Source notebook (clean)
│   └── Phase2_1_TrustFin_*_run.ipynb   # Executed notebook (with outputs)
├── docs/                               # Documentation & deep explanations
│   ├── exp_1.md                        # Phase-2 cell-by-cell walkthrough
│   ├── exp_2.md                        # Phase-2.1 cell-by-cell walkthrough
│   └── *.docx                          # Reference guides
├── papers/                             # Reference research papers (PDFs)
├── artifacts/
│   └── phase2_1/                       # Exported data contracts (Phase-2.1)
│       ├── datasets/
│       │   ├── train.jsonl             # SFT training (FinQA + FiQA)
│       │   └── validation.jsonl        # Validation + calibration
│       ├── eval/
│       │   ├── finqa_test.jsonl        # Protected FinQA test
│       │   ├── fiqa_test.jsonl         # Protected FiQA test
│       │   ├── finben_fomc_test.jsonl  # Protected FinBen-FOMC test
│       │   └── fintrust_registry.json  # Protected FinTrust registry
│       ├── manifests/
│       │   ├── dataset_registry.json
│       │   ├── source_fingerprints.json
│       │   ├── phase3_data_config.json
│       │   ├── phase4_eval_config.json
│       │   └── phase2_1_validation_report.json
│       ├── audits/
│       │   ├── finqa_official_gold_execution_audit.csv
│       │   └── near_duplicate_train_vs_protected_test.csv
│       └── python/
│           └── phase2_1_interface.py   # Downstream loader module
└── data/
    └── raw/                            # Downloaded raw data (re-runnable)
        ├── finqa/                      # FinQA GitHub raw files
        ├── finqa_official_evaluator/   # Official FinQA evaluate.py
        ├── fintrust_hf/               # FinTrust HF snapshot
        └── fintrust_eval_code/        # FinTrust evaluator repo clone
```

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
| **Validation** | FinQA dev + FiQA valid | Calibration, threshold selection |
| **Protected test** | FinQA test + FiQA test + FinBen-FOMC + FinTrust | Final evaluation only |

- FinBen-FOMC and FinTrust **never** enter SFT, calibration, or threshold tuning.
- All exported artifacts are fingerprinted with SHA-256 for reproducibility.
- The official FinQA executor validates every gold program.

## Phase-2.1 Validation Status

```
PHASE-2.1 STATUS:    PASS (68 gates, 0 failures)
READY FOR PHASE-3:   True
READY FOR PHASE-4:   True
```

## How to Re-run

```bash
# Install dependencies
pip install pandas numpy requests datasets huggingface_hub pyarrow scikit-learn tqdm datasketch sympy jupyter nbconvert ipykernel

# Run the notebook
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=3600 \
  --output "Phase2_1_TrustFin_*_run.ipynb" \
  --output-dir="./notebooks/" \
  "./notebooks/Phase2_1_TrustFin_Research_Validation_and_Data_Contract.ipynb"
```

## Schema Version

`trustfin-phase2.1-v1.0`

## Downstream Use

Phase-3 and Phase-4 notebooks should import `artifacts/phase2_1/python/phase2_1_interface.py` and read JSONL/JSON artifacts directly. Do not re-download raw datasets independently.

```python
from artifacts.phase2_1.python.phase2_1_interface import load_registry, load_train, load_validation

registry_path = "artifacts/phase2_1/manifests/dataset_registry.json"
train = load_train(registry_path)
validation = load_validation(registry_path)
```
