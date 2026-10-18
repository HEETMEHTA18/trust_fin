# Phase 2 and Phase 2.1 Evidence Inventory

This inventory separates Phase 2 data-contract evidence from later model-result plots. Phase 2/2.1 is a validation and gating stage; it does not produce model-accuracy curves.

## Architecture and methodology diagrams

| Item | Location | Purpose |
|---|---|---|
| Phase-2.1 architecture | `trust_fin_extra_phase/paper/TrustFin_Conference_Paper.tex`, `fig:phase2-architecture` | Raw sources to normalized records, gates, frozen exports, Phase 3, and Phase 4 |
| Phase-2.1 gate lifecycle | `trust_fin_extra_phase/paper/TrustFin_Conference_Paper.tex`, `fig:phase2-gates` | Schema, official execution, leakage, near-duplicate, and freeze decision path |
| Source notebook data-flow diagram | `docs/exp_2.md` and `notebooks/Phase2_1_TrustFin_Research_Validation_and_Data_Contract.ipynb` | Operational notebook-level data flow |

## Phase 2.1 gate evidence

| Evidence | Path |
|---|---|
| Validation status and gate ledger | `artifacts/phase2_1/manifests/phase2_1_validation_report.json` |
| Dataset registry and split roles | `artifacts/phase2_1/manifests/dataset_registry.json` |
| Source fingerprints | `artifacts/phase2_1/manifests/source_fingerprints.json` |
| Phase 3 data contract | `artifacts/phase2_1/manifests/phase3_data_config.json` |
| Phase 4 evaluation contract | `artifacts/phase2_1/manifests/phase4_eval_config.json` |
| Official FinQA gold execution audit | `artifacts/phase2_1/audits/finqa_official_gold_execution_audit.csv` |
| Train versus protected-test near-duplicate audit | `artifacts/phase2_1/audits/near_duplicate_train_vs_protected_test.csv` |
| Stable downstream loader | `artifacts/phase2_1/python/phase2_1_interface.py` |

## Result plots available in the workspace

All publication copies are stored in `paper/figures/`. They are generated from the
repository artifacts by `tools/prepare_paper_figures.py`.

### Phase 3 plots

| Plot | Path | Result represented |
|---|---|---|
| FinQA execution accuracy | `artifacts/phase3/plots/finqa_execution_accuracy.png` | Phase-3 validation baseline comparison |
| FiQA MAE | `artifacts/phase3/plots/fiqa_mae.png` | Phase-3 sentiment-regression baseline comparison |
| FinQA execution accuracy, publication copy | `paper/figures/fig_phase3_finqa_execution_accuracy.png` | Human-readable Phase-3 baseline comparison |
| FiQA MAE, publication copy | `paper/figures/fig_phase3_fiqa_mae.png` | Human-readable Phase-3 regression comparison |

### Phase 4 plots

| Plot | Path | Result represented |
|---|---|---|
| FiQA sampling aggregation | `paper/figures/fig_phase4_fiqa_aggregation.png` | Sampled-prediction aggregation diagnostic |
| Risk-coverage curves | `paper/figures/fig_phase4_risk_coverage.png` | Selective-risk behavior on validation |
| EVG selection ablation | `paper/figures/fig_phase4_selection_ablation.png` | Candidate-count and selection-rule ablation |

## Existing diagnostic figures

The following figures are Phase-3 failure-analysis outputs, not Phase-2 gate plots:

- `paper/figures/fig1_finqa_failure_taxonomy.png`
- `paper/figures/fig2_finqa_failure_donut.png`
- `paper/figures/fig3_finqa_confidence_by_failure.png`
- `paper/figures/fig4_finqa_confidence_bins_accuracy.png`
- `paper/figures/fig5_finqa_error_magnitude.png`
- `paper/figures/fig6_finqa_failure_overview.png`
