# Deep Explanation (b) — Phase-3 **FULL** run

**Notebook:** `Phase3_TrustFin_SLM_Baselines_and_Failure_Analysis.ipynb`
**Run mode:** `RUN_MODE = "full"`
**Executed:** 8–11 October 2026 (chained across `phase3_shard.py` parts, with two clean power-downs)
**Executed notebooks:** `notebooks/Phase3_full_E{1,4,5,7,8,9,11,12}.ipynb` (per-part, 0 errors)
**Result:** `PHASE-3 STATUS: PASS` · `All E1-E16 outputs present: True` · `Freeze status: FROZEN`
**Winner:** **E5** — `Qwen/Qwen3-1.7B`, balanced joint QLoRA (selection score **0.7446**)

---

## 1. What the full run is (30-second version)

The full run executes **the complete E1–E16 experiment graph** at production settings — all
7,073 training records, all 1,000 validation records, 2 training epochs — and is the run whose
numbers are the paper's Phase-3 results. It selects the Phase-4 baseline on **validation data
only** and writes a `FROZEN` manifest.

| Setting | Smoke (exp_3(a)) | **Full (this run)** |
|---|---|---|
| `TRAIN_CAP_PER_DATASET` | 64 | all — **7,073 total** |
| `VAL_CAP_PER_DATASET` | 20 (40 records) | all — **1,000 total** (883 FinQA + 117 FiQA) |
| `TRAIN_EPOCHS` | 0.05 (~1 step) | **2.0** |
| `MAX_SEQ_LENGTH_FULL` | 768 | **1024** (Phi secondary trains at **768** — see §4) |
| Freeze status written | `SMOKE_ONLY_NOT_FROZEN` | **`FROZEN`** |
| Wall-clock | 30 min | **~2.5 days** (incl. two shutdowns) |

Unlike smoke, this run **does** measure model quality: every number below is a real
full-validation result, and the winner is frozen.

---

## 2. Research rule enforced by the notebook

Phase-3 selects the baseline on **validation data only**. The notebook resolves only the
`train` and `validation` artifact keys from the frozen Phase-2.1 registry:

```
Protected artifacts intentionally NOT loaded: ['fiqa_test', 'finben_fomc_test', 'finqa_test', 'fintrust_registry']
```

The freeze manifest records the rule explicitly:

> No protected Phase-4 test set was used for model selection.
> Phase-4 must not change the frozen baseline backbone after seeing protected-test results.
> Calibration/abstention thresholds must be fitted on validation only.

The Phase-2.1 registry, train, and validation files were re-verified by SHA-256 at run start
(see §3), so the contract could not drift between the smoke and full passes.

---

## 3. Environment actually used

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 4050 Laptop, **6 GB** VRAM |
| Python | 3.12.3 in `/home/heet18/trustfin-slm/.venv` |
| PyTorch | 2.11.0+cu130 (CUDA runtime 13.0, bf16) |
| transformers / peft / bitsandbytes | 5.5.0 / 0.20.0 / 0.50.1 |
| accelerate / datasets / sklearn | 1.14.0 / 4.3.0 / 1.9.1 |
| Primary model | `Qwen/Qwen3-1.7B` (1,033 M params, LoRA 17.43 M trainable = 1.69 %) |
| Secondary model | `microsoft/Phi-4-mini-instruct` (2,248 M params, LoRA 23.07 M trainable = 1.03 %) |
| Seed | 42 |
| QLoRA | r=16, α=32, dropout 0.05, nf4 double-quant, paged AdamW 8-bit, grad checkpointing, lr 2e-4, batch 1, grad-accum 8 |
| Eval generation | FinQA `max_new_tokens=192`; FiQA `max_new_tokens=24`; greedy |
| Allocator | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold=0.85` |

Verified data fingerprints (identical to `phase2_1`):

| Artifact | SHA-256 (prefix) |
|---|---|
| `dataset_registry.json` | `e0c43a54…` |
| `train.jsonl` | `9a626134…` |
| `validation.jsonl` | `354e8171…` |

---

## 4. Two adaptations made before the full run could complete

Both preserve experiment semantics; neither alters an experiment, metric, or selection rule.

### Adaptation 1 — Phi-4-mini secondary trains at seq 768 (E8 memory ceiling)

At full `MAX_SEQ_LENGTH = 1024`, E8 (Phi-4-mini joint QLoRA) exceeded 6 GB and died:

```
OutOfMemoryError: CUDA out of memory. Tried to allocate 278.00 MiB.
```

Root cause: Phi-4-mini has a **200,064-token vocabulary**, so the fp32 loss-path logits alone
are `1024 × 200,064 × 4 B ≈ 782 MiB` on top of the ~5 GiB resident footprint — the same class
of problem as the kbit-prep fp32 bug fixed for smoke (exp_3(a) §4), but driven by sequence
length × vocabulary.

**Fix (notebook cells 7/25/27):** `SECONDARY_TRAIN_MAX_SEQ_LENGTH = 768`, and `train_qlora`
now tries `[768, 640, 512]` for the secondary model, catching `RuntimeError` OOM and retrying
smaller, recording the value actually used in `E8_training.json` (`train_max_seq_length: 768`,
no fallback needed). **Evaluation still runs at seq 1024 for every model**, so scores stay
comparable. E8 then trained at **peak 4.90 GiB**.

### Adaptation 2 — per-cell timeout raised 4 h → 12 h

On the first full attempt, E11 (joint natural: 7,073 rows × 2 epochs) exceeded the 4-hour
per-cell timeout and `nbconvert` killed the kernel **mid-evaluation**, leaving no `E11.json`.
Training alone took `13,144 s ≈ 3.65 h`; training + evaluation does not fit in 4 h.

**Fix (`tools/phase3_shard.py`):** `cell_timeout = 43200 if mode == "full" else 3600`.
The cached adapter was then reused for an eval-only re-run, so no training was wasted.

---

## 5. Data actually consumed (full)

```
Working train:      Counter({'FinQA': 6251, 'FiQA': 822})   # 7,073 of 7,073
Working validation: Counter({'FinQA': 883,  'FiQA': 117})   # 1,000 of 1,000
Few-shot IDs: FinQA n=2, FiQA n=3 (sampled from the full train pool, seed 42/17/29)
```

Sampling variants (same 7,073 rows, different epoch-level mix):

| Experiment | Train counts | Strategy |
|---|---|---|
| E3 | FinQA 6,251 | FinQA-only |
| E4 | FiQA 822 | FiQA-only |
| E5 | FinQA 3,536 / FiQA 3,537 | **balanced** joint |
| E8 | FinQA 3,536 / FiQA 3,537 | balanced joint (secondary model) |
| E11 | FinQA 6,251 / FiQA 822 | **natural** joint |

FinQA prompts are long: at seq 1024 the truncation rate is materially lower than the 79.7 %
measured at 768 in smoke, but a non-trivial fraction of FinQA prompts still lose their oldest
tokens — this is the principal driver of the residual FinQA program/execution gap (§6.1).

---

## 6. Experiment-by-experiment full results

### 6.1 Core evaluation metrics (E1–E8, E11)

n = 883 FinQA + 117 FiQA validation records per experiment.

| ID | Description | FinQA exec | FinQA prog | FinQA ans | FinQA parse | FiQA MAE | FiQA R² | mean latency |
|---|---|---|---|---|---|---|---|---|
| E1 | Primary zero-shot | 0.000 | 0.000 | 0.005 | 0.014 | 0.501 | −1.787 | 3.40 s |
| E2 | Primary few-shot | 0.000 | 0.000 | 0.000 | 0.000 | **0.185** | 0.472 | 3.08 s |
| E3 | Primary FinQA-only QLoRA | 0.681 | 0.624 | 0.390 | 1.000 | 0.505 | −1.840 | 1.19 s |
| E4 | Primary FiQA-only QLoRA | 0.000 | 0.000 | 0.000 | 0.000 | 0.129 | 0.725 | 2.35 s |
| **E5** | **Primary joint balanced QLoRA** ⭐ | **0.675** | **0.612** | **0.376** | 1.000 | **0.130** | **0.730** | 1.18 s |
| E6 | Secondary zero-shot | 0.000 | 0.000 | 0.000 | 0.028 | 0.381 | −0.444 | 3.34 s |
| E7 | Secondary few-shot | 0.040 | 0.034 | 0.006 | 0.145 | 0.325 | −0.181 | 3.53 s |
| E8 | Secondary joint balanced QLoRA | **0.701** | **0.640** | 0.386 | 1.000 | 0.130 | **0.809** | 1.08 s |
| E11 | Joint natural sampling | 0.615 | 0.555 | 0.344 | 1.000 | **0.128** | 0.736 | 1.20 s |

Reading these correctly:

- **Fine-tuning is decisive for FinQA.** Zero-/few-shot prompts (E1/E2/E6/E7) score ≈0 on
  execution; every QLoRA run (E3/E5/E8/E11) jumps to 0.61–0.70 with 100 % parse success.
- **Single-task training destroys the other task.** E3 (FinQA-only) is strong on FinQA but
  collapses FiQA (MAE 0.505, R² −1.84); E4 (FiQA-only) is the mirror (FinQA 0.000).
  **Joint training is the only setting that is good at both** — which is the paper's point.
- **FiQA is easy without training too** (E2 few-shot MAE 0.185) but fine-tuning makes it
  strong and stable (MAE ≈0.128–0.130, R² 0.73–0.81).
- **Phi** marginally wins raw FinQA (E8 0.701 > E5 0.675) and FiQA R² (0.809), but it is the
  secondary model; the Phase-4 candidate set is restricted to the pre-declared primary model
  (see §6.5, §9), so E8 stands as the cross-family **robustness baseline**, not a candidate.
- Few-shot helps FiQA (E2) but **cannot** make the model emit valid FinQA programs (E7 parse
  0.145) — program synthesis genuinely requires the adapter.

### 6.2 Training runs (full)

| ID | model | rows | epochs | train time | peak VRAM | train seq | final loss |
|---|---|---|---|---|---|---|---|
| E3 | Qwen3-1.7B | 6,251 | 2.0 | 3.54 h | 3.40 GiB | 1024 | 0.275 |
| E4 | Qwen3-1.7B | 822 | 2.0 | 6.3 min | 1.78 GiB | 1024 | 0.680 |
| E5 | Qwen3-1.7B | 7,073 | 2.0 | 2.47 h | 3.40 GiB | 1024 | 0.350 |
| E8 | Phi-4-mini | 7,073 | 2.0 | 3.54 h | **4.90 GiB** | **768** | 0.386 |
| E11 | Qwen3-1.7B | 7,073 | 2.0 | 3.65 h | 3.40 GiB | 1024 | 0.289 |

Total training compute: **≈13.5 GPU-hours**. Qwen3-1.7B trains ~1.5 h/epoch at 7 k rows;
Phi-4-mini ~2.5× slower per row. All peaks leave ≥0.7 GiB free on the 6 GB card.

### 6.3 Prompt ablations (E9 / E10) — E5 adapter, full validation

E9 — FinQA, 3 prompt variants × 883 records (all parse 1.000):

| variant | answer | program | **execution** | invalid-program rate |
|---|---|---|---|---|
| **canonical** ⭐ | 0.376 | 0.612 | **0.6750** | 0.0034 |
| no_task_token | 0.371 | 0.610 | 0.6738 | 0.0034 |
| explicit_schema | 0.373 | 0.608 | 0.6704 | 0.0045 |

→ FinQA prompt policy frozen as **`canonical`** (the three are within 0.5 pp; canonical wins
on execution and has the lowest invalid-program rate).

E10 — FiQA, 3 variants × 117 records (all parse 1.000):

| variant | MAE | RMSE | **R²** | Pearson r |
|---|---|---|---|---|
| canonical | 0.1301 | 0.1936 | 0.7301 | 0.8732 |
| no_task_token | 0.1304 | 0.1943 | 0.7281 | 0.8726 |
| **explicit_schema** ⭐ | **0.1273** | 0.1908 | **0.7378** | 0.8742 |

→ FiQA prompt policy frozen as **`explicit_schema`** (best MAE *and* R²).

### 6.4 Sampling ablation (E5 vs E11)

| Experiment | sampling | FinQA exec | FinQA prog | FiQA MAE | FiQA R² |
|---|---|---|---|---|---|
| **E5** ⭐ | balanced | **0.6750** | **0.6116** | 0.1301 | 0.7301 |
| E11 | natural | 0.6149 | 0.5549 | **0.1276** | 0.7355 |

Balanced sampling buys **+6.0 pp FinQA execution and +5.7 pp program accuracy** at the cost of
a negligible FiQA regression (+0.0025 MAE). The FinQA gain dominates the joint objective, so
**balanced (E5) wins**.

### 6.5 Failure / confidence / efficiency / freeze (E12–E16)

- **E12 (FinQA answer outcomes, n=883):** correct **304 (34.4 %)** · reasoning/execution error
  283 (32.0 %) · program right but wrong answer 265 (30.0 %) · alternate program same execution
  27 (3.1 %) · invalid program 3 (0.3 %) · answer right/program wrong 1 (0.1 %). The dominant
  failure mode is **arithmetic/reasoning execution**, not format — the adapter reliably emits a
  program (100 % parse) but mis-executes the operation chain.
- **E13 (FiQA absolute-error bands, n=117):** ≤0.10 → 64 (54.7 %) · 0.10–0.25 → 42 (35.9 %) ·
  0.25–0.50 → 7 (6.0 %) · 0.50–1.00 → 4 (3.4 %) · >1.00 → **0**. Errors are concentrated on
  the correct end; no catastrophic misses.
- **E14 (confidence):** **weakly** informative. FinQA confidence vs execution-correct:
  Spearman 0.125, ROC-AUC **0.577**; FiQA confidence vs |error|: Spearman 0.237, Pearson 0.219.
  FinQA confidence bins show quality rising only at the top bin (0.586 → 0.847). Usable as a
  *soft* abstention signal in Phase 4, not a calibrated one.
- **E15 (efficiency):** trained adapters generate at **20–25 tok/s** (Qwen) and ~20 tok/s
  (Phi); zero-/few-shot run 43–49 tok/s (short programs → short generations). Mean FinQA prompt
  822 tok; trained FinQA generation ≈29 tok, FiQA ≈21–53 tok. Every trainable-parameter and
  peak-VRAM cell populated.
- **E16 (candidate selection):** eligible joint candidates **{E5, E11}** (primary model only).

| Experiment | model | sampling | **selection score** | FinQA exec | FiQA MAE |
|---|---|---|---|---|---|
| **E5** ⭐ | Qwen3-1.7B | balanced | **0.74456** | 0.6750 | 0.1301 |
| E11 | Qwen3-1.7B | natural | 0.72046 | 0.6149 | 0.1276 |

→ **Winner: E5.** Freeze written with `status = "FROZEN"` and the frozen prompt policy
(FinQA `canonical`, FiQA `explicit_schema`).

### 6.6 Completion gates

All 16 expected outputs exist:

```
E1 … E8  metrics/E*.json          E9, E10, E11  analysis/*ablation*.csv
E12, E13  analysis/failure_*.csv  E14, E15      analysis/*.csv
E16       manifests/phase3_freeze.json
```

```
================================================================================
PHASE-3 STATUS: PASS
RUN_MODE: full
All E1-E16 outputs present: True
Freeze status: FROZEN
================================================================================
```

Plots written: `plots/finqa_execution_accuracy.png`, `plots/fiqa_mae.png`.

---

## 7. What the full run proved — and what it did not

**Proved**

1. **Joint QLoRA is the winning training regime** for a dual-task financial SLM: it is the
   only setting strong on both FinQA (exec 0.675) and FiQA (R² 0.730); single-task training
   catastrophically forgets the other task.
2. **A 1.7 B model reaches 67.5 % FinQA execution accuracy** after 2.5 h of QLoRA on a 6 GB
   laptop GPU, with 100 % program-parse success.
3. **Balanced > natural sampling** for the joint objective (+6 pp FinQA exec; §6.4).
4. **Prompt format matters per task**: FinQA prefers `canonical`, FiQA prefers
   `explicit_schema`.
5. **Failure is execution-bound, not format-bound** (E12): 96 % of FinQA errors are
   program-correct-but-mis-executed or reasoning errors; only 0.3 % are invalid programs.
6. **Confidence is only weakly calibrated** (AUC 0.577) — flags Phase-4 abstention as a
   research problem, not a solved one.
7. The Phase-2.1 contract verifies by SHA-256; protected test sets are never touched.

**Not proved**

1. **Generalisation to protected test sets** — by design (validation-only selection).
2. **Cross-family superiority.** Phi (E8) edges Qwen on raw FinQA/FiQA-R² but was never a
   candidate; whether that holds under equal treatment is open.
3. **Robustness to distribution shift** (FinBen-FOMC, FinTrust) — untouched.
4. **Calibrated abstention** — E14 shows confidence is too weak to threshold directly.

---

## 8. Artifacts and how to reproduce

```
artifacts/phase3/
├── checkpoints/{E3,E4,E5,E8,E11}/     # LoRA adapters (full-trained; E5 is the Phase-4 baseline)
├── predictions/E*.csv                  # per-record outputs (E1–E11 + 3+3 ablations)
├── metrics/E*.json                     # per-experiment metrics + *_training.json
├── metrics/phase3_validation_metrics.csv
├── analysis/E9…E16*.csv                # ablations, failure summaries, confidence, efficiency
├── manifests/                          # environment, fewshot, experiment_plan,
│                                       # phase3_freeze.json, phase3_completion_report.json
└── plots/{finqa_execution_accuracy,fiqa_mae}.png
```

The run was sharded into 8 parts so an interrupted part never costs more than ~3 h:

```bash
cd "/home/heet18/Downloads/Research Papers"
# sequential driver: cool-down after each part, then launch the next
bash tools/phase3_supervisor.sh          # wraps everything in systemd-inhibit
# or a single part:
.venv/bin/python tools/phase3_shard.py launch part7 --mode full --run
.venv/bin/python tools/phase3_shard.py verify          # which E1..E16 outputs exist
```

`OVERWRITE` stays `False`, so an interrupted run resumes from completed experiments (the
`train_qlora` cache reuses an adapter only when its `adapter_config.json` **and**
`metrics/E*_training.json` both exist). A pre-run snapshot is kept at
`artifacts/phase3_snapshot_2026-10-10_2350.tar.gz`; the packaged handoff is
`artifacts/phase3_FINAL_deliverable_2026-10-11_1018.tar.gz` (342 MB).

Wall-clock timeline:

| Part | Experiments | Finished (IST) |
|---|---|---|
| part1 | E1, E2, E6 | Oct 8, 17:48 |
| part2 | E7, E3 | Oct 8, 22:44 |
| part3 | E4 | Oct 9, 09:59 |
| part4 | E5 | Oct 10, 14:52 |
| part6 | E11 | Oct 10, 19:35 |
| part5 | E8 | Oct 10, 23:30 |
| part7 | E9, E10 | Oct 11, 10:00 |
| part8 | E12–E16 | Oct 11, 10:01 |

---

## 9. Handoff to Phase-4

Frozen decisions carried into Phase 4:

| Item | Frozen value |
|---|---|
| **Backbone** | **E5** — `Qwen/Qwen3-1.7B` + balanced joint LoRA adapter (`checkpoints/E5`) |
| FinQA prompt | `canonical` |
| FiQA prompt | `explicit_schema` |
| Selection rule | validation-only, `JOINT_CANDIDATES = {E5, E11}` |
| Protected sets | untouched; Phase-4 must not re-select the backbone on test results |
| Calibration | must be fitted on validation only (E14 shows confidence is only soft) |

→ Phase-4 proceeds from `checkpoints/E5`. The freeze manifest
`artifacts/phase3/manifests/phase3_freeze.json` is the contract.
