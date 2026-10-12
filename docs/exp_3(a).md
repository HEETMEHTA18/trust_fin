# Deep Explanation (a) — Phase-3 **SMOKE** run

**Notebook:** `Phase3_TrustFin_SLM_Baselines_and_Failure_Analysis.ipynb`
**Run mode:** `RUN_MODE = "smoke"`
**Executed:** 2 October 2026, 19:45–20:15 IST (final successful pass)
**Executed notebook:** `notebooks/Phase3_smoke_executed.ipynb` (36/36 code cells, 0 errors)
**Result:** `PHASE-3 STATUS: NOT YET FROZEN` · `All E1-E16 outputs present: True` · `Freeze status: SMOKE_ONLY_NOT_FROZEN`

---

## 1. What the smoke run is (30-second version)

Smoke mode executes **the complete E1–E16 experiment graph** with tiny caps so the whole
pipeline — data loading, QLoRA training, deterministic generation, the official FinQA
executor, prompt/sampling ablations, failure analysis, confidence correlation, efficiency
profiling, candidate selection, and the freeze/gate machinery — is proven end-to-end
before the expensive full run.

| Setting | Smoke value | Full value |
|---|---|---|
| `TRAIN_CAP_PER_DATASET` | 64 | all (7,073 total) |
| `VAL_CAP_PER_DATASET` | 20 (40 records total) | all (1,000 total) |
| `TRAIN_EPOCHS` | 0.05 (~1 optimizer step) | 2.0 |
| `MAX_SEQ_LENGTH` | 768 (this run — see §4) | see exp_3(b) |
| Freeze status written | `SMOKE_ONLY_NOT_FROZEN` | `FROZEN` |

**Smoke does not measure model quality.** With one training step and 40-record eval
subsets, the numbers below only prove that every metric *computes* and every artifact
*lands on disk with the right schema*. Quality conclusions come from the full run
(`exp_3(b).md`).

---

## 2. Research rule enforced by the notebook

Phase-3 selects the baseline on **validation data only**. The notebook resolves only the
`train` and `validation` artifact keys from the frozen Phase-2.1 registry and explicitly
prints that protected keys are never loaded:

```
Protected artifacts intentionally NOT loaded: ['fiqa_test', 'finben_fomc_test', 'finqa_test', 'fintrust_registry']
```

This held in smoke mode — the completion report records
`"protected_test_policy": "No Phase-4 protected artifact was loaded by this notebook."`

---

## 3. Environment actually used

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 4050 Laptop, **6 GB** VRAM (notebook recommends a 40 GB A100) |
| Python | 3.12.3 in `/home/heet18/trustfin-slm/.venv` |
| PyTorch | 2.11.0+cu130 (CUDA runtime 13.0, bf16 supported) |
| transformers / peft / bitsandbytes | 5.5.0 / 0.20.0 / 0.50.1 |
| Primary model | `Qwen/Qwen3-1.7B` (cached, 3.9 GB) |
| Secondary model | `microsoft/Phi-4-mini-instruct` (downloaded this session, 7.2 GB) |
| Seed | 42 (`seed_everything`) |
| Allocator | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.85` |
| Phase-2.1 artifacts | resolved via `trustfin_slm_workspace/` symlink shim → `artifacts/`, `data/` |

The workspace restructure moved `trustfin_slm_workspace/{artifacts,data}` to
`Research Papers/{artifacts,data}`; the registry stores absolute paths, so a two-symlink
shim restores every Phase-2.1 path without editing the frozen contract files.

---

## 4. Four real bugs fixed before the run could pass

The notebook as delivered could not complete even smoke mode on this machine. Each fix
is minimal, commented in-place, and preserves the experiment semantics.

### Bug 1 — pyarrow cannot ingest mixed-type SFT records (E3 crash)

`tokenize_sft_records` built `Dataset.from_list(raw_records)`. In the Phase-2.1 export,
`target_answer` is `float` for 6,127 FinQA rows but `str` for 124 rows (47 of them the
literal `"no"`). pyarrow infers `double` from the leading floats and dies:

```
ArrowInvalid: Could not convert 'no' with type str: tried to convert to double
```

**Fix (cell 25):** encode each record to integer token arrays first, then build the
Dataset from `input_ids / attention_mask / labels` only — the columns arrow can type
safely.

### Bug 2 — Phi-4-mini remote code vs transformers 5.x (E6/E7 crash)

With `trust_remote_code=True`, transformers loads Phi-4-mini's bundled
`modeling_phi3.py`, which imports `SlidingWindowCache` — removed in transformers 5.x:

```
ImportError: cannot import name 'SlidingWindowCache' from 'transformers.cache_utils'
```

**Fix (cell 21):** `trust_remote_code=False` for tokenizer and base model. Phi-4-mini's
`architectures: ["Phi3ForCausalLM"]` maps to the native, maintained Phi3 implementation.
Verified standalone: config, tokenizer, 4-bit load, and generation all work.

### Bug 3 — kbit-prep silently pins +1.14 GiB in frozen fp32 (E8 OOM)

`prepare_model_for_kbit_training` upcasts the frozen `embed_tokens` (and, tied,
`lm_head`) to fp32. On Phi-4-mini's 100k vocabulary that is +1.15 GiB of parameters that
**never receive gradients** in QLoRA. Combined with 6 GB VRAM, E8 OOMed:

```
OutOfMemoryError: CUDA out of memory. Tried to allocate 278.00 MiB.
Of the allocated memory 5.40 GiB is allocated by PyTorch ...
```

Measured directly:

| Stage | Before fix | After fix |
|---|---|---|
| 4-bit model loaded | 2.69 GiB | 2.69 GiB |
| after `prepare_model_for_kbit_training` | 3.84 GiB | 3.84 GiB |
| after dtype restoration | (same) | **2.69 GiB** |
| after LoRA wrap | 3.92 GiB | **2.78 GiB** |

**Fix (cell 21):** after kbit-prep, any parameter that is fp32, frozen, and >10 M
elements is cast back to the shipped bf16. Verified: `embed_tokens.weight` restored
(`lm_head` is tied to it in Phi-3/4). E8 then trained at **peak 4.88 GiB** — fits.

### Bug 4 — memory envelope on a 6 GB GPU

Two adaptations, both commented in the configuration cell:

- `MAX_SEQ_LENGTH_SMOKE`: 1536 → **768**. At 1536, Qwen QLoRA training OOMed
  (4.64 GiB allocated + 892 MiB transient). Smoke is a pipeline check; 768 suffices.
- allocator env adds `expandable_segments` (fragmentation fell from 1.22 GiB to
  ~64 MiB reserved-but-unallocated) and a 0.85 garbage-collection threshold.

Notebook-level behavioral change: **none** — no experiment, metric, or selection rule
was altered.

---

## 5. Data actually consumed (smoke caps)

```
Working train:   Counter({'FinQA': 64, 'FiQA': 64})   # 128 of 7,073
Working validation: Counter({'FinQA': 20, 'FiQA': 20}) # 40 of 1,000
Few-shot IDs: FinQA n=2, FiQA n=3 (from full train pool, seed 42/17/29)
```

Tokenization audit over a 300-record sample (true rendered lengths, independent of the
smoke cap):

| dataset | n | mean | p50 | p90 | max | truncation @768 |
|---|---|---|---|---|---|---|
| FiQA | 28 | 131 | 128 | 142 | 151 | ~0 % |
| FinQA | 272 | 1,130 | 1,084 | 1,546 | 3,330 | **79.7 %** |

This is the single biggest reason FinQA scores are ≈0 in smoke: four out of five FinQA
prompts lose their oldest tokens at seq 768, and only one optimizer step was run.

---

## 6. Experiment-by-experiment smoke results

### 6.1 Core evaluation metrics (E1–E8, E11)

n = 40 validation records per experiment (20 FinQA + 20 FiQA).

| ID | Description | FinQA exec / prog / ans | FiQA MAE | FiQA R² | mean latency |
|---|---|---|---|---|---|
| E1 | Primary zero-shot | 0.000 / 0.000 / 0.000 | 0.512 | −1.677 | 2.03 s |
| E2 | Primary few-shot | 0.000 / 0.000 / 0.000 | **0.160** | **0.653** | 2.10 s |
| E3 | Primary FinQA-only QLoRA | 0.000 / 0.000 / 0.000 | 0.512 | −1.677 | 3.77 s |
| E4 | Primary FiQA-only QLoRA | 0.000 / 0.000 / 0.000 | 0.512 | −1.677 | 3.64 s |
| E5 | Primary joint balanced QLoRA | 0.000 / 0.000 / 0.000 | 0.512 | −1.677 | 3.36 s |
| E6 | Secondary zero-shot | 0.000 / 0.000 / 0.000 | 0.334 | −0.121 | 2.13 s |
| E7 | Secondary few-shot | 0.050 / 0.050 / 0.000 | 0.318 | 0.167 | 2.13 s |
| E8 | Secondary joint balanced QLoRA | 0.000 / 0.000 / 0.000 | 0.334 | −0.121 | 3.35 s |
| E11 | Joint natural sampling | 0.000 / 0.000 / 0.000 | 0.512 | −1.677 | 3.60 s |

Reading these correctly:

- The pipeline works: every experiment produced predictions, parsed outputs, executed
  programs, computed metrics, and wrote `metrics/E*.json` + `predictions/E*.csv`.
- **FiQA few-shot (E2) already separates from zero-shot** (MAE 0.160 vs 0.512) — a real
  signal that the prompt/eval path is functioning.
- One-step QLoRA adapters (E3/E4/E5/E11) are statistically indistinguishable from their
  zero-shot baselines — exactly what 0.05 epochs of training should produce.
- E7 (Phi few-shot) produced the run's only correct FinQA execution (1/20).

### 6.2 Training runs (smoke)

| ID | rows | epochs | train time | peak VRAM | model |
|---|---|---|---|---|---|
| E3 | 64 (FinQA) | 0.05 | 9 s | 4.53 GiB | Qwen3-1.7B |
| E4 | 64 (FiQA) | 0.05 | 3 s | 3.99 GiB | Qwen3-1.7B |
| E5 | 128 (64/64) | 0.05 | 7 s | 4.04 GiB | Qwen3-1.7B |
| E8 | 128 (64/64) | 0.05 | 20 s | **4.88 GiB** | Phi-4-mini |
| E11 | 128 (64/64) | 0.05 | 5 s | 4.02 GiB | Qwen3-1.7B |

QLoRA parameter counts: Qwen 17.4 M trainable / 1,033 M total (1.69 %);
Phi 23.1 M / 2,248 M (1.03 %). LoRA targets discovered automatically
(Qwen: `q/k/v/o/gate/up/down_proj`; Phi: `qkv_proj, gate_up_proj, v/o/up/down_proj`,
r=16, α=32, dropout 0.05, nf4 double-quant, paged AdamW 8-bit, grad checkpointing).

### 6.3 Prompt ablations (E9 / E10) — E5 adapter

E9 — FinQA, 3 prompt variants × 20 records:

| variant | parse_success | answer | program | execution |
|---|---|---|---|---|
| canonical | 0.00 | 0.00 | 0.00 | 0.00 |
| no_task_token | 0.00 | 0.00 | 0.00 | 0.00 |
| explicit_schema | **0.20** | 0.00 | 0.00 | 0.00 |

Only `explicit_schema` yielded parseable programs at all — a hint that the schema line
matters for format compliance (to be re-judged at full scale).

E10 — FiQA, 3 variants × 20 records (all parse 1.00):

| variant | MAE | R² | Pearson r |
|---|---|---|---|
| canonical | 0.512 | −1.677 | 0.629 |
| **no_task_token** | **0.401** | −0.736 | **0.720** |
| explicit_schema | 0.476 | −1.533 | 0.618 |

Smoke prompt policy chosen by E16: FinQA → `canonical`, FiQA → `no_task_token`.

### 6.4 Sampling ablation (E11 vs E5)

Balanced (E5) and natural (E11) joint training were byte-identical on smoke metrics —
expected with one shared optimizer step and a 50/50-capped pool. Full mode is where the
two sampling strategies actually diverge.

### 6.5 Failure / confidence / efficiency / freeze (E12–E16)

- **E12 (FinQA failures):** `parse_failure = 20/20 (100 %)` — the one-step adapter never
  emitted a valid `PROGRAM:` block at seq 768. Schema-gated, not executor-gated.
- **E13 (FiQA error bands):** ≤0.10: 20 % · 0.10–0.25: 15 % · 0.25–0.50: 20 % ·
  0.50–1.00: 30 % · >1.00: 15 %. Broadly spread — the pipeline classifies correctly.
- **E14 (confidence):** correlation machinery runs (FiQA Spearman −0.050 on n=20;
  FinQA AUC undefined because there were no correct samples). Meaningful values need
  the full run.
- **E15 (efficiency):** 22–42 tokens/s generation on the 4050; trainable-parameter and
  peak-VRAM columns populated for every experiment.
- **E16 (candidate selection):** joint candidates E5 (balanced) vs E11 (natural) tied at
  selection score 0.37195 → winner E5; freeze written with
  `status = "SMOKE_ONLY_NOT_FROZEN"` (the notebook refuses to freeze outside full mode —
  verified working as designed).

### 6.6 Completion gates

All 16 expected outputs exist:

```
E1 … E8  metrics/E*.json          E9, E10, E11  analysis/*ablation*.csv
E12, E13  analysis/failure_*.csv  E14, E15      analysis/*.csv
E16       manifests/phase3_freeze.json
```

```
================================================================================
PHASE-3 STATUS: NOT YET FROZEN
RUN_MODE: smoke
All E1-E16 outputs present: True
Freeze status: SMOKE_ONLY_NOT_FROZEN
================================================================================
```

Plots written: `plots/finqa_execution_accuracy.png`, `plots/fiqa_mae.png`.

---

## 7. What smoke proved — and what it did not

**Proved**

1. All 16 experiments (E1–E16) execute end-to-end and write schema-correct artifacts.
2. QLoRA training, adapter save/reload, deterministic generation, confidence capture,
   official FinQA execution, and the selection/freeze rules all function.
3. The Phase-2.1 contract loads with SHA-256 verification; protected test sets are
   never touched.
4. The whole stack fits and runs on a 6 GB GPU (peak 4.88 GiB with the §4 fixes).

**Not proved**

1. Model quality — 1 optimizer step, 40 eval records, 79.7 % FinQA truncation.
2. Balanced-vs-natural sampling differences.
3. Confidence calibration (too few correct samples for AUC).
4. Any freeze: the gate correctly refuses `FROZEN` outside full mode.

---

## 8. Artifacts and how to reproduce

```
artifacts/phase3/
├── checkpoints/{E3,E4,E5,E8,E11}/    # LoRA adapters (smoke-trained)
├── predictions/E*.csv                 # per-record outputs (E1–E11 + 3+3 ablations)
├── metrics/E*.json                    # per-experiment metrics + *_training.json
├── analysis/E9…E15*.csv               # ablations, failure summaries, confidence, efficiency
├── manifests/                         # environment, fewshot, experiment_plan,
│                                      # phase3_freeze.json, phase3_completion_report.json
└── plots/*.png
```

Smoke state backed up to `artifacts/phase3_smoke_backup.tar.gz` (86 MB) before the full
run wiped the tree.

Reproduce:

```bash
cd "/home/heet18/Downloads/Research Papers"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.85 \
/home/heet18/trustfin-slm/.venv/bin/jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=5400 --ExecutePreprocessor.kernel_name=trustfin \
  --output Phase3_smoke_executed.ipynb --output-dir ./notebooks/ \
  "./Phase3_TrustFin_SLM_Baselines_and_Failure_Analysis.ipynb"
```

(`RUN_MODE` defaults to `"smoke"` in the configuration cell; the smoke artifacts above
must be cleared or `OVERWRITE=True` set to force a clean re-run.)

---

## 9. Handoff to the full run

Configuration change made for the full pass (and only this change, plus `RUN_MODE`):

1. `RUN_MODE = "full"` — full 7,073/1,000 data, 2 epochs, freeze enabled.
2. `MAX_SEQ_LENGTH_FULL` 3072 → **1024** — memory-justified on 6 GB (see exp_3(b) §3):
   with fp32 loss-path logits at `seq × 151,936` vocab, measured peaks at seq 768
   (Qwen 4.53 GiB, Phi 4.88 GiB) scale past the 5.64 GiB ceiling somewhere above
   ~1,300 tokens; 1024 leaves both models ≥0.5 GiB margin while cutting FinQA
   truncation from 79.7 % (768) to roughly 65 %.

`OVERWRITE` stays `False` so an interrupted full run resumes from completed
experiments. `artifacts/phase3/` was wiped before launch so no smoke-trained adapter or
40-record metric can leak into the full result.

→ Full-run results: **`exp_3(b).md`**
