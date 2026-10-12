# Phase-3 — Baselines & Failure Analysis

**Notebook:** `../../Phase3_TrustFin_SLM_Baselines_and_Failure_Analysis.ipynb` (master/executed, workspace root)
**Result:** `PHASE-3 STATUS: PASS` · 16/16 experiments · `Freeze status: FROZEN`
**Winner (Phase-4 backbone):** **E5** — `Qwen/Qwen3-1.7B`, balanced joint QLoRA

Phase-3 trains and evaluates trustworthy SLMs on the frozen Phase-2.1 contract and selects the
Phase-4 baseline on **validation data only**. It runs the full E1–E16 graph in two modes:

| Mode | Doc | Outputs |
|---|---|---|
| smoke (`RUN_MODE="smoke"`) | `docs/exp_3(a).md` | `notebooks/Phase3_smoke_executed.ipynb` |
| **full** (`RUN_MODE="full"`) | **`docs/exp_3(b).md`** | this directory + `artifacts/phase3/` |

## Layout

```
phases/phase3/
├── README.md                      # this file
├── RESUME_phase3_2026-10-10.md    # mid-run checkpoint note (run now complete)
├── notebooks/                     # executed per-part notebooks (nbconvert output)
│   ├── Phase3_full_E1.ipynb       # part1: E1, E2, E6
│   ├── Phase3_full_E7.ipynb       # part2: E7, E3
│   ├── Phase3_full_E4.ipynb       # part3: E4
│   ├── Phase3_full_E5.ipynb       # part4: E5
│   ├── Phase3_full_E8.ipynb       # part5: E8
│   ├── Phase3_full_E11.ipynb      # part6: E11
│   ├── Phase3_full_E9.ipynb       # part7: E9, E10
│   ├── Phase3_full_E12.ipynb      # part8: E12–E16 (+ freeze gates)
│   └── Phase3_smoke_run3.ipynb    # smoke
└── logs/                          # run logs & supervisor output
    ├── phase3_part{1..8}_full.log # per-part nbconvert logs
    ├── phase3_full_run.log        # first (interrupted) full attempt
    ├── phase3_smoke_run3.log
    ├── supervisor.log             # supervisor driver log
    └── supervisor*.out, part1_*.out
```

Results live in **`artifacts/phase3/`** (metrics, analysis, plots, predictions, manifests,
checkpoints) — see `docs/exp_3(b).md` §8 for the full tree.

## How the full run was executed

The full run (~13.5 GPU-hours) was sharded into 8 parts so an interruption never costs more
than ~3 h. The driver is `tools/phase3_supervisor.sh`, which cools the GPU down after each
part and launches the next via `tools/phase3_shard.py`:

```bash
cd "/home/heet18/Downloads/Research Papers"
bash tools/phase3_supervisor.sh                 # sequential, all 8 parts
.venv/bin/python tools/phase3_shard.py verify   # E1..E16 output inventory
```

Parts are written to the **workspace root** as `Phase3_<mode>_E<first>.ipynb`; the historical
ones were archived into `notebooks/` here for tidiness (re-running recreates them at root).

## Reproduce (single pass, no sharding)

```bash
cd "/home/heet18/Downloads/Research Papers"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.85 \
/home/heet18/trustfin-slm/.venv/bin/jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=43200 --ExecutePreprocessor.kernel_name=trustfin \
  --output Phase3_full_executed.ipynb --output-dir ./notebooks/ \
  "./Phase3_TrustFin_SLM_Baselines_and_Failure_Analysis.ipynb"
```

`OVERWRITE=False` reuses cached adapters/metrics; set `OVERWRITE=True` (and clear
`artifacts/phase3/`) for a clean re-run. Set `TRUSTFIN_EXPERIMENTS` / `TRUSTFIN_RUN_MODE` to
subset experiments or switch to smoke.

## Freeze

`artifacts/phase3/manifests/phase3_freeze.json` is the contract handed to Phase-4:
backbone `checkpoints/E5`, FinQA prompt `canonical`, FiQA prompt `explicit_schema`,
validation-only selection. **No protected Phase-4 test set was used for model selection.**
