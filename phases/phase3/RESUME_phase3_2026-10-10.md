# Phase 3 — resume point (stopped Sat 2026-10-10 ~23:50)

## Where we stopped
User requested a clean stop to continue tomorrow. All processes (supervisor,
nbconvert, ipykernel) were terminated; GPU released. No completed work lost.

## Snapshot (full state)
`artifacts/phase3_snapshot_2026-10-10_2350.tar.gz` (341 MB, 84 entries)
Contains: metrics/*.json, checkpoints/* (all adapters, trainer_tmp excluded),
predictions/*, analysis/*, Phase3_full_*.ipynb, the notebook, tools/*.py.
Restore with: `tar xzf artifacts/phase3_snapshot_2026-10-10_2350.tar.gz`

## Completed (9 / 16)
E1, E2, E3, E4, E5, E6, E7, E8, E11  — each has metrics JSON (+training meta where applicable).
Adapters saved: E3, E4, E5, E8, E11.

## Partially complete
E9 (FinQA prompt ablation, 3 variants): `E9_canonical.json` cached → **canonical done (1/3)**.
Remaining in E9: `no_task_token`, `explicit_schema`. These will be SKIPPED-not-redone
on resume (per-variant cache in run_eval_experiment / predictions/E9_<variant>.csv).
E10 (FiQA prompt ablation): not started (3 variants).

## Pending
- E9 remaining 2 variants
- E10 all 3 variants
- E12, E13, E14, E15, E16 (analysis + validation/freeze; depend on E9/E10 + BEST_JOINT_EXP)

## Key results so far (joint models)
| Exp | model | sampling | FinQA exec_acc | FiQA r2 |
|-----|-------|----------|----------------|---------|
| E5  | Qwen3-1.7B    | balanced | 0.6750 | 0.7301 |
| E11 | Qwen3-1.7B    | natural  | 0.6149 | 0.7355 |
| E8  | Phi-4-mini    | balanced | 0.7010 | 0.8091 |  (best joint)

## Config changes made this session (already saved in files)
1. Notebook `Phase3_TrustFin_SLM_Baselines_and_Failure_Analysis.ipynb`:
   - cell 7: added `SECONDARY_TRAIN_MAX_SEQ_LENGTH = 768`
   - cell 25: `tokenize_sft_records(..., max_length=None)` param
   - cell 27: `train_qlora` trains secondary model with seq fallback 768→640→512,
     records `train_max_seq_length` in meta. Fixes Phi-4-mini OOM at seq 1024.
2. `tools/phase3_shard.py`: full-mode cell timeout raised 14400 → **43200 s**
   (E11 train+eval exceeded 4h and was killed mid-eval; E8 needs the headroom).

## How to resume tomorrow
Run part7 (E9/E10) then part8 (E12-E16), wrapped in systemd-inhibit:

    cd "/home/heet18/Downloads/Research Papers"
    setsid nohup systemd-inhibit --what=sleep:idle:handle-lid-switch --mode=block \
      --why="TrustFin Phase3 part7 part8" \
      ./tools/phase3_supervisor.sh part7 part8 > supervisor4.out 2>&1 < /dev/null &

Notes / prerequisites for resume:
- E9/E10 need E5 adapter → `artifacts/phase3/checkpoints/E5` (present).
- part8 needs `E5/E8/E11.json` + training metas for BEST_JOINT_EXP (all present).
- GPU must be free; supervisor waits for idle + cools to <78C before each part.
- Expected remaining wall time: part7 ~3h, part8 ~1h → ~4h total.

## ETA
With the stop at 23:50, resume + finish (part7+part8) ≈ 4 h after you restart,
then run `tools/phase3_shard.py verify` (expect 16/16 + freeze) and tar artifacts.
