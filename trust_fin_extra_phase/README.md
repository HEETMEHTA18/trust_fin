# TrustFin-SLM — Phase-4 workspace (EVG-CAS)

Workspace for the **Phase-4 proposed method** of the TrustFin-SLM research pipeline.
The outer folder (`../`) holds Phases 2, 2.1 and 3 and is **read-only from here**.

## What Phase-4 is

**EVG-CAS** = *Execution-Verified Generation* (C1) + *Calibrated Abstention* (C2/C3), an
inference-time method layered on the Phase-3 frozen backbone `E5`
(`Qwen/Qwen3-1.7B` + balanced joint QLoRA). Nothing inherited is re-selected; see
`PLAN.md` §3 for the contract and §2 for the three pre-declared claims.

## Layout

```
trust_fin_extra_phase/
├── PLAN.md                              design, claims, protocol, guardrails
├── README.md                            this file
├── notebooks/Phase4_TrustFin_Proposed_Method.ipynb
├── notebooks/Phase4_2_Evaluation_Improvement.ipynb   # review-driven matched-coverage pass
├── docs/exp_4(a).md                     validation deep-dive + smoke harness (pre-protected)
├── docs/exp_4(b).md                     protected-test deep-dive + claims verdict
├── docs/exp_4(c).md                     matched-coverage / ablations / CIs / taxonomy (Phase-4.2)
├── docs/paper_revisions.md              concrete prose+numbers corrections for the draft
├── docs/eval_rigor.md                   statistical rigor annex (McNemar, bootstrap CIs, depth)
├── docs/novelty_check.md                dated novelty audit vs 2024–2026 selective-abstention work
├── docs/p4_e9_plan.md                   FinTrust 7-dim run plan/implementation notes
├── docs/exp_4(d).md                     P4-E9 FinTrust results (generation complete; fairness scored)
├── scripts/eval_rigor.py                rigor statistics (read-only on artifacts/phase4/)
├── scripts/eval_matched_coverage.py     matched-coverage + ablation analyzer → artifacts/phase4_2/
├── scripts/eval_fintrust.py             P4-E9 resumable FinTrust generator (official sample subsets)
├── scripts/score_fintrust.py            P4-E9 judge-later scorer (runs official postprocessors)
├── paper/                               self-contained corrected paper copy (tex + md mirror)
├── tools/phase4_shard.py                stage launcher / verifier / freeze checker
├── logs/                                run logs
├── artifacts/phase4_smoke/              RUN_MODE=smoke only
├── artifacts/phase4/                    RUN_MODE=full only (the deliverable, FROZEN)
│   ├── manifests/phase4_e9_prereg.json  P4-E9 pre-registration (written before generation)
│   ├── predictions/P4-E9_{dim}.json     P4-E9 generations (official schema, judge-ready)
│   └── metrics/P4-E9.json               P4-E9 per-dim counts/status/scores
└── artifacts/phase4_2/                  Phase-4.2 analysis tree (derived, read-only on phase4)
    └── manifest.json                    provenance: input SHAs + params + headline results
```

**Smoke and full never share paths on purpose:** a 20-record smoke freeze or a cached
smoke candidate file must not be able to satisfy a full run's gates.

## How to run

The notebook is generated, not hand-edited. Edit `build_p4_part{1,2,3}.py` sources, then:

```bash
python3 /tmp/opencode/build_p4_nb.py
```

### Stage groups (dependency order)

| Group | Stages | Device | Writes |
|---|---|---|---|
| `P4EVG` | E1, E2, E3, E3b | GPU | predictions + ablation CSVs |
| `P4CAL` | E1, E4, E5, E6, E7 | CPU | calibration CSVs + **`phase4_freeze.json`** |
| `P4TEST` | E8 (a–e) | GPU | protected-test predictions, metrics, **frozen abstention applied** |

### Commands

```bash
cd "/home/heet18/Downloads/Research Papers/trust_fin_extra_phase"
VENV=/home/heet18/trustfin-slm/.venv/bin

# 1) validate + sample on the FULL validation split (~1.5–2 h GPU)
TRUSTFIN4_RUN_MODE=full TRUSTFIN4_EXPERIMENTS="P4-E1,P4-E2,P4-E3,P4-E3b" \
  TRUSTFIN4_WORKSPACE="$PWD" \
  $VENV/jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=14400 --ExecutePreprocessor.kernel_name=trustfin \
    --output Phase4_full_EVG.ipynb --output-dir "./notebooks" \
    "./notebooks/Phase4_TrustFin_Proposed_Method.ipynb"

# 2) calibrate + freeze (minutes, CPU)
TRUSTFIN4_RUN_MODE=full TRUSTFIN4_EXPERIMENTS="P4-E1,P4-E4,P4-E5,P4-E6,P4-E7" \
  TRUSTFIN4_WORKSPACE="$PWD" \
  $VENV/jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=trustfin \
    --output Phase4_full_CAL.ipynb --output-dir "./notebooks" \
    "./notebooks/Phase4_TrustFin_Proposed_Method.ipynb"

# 3) verify the freeze, then open protected tests (TWO keys required)
python3 tools/phase4_shard.py verify
TRUSTFIN4_RUN_MODE=full TRUSTFIN4_EXPERIMENTS="P4-E8" \
  TRUSTFIN4_WORKSPACE="$PWD" TRUSTFIN4_PROTECTED=1 \
  $VENV/jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=14400 --ExecutePreprocessor.kernel_name=trustfin \
    --output Phase4_full_TEST.ipynb --output-dir "./notebooks" \
    "./notebooks/Phase4_TrustFin_Proposed_Method.ipynb"
```

Or with the helper (it injects env + `TRUSTFIN4_WORKSPACE` and checks dependencies):

```bash
python3 tools/phase4_shard.py env P4EVG --mode full
python3 tools/phase4_shard.py run P4EVG --mode full --run
python3 tools/phase4_shard.py verify
python3 tools/phase4_shard.py env P4TEST --mode full --protected
```

**Smoke:** same commands with `--mode smoke` / `TRUSTFIN4_RUN_MODE=smoke` (caps validation
at 20 records/dataset, `EVG_N=2`, writes to `artifacts/phase4_smoke/`).

## Guardrails enforced in code

1. Protected artifact keys are unresolvable until `P4-E8`, and only if
   `phase4_freeze.json.status == "FROZEN"` **and** `TRUSTFIN4_PROTECTED=1`.
2. No hyper-parameter is selected from any protected record; `n`, temperature, top-p,
   token budgets and coverage targets are pre-declared in `PLAN.md` §3/§6.
3. Calibration folds are fixed (`KFold(5, shuffle=True, random_state=42)`).
4. Every Phase-4 output lands under this folder; `../artifacts/**` and `../data/**` are
   opened read-only.
5. Thresholds come from pre-declared coverage targets on validation, never from test risk.

## Results index

| Doc | Covers |
|---|---|
| `docs/exp_4(a).md` | validation deep-dive, smoke harness, freeze rationale |
| `docs/exp_4(b).md` | protected test (FinQA 1147, FiQA 234, FOMC 496), final claim verdict |
| `docs/exp_4(c).md` | matched-coverage selective prediction, ablations, bootstrap CIs, error-taxonomy reconciliation |
| `docs/exp_4(d).md` | P4-E9 FinTrust (7 dims, official sample subsets): generation complete, fairness scored |
| `docs/paper_revisions.md` | reviewer-facing corrections (prose + numbers) for the draft |
| `paper/TrustFin_Conference_Paper.{tex,md}` | corrected, self-contained paper copy (draft v2) |

**Status: validation + protected stages complete** (2026-10-12); **Phase-4.2 evaluation-
improvement pass complete** (matched-coverage: agreement beats confidence-based abstention
at equal coverage — paired bootstrap CI excludes zero, P=1.0; ablation: agreement necessary
& sufficient; error taxonomy reconciled to mutually exclusive cells summing to 1,147).
Headline: C1 rejected (tie on FinQA, small negative on FiQA); C2 and C3 supported on FinQA —
the frozen abstention policy cuts FinQA test error by 31 % at 80 % coverage (validation
predicted 34 %; bootstrap CI [22.8 %, 39.3 %]). FOMC zero-shot OOD: E5 adapter +8.9 pp
macro-F1 over the base model. **P4-E9 (FinTrust 7 dims): generation complete** on the official
sample subsets (3,520 generations, `scripts/eval_fintrust.py`); fairness rule-scored
(personal consistency 0.67; industry mean |Cohen's d| 0.41); other dims stored in the official
schema and judge-pending (`scripts/score_fintrust.py`, needs an OpenAI key).

## Environment

`/home/heet18/trustfin-slm/.venv`, kernel `trustfin`, torch 2.11.0+cu130, RTX 4050 6 GB.
Always set `TRUSTFIN4_WORKSPACE` (the shard tool does this for you) — nbconvert runs the
kernel with cwd = the notebook's directory, so path discovery from `cwd` alone is wrong.
