#!/usr/bin/env python3
"""phase4_shard.py — run the TrustFin Phase-4 notebook (EVG-CAS) in stages.

Phase-4 never trains: it loads the frozen E5 adapter from ../artifacts/phase3 and writes
everything under trust_fin_extra_phase/artifacts/phase4. Every stage resumes from disk
(OVERWRITE=False), so an interrupted run only redoes the record it died on.

Stage groups (run in this order)
--------------------------------
  P4EVG   : P4-E2, P4-E3, P4-E3b   sampling pass + selection/FiQA aggregation (GPU, ~1-1.5 h)
  P4CAL   : P4-E1, P4-E4..P4-E7    baselines, cross-fitted calibration, freeze (CPU, minutes)
  P4TEST  : P4-E8                  protected test evaluation (GPU, ~2-3 h)
            -- requires phase4_freeze.json status FROZEN **and** TRUSTFIN4_PROTECTED=1

Usage
-----
  python3 tools/phase4_shard.py plan                 # groups, prerequisites, estimates
  python3 tools/phase4_shard.py env P4EVG            # print export + command
  python3 tools/phase4_shard.py run P4EVG --mode full --run --inhibit
  python3 tools/phase4_shard.py run "P4-E2,P4-E3" --mode full --run   # raw stage list
  python3 tools/phase4_shard.py verify               # which P4-E* outputs exist
  python3 tools/phase4_shard.py freeze               # show phase4_freeze.json status
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]          # .../trust_fin_extra_phase
OUTER = WORKSPACE.parent                                 # .../Research Papers
NOTEBOOK = WORKSPACE / "notebooks" / "Phase4_TrustFin_Proposed_Method.ipynb"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import find_python, using_fallback_python         # noqa: E402

# Phase-5A: no hard-coded machine paths. Resolution order:
#   $TRUSTFIN_PYTHON  >  ~/trustfin-slm/.venv/bin/python  >  current interpreter
VENV_PY = find_python()
P4_DIR = WORKSPACE / "artifacts" / "phase4"
FREEZE = P4_DIR / "manifests" / "phase4_freeze.json"
PHASE3_FREEZE = OUTER / "artifacts" / "phase3" / "manifests" / "phase3_freeze.json"
CONTRACT = OUTER / "artifacts" / "phase2_1" / "manifests" / "phase3_data_config.json"

ALL_STAGES = ["P4-E1", "P4-E2", "P4-E3", "P4-E3b", "P4-E4", "P4-E5", "P4-E6",
              "P4-E7", "P4-E8", "P4-E9"]

GROUPS = {
    "P4EVG": ["P4-E2", "P4-E3", "P4-E3b"],
    "P4CAL": ["P4-E1", "P4-E4", "P4-E5", "P4-E6", "P4-E7"],
    "P4TEST": ["P4-E8"],
    "P4ALL": ALL_STAGES[:-1],          # everything except the deferred FinTrust stage
}

# Mirrors EXPECTED_OUTPUTS in the notebook's completion-gate cell.
EXPECTED = {
    "P4-E1": P4_DIR / "metrics" / "P4-E1.json",
    "P4-E2": P4_DIR / "predictions" / "P4-E2_candidates.csv",
    "P4-E3": P4_DIR / "analysis" / "P4-E3_selection_ablation.csv",
    "P4-E3b": P4_DIR / "analysis" / "P4-E3b_fiqa_sampling.csv",
    "P4-E4": P4_DIR / "analysis" / "P4-E4_calibration.csv",
    "P4-E5": P4_DIR / "analysis" / "P4-E5_calibration.csv",
    "P4-E6": P4_DIR / "analysis" / "P4-E6_calibration.csv",
    "P4-E7": P4_DIR / "manifests" / "phase4_freeze.json",
    "P4-E8": P4_DIR / "manifests" / "phase4_completion_report.json",
    "P4-E9": P4_DIR / "metrics" / "P4-E9.json",
}

DEPENDS = {
    "P4-E1": [],
    "P4-E2": [],
    "P4-E3": ["P4-E2"],
    "P4-E3b": [],
    "P4-E4": ["P4-E2", "P4-E3"],
    "P4-E5": ["P4-E2", "P4-E3"],
    "P4-E6": ["P4-E2", "P4-E3"],
    "P4-E7": ["P4-E4", "P4-E5", "P4-E6"],
    "P4-E8": ["P4-E7"],
    "P4-E9": ["P4-E7"],
}

ESTIMATES = {
    "P4EVG_smoke_min": "5-10",
    "P4EVG_full_h": "1.0-1.5",
    "P4CAL_full_min": "2-5",
    "P4TEST_full_h": "2-3",
}


def stages_of(name: str) -> list[str]:
    if "," in name:
        exps = [x.strip().upper() for x in name.split(",") if x.strip()]
        unknown = [e for e in exps if e not in ALL_STAGES]
        if unknown:
            sys.exit(f"unknown stage IDs: {unknown}")
        return exps
    if name in GROUPS:
        return GROUPS[name]
    if name in ALL_STAGES:
        return [name]
    sys.exit(f"unknown group/stage {name!r}; choose from {sorted(GROUPS)} or {ALL_STAGES}")


def build_command(stages: list[str], mode: str) -> tuple[dict, list[str]]:
    env = dict(os.environ)
    env["TRUSTFIN4_RUN_MODE"] = mode
    env["TRUSTFIN4_EXPERIMENTS"] = ",".join(stages)
    env["TRUSTFIN4_WORKSPACE"] = str(WORKSPACE)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                   "expandable_segments:True,garbage_collection_threshold:0.85")
    cell_timeout = 43200 if mode == "full" else 3600
    out_name = f"Phase4_{mode}_{'_'.join(stages[:1]).replace('-', '')}.ipynb"
    cmd = [
        str(VENV_PY.parent / "jupyter"), "nbconvert",
        "--to", "notebook", "--execute",
        f"--ExecutePreprocessor.timeout={cell_timeout}",
        "--ExecutePreprocessor.kernel_name=trustfin",
        "--output", out_name,
        "--output-dir", str(WORKSPACE / "notebooks"),
        str(NOTEBOOK),
    ]
    return env, cmd


def preflight() -> list[str]:
    problems = []
    if not NOTEBOOK.exists():
        problems.append(f"notebook missing: {NOTEBOOK}")
    if not CONTRACT.exists():
        problems.append(f"Phase-2.1 contract missing: {CONTRACT}")
    if not PHASE3_FREEZE.exists():
        problems.append(f"Phase-3 freeze missing: {PHASE3_FREEZE}")
    else:
        try:
            st = json.loads(PHASE3_FREEZE.read_text()).get("status")
            if st != "FROZEN":
                problems.append(f"Phase-3 freeze status is {st!r} (expected FROZEN)")
        except json.JSONDecodeError:
            problems.append("Phase-3 freeze unreadable")
    if not (VENV_PY.exists() or shutil.which(str(VENV_PY))):
        problems.append(f"venv python missing: {VENV_PY}")
    elif not os.environ.get("TRUSTFIN_PYTHON") and using_fallback_python():
        problems.append("no project venv found (set TRUSTFIN_PYTHON); "
                        "using the current interpreter")
    cache = Path.home() / ".cache/huggingface/hub"
    if not (cache / "models--Qwen--Qwen3-1.7B").exists():
        problems.append("model not cached: Qwen/Qwen3-1.7B")
    return problems


def cmd_plan(_a) -> None:
    print("Phase-4 plan — EVG-CAS (proposed method)")
    print("=" * 72)
    print(f"workspace : {WORKSPACE}")
    print(f"outer RO  : {OUTER}")
    print(f"notebook  : {NOTEBOOK.relative_to(WORKSPACE)}")
    print(f"phase3    : {PHASE3_FREEZE} [{'OK' if PHASE3_FREEZE.exists() else 'MISSING'}]")
    print(f"outputs   : {P4_DIR} [{'exists' if P4_DIR.exists() else 'fresh'}]")
    probs = preflight()
    print("preflight :", "OK" if not probs else "PROBLEMS")
    for p in probs:
        print("   -", p)
    print()
    for g, st in GROUPS.items():
        print(f"  {g:<8} -> {', '.join(st)}")
    print()
    print("order: P4EVG (GPU) -> P4CAL (CPU, writes the freeze) -> P4TEST (GPU, two-key)")
    print(f"estimates: P4EVG {ESTIMATES['P4EVG_full_h']} h | "
          f"P4CAL {ESTIMATES['P4CAL_full_min']} min | "
          f"P4TEST {ESTIMATES['P4TEST_full_h']} h")
    print()
    print("protected gate: phase4_freeze.json must read FROZEN *and* TRUSTFIN4_PROTECTED=1")
    print("run `verify` for output status, `freeze` for the freeze manifest.")


def cmd_env(a) -> None:
    stages = stages_of(a.group)
    env, cmd = build_command(stages, a.mode)
    print(f"# {a.group} ({a.mode})")
    print(f"export TRUSTFIN4_RUN_MODE={env['TRUSTFIN4_RUN_MODE']}")
    print(f"export TRUSTFIN4_EXPERIMENTS={env['TRUSTFIN4_EXPERIMENTS']}")
    print(f"export TRUSTFIN4_WORKSPACE={env['TRUSTFIN4_WORKSPACE']}")
    print(f"export PYTORCH_CUDA_ALLOC_CONF={env['PYTORCH_CUDA_ALLOC_CONF']}")
    print("# from trust_fin_extra_phase/:")
    print(" ".join(cmd))


def cmd_run(a) -> None:
    stages = stages_of(a.group)
    env, cmd = build_command(stages, a.mode)
    probs = preflight()
    if probs:
        print("preflight failed:")
        for p in probs:
            print(" -", p)
        sys.exit(1)

    if "P4-E8" in stages:
        if not FREEZE.exists():
            sys.exit("refusing: P4-E8 needs artifacts/phase4/manifests/phase4_freeze.json "
                     "(run P4EVG then P4CAL first).")
        status = json.loads(FREEZE.read_text()).get("status")
        if status != "FROZEN":
            sys.exit(f"refusing: freeze status is {status!r}, expected FROZEN.")
        if env.get("TRUSTFIN4_PROTECTED", "") not in {"1", "true", "True"}:
            sys.exit("refusing: P4-E8 needs TRUSTFIN4_PROTECTED=1 (two-key gate).")

    print(f"group {a.group}: RUN_MODE={a.mode} STAGES={','.join(stages)}")
    if not a.run:
        print("(dry run — pass --run to execute)")
        print(" ".join(cmd))
        return
    if a.inhibit:
        cmd = ["systemd-inhibit", "--what=sleep:idle:handle-lid-switch",
               "--mode=block", f"--why=TrustFin Phase4 {a.group} {a.mode}"] + cmd
    log = WORKSPACE / "logs" / f"phase4_{a.group}_{a.mode}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as lf:
        proc = subprocess.Popen(cmd, cwd=WORKSPACE, env=env, stdout=lf,
                                stderr=subprocess.STDOUT, start_new_session=True)
    print(f"started pid={proc.pid} log={log}")


def cmd_verify(_a) -> None:
    width = max(len(s) for s in ALL_STAGES)
    owe: dict[str, list[str]] = {}
    print(f"{'stage':<{width}}  status      notes")
    for st in ALL_STAGES:
        out = EXPECTED[st]
        exists = out.exists()
        missing = [d for d in DEPENDS.get(st, []) if not EXPECTED[d].exists()]
        if exists:
            status, note = "DONE", ""
        else:
            status = "MISSING"
            note = f"needs {','.join(missing)} first" if missing else ""
            for g, gss in GROUPS.items():
                if st in gss:
                    owe.setdefault(g, []).append(st)
                    break
        print(f"{st:<{width}}  {status:<10}  {note}")
    done = sum(1 for s in ALL_STAGES if EXPECTED[s].exists())
    print(f"\n{done}/{len(ALL_STAGES)} outputs present in {P4_DIR}")
    for g, gss in owe.items():
        print(f"  owes {g}: {','.join(gss)}")
    if FREEZE.exists():
        try:
            fz = json.loads(FREEZE.read_text())
            print("freeze status:", fz.get("status"), "| frozen at:", fz.get("frozen_at_utc"))
        except json.JSONDecodeError:
            print("freeze status: <unreadable>")
    else:
        print("freeze status: ABSENT (run P4EVG then P4CAL)")


def cmd_freeze(_a) -> None:
    if not FREEZE.exists():
        sys.exit("phase4_freeze.json not found — Phase-4 validation stages are not complete.")
    fz = json.loads(FREEZE.read_text())
    print(json.dumps({k: fz[k] for k in
                      ("phase", "status", "method_name", "frozen_at_utc", "evg",
                       "calibration", "operating_points", "important")
                      if k in fz}, indent=2, default=str))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan", help="groups, prerequisites, estimates")

    p = sub.add_parser("env", help="print env + command for a group")
    p.add_argument("group", help="P4EVG | P4CAL | P4TEST | P4ALL | P4-E* or a comma list")
    p.add_argument("--mode", default="full", choices=["smoke", "full"])

    p = sub.add_parser("run", help="launch a group")
    p.add_argument("group", help="P4EVG | P4CAL | P4TEST | P4ALL | P4-E* or a comma list")
    p.add_argument("--mode", default="full", choices=["smoke", "full"])
    p.add_argument("--run", action="store_true", help="actually execute (default: dry run)")
    p.add_argument("--inhibit", action="store_true", help="block sleep during the run")

    sub.add_parser("verify", help="P4-E* output status and outstanding groups")
    sub.add_parser("freeze", help="print phase4_freeze.json")

    a = ap.parse_args()
    {"plan": cmd_plan, "env": cmd_env, "run": cmd_run,
     "verify": cmd_verify, "freeze": cmd_freeze}[a.cmd](a)


if __name__ == "__main__":
    main()
