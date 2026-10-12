#!/usr/bin/env python3
"""phase3_shard.py — run the TrustFin Phase-3 notebook across multiple PCs.

The notebook (Phase3_TrustFin_SLM_Baselines_and_Failure_Analysis.ipynb) resumes
everything from artifacts/phase3/: QLoRA adapters, eval metrics, and prediction
CSVs are cached on disk and reloaded when present (OVERWRITE=False). That makes
the experiment set safely splittable across machines.

Waves
-----
Wave 1 (parallel, one shard per PC):
  shard-a : E1 E2 E3 E4 E6 E7   (zero/few-shot + FinQA-only & FiQA-only QLoRA)
  shard-b : E5 E9 E10           (joint balanced QLoRA + prompt ablations; E9/E10 need E5's adapter)
  shard-c : E8 E11              (secondary joint QLoRA + natural-sampling joint QLoRA)
Wave 2 (one PC, after merging all artifacts):
  consolidate : E12 E13 E14 E15 E16 + completion gates (analysis-only, fast)

Usage
-----
  python3 tools/phase3_shard.py plan                     # allocation, prerequisites, time estimates
  python3 tools/phase3_shard.py env shard-b              # print export + launch command for a shard
  python3 tools/phase3_shard.py launch shard-b           # print and run it (add --run to execute)
  python3 tools/phase3_shard.py merge /path/to/phase3    # fold a returned artifacts/phase3 into this workspace
  python3 tools/phase3_shard.py verify                   # which of E1..E16 outputs exist, who still owes what
  python3 tools/phase3_shard.py estimate                 # time estimates (measured smoke + modeled full)

PC checklist (each machine): Python 3.12 venv with the trustfin requirements
(torch+cu, transformers, peft, bitsandbytes, datasets, trl, pandas, scipy,
scikit-learn, matplotlib, jupyter), this workspace checked out, both base models
cached (Qwen/Qwen3-1.7B ~3.5 GB, microsoft/Phi-4-mini-instruct ~7.2 GB), and
artifacts/phase2_1 present (the frozen contract).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

WORKSPACE = Path(
    os.environ.get("TRUSTFIN_WORKSPACE", "/home/heet18/Downloads/Research Papers")
).expanduser()
NOTEBOOK = WORKSPACE / "Phase3_TrustFin_SLM_Baselines_and_Failure_Analysis.ipynb"
VENV_PY = Path(
    os.environ.get(
        "TRUSTFIN_PYTHON",
        "/home/heet18/trustfin-slm/.venv/bin/python",
    )
)
PHASE3_DIR = WORKSPACE / "artifacts" / "phase3"
CONTRACT = (
    WORKSPACE / "artifacts" / "phase2_1" / "manifests" / "phase3_data_config.json"
)

ALL_EXPERIMENTS = [f"E{i}" for i in range(1, 17)]

SHARDS = {
    "shard-a": ["E1", "E2", "E3", "E4"],
    "shard-b": ["E5", "E9", "E10", "E11"],
    "shard-c": ["E6", "E7", "E8"],
    "consolidate": ["E12", "E13", "E14", "E15", "E16"],
}

# Mirrors EXPECTED_EXPERIMENT_OUTPUTS in the notebook's completion-gate cell.
EXPECTED_OUTPUTS = {
    **{f"E{i}": PHASE3_DIR / "metrics" / f"E{i}.json" for i in range(1, 9)},
    "E9": PHASE3_DIR / "analysis" / "E9_finqa_prompt_ablation.csv",
    "E10": PHASE3_DIR / "analysis" / "E10_fiqa_prompt_ablation.csv",
    "E11": PHASE3_DIR / "analysis" / "E11_sampling_ablation.csv",
    "E12": PHASE3_DIR / "analysis" / "E12_finqa_failure_summary.csv",
    "E13": PHASE3_DIR / "analysis" / "E13_fiqa_failure_summary.csv",
    "E14": PHASE3_DIR / "analysis" / "E14_confidence_error_correlation.csv",
    "E15": PHASE3_DIR / "analysis" / "E15_efficiency_profile.csv",
    "E16": PHASE3_DIR / "manifests" / "phase3_freeze.json",
}

# Dependencies that must exist BEFORE an experiment runs.
DEPENDS = {
    "E9": ["E5"],  # uses E5's adapter on the same PC
    "E10": ["E5"],
    "E11": ["E5"],  # E11 cell writes the E5-vs-E11 comparison table
    "E12": ["E5", "E11"],  # candidate selection reads both joint runs
    "E13": ["E5"],
    "E14": ["E1", "E5"],
    "E15": ["E3", "E4", "E5", "E8", "E11"],  # reads training metas
    "E16": ["E5", "E11", "E9", "E10"],  # final comparison + freeze
}

# Measured / modeled estimates (RTX 4050 6 GB laptop):
#  - smoke artifact window measured Oct-02: 73 min (plus ~7 min startup/load)
#  - full mode: seq 1024, full validation (1000 records) per eval, 2-epoch trainings
ESTIMATES = {
    "smoke_single_pc_min": (80, 95),
    "full_single_pc_h": (12, 14),
    "full_shard_a_h": (3.5, 4.5),
    "full_shard_b_h": (3.0, 4.0),
    "full_shard_c_h": (2.5, 3.5),
    "full_consolidate_h": (0.5, 1.0),
    "full_3pc_wall_h": (4.5, 5.5),
}


# Single-PC schedule: ~3h sessions, safe against power cuts (everything resumes
# from artifacts/phase3 on disk). partN or a raw list "E1,E2,E6" both work.
PARTS = {
    "part1": ["E1", "E2", "E6"],  # prompting baselines (~3h)
    "part2": ["E7", "E3"],  # Phi few-shot + Qwen FinQA-only (~3h)
    "part3": ["E4"],  # Qwen FiQA-only (~2.5-3h)
    "part4": ["E5"],  # joint balanced (~2.5-3h)
    "part5": ["E8"],  # secondary joint (~2.5-3h)
    "part6": ["E11"],  # joint natural (~2.5-3h)
    "part7": ["E9", "E10"],  # prompt ablations (~3h)
    "part8": ["E12", "E13", "E14", "E15", "E16"],  # analysis + freeze gates (~1h)
}


def shard_experiments(name: str) -> list[str]:
    if "," in name:
        exps = [x.strip().upper() for x in name.split(",") if x.strip()]
        unknown = [e for e in exps if e not in ALL_EXPERIMENTS]
        if unknown:
            sys.exit(f"unknown experiment IDs: {unknown}")
        return exps
    if name in PARTS:
        return PARTS[name]
    if name in SHARDS:
        return SHARDS[name]
    if name in ALL_EXPERIMENTS:
        return [name]
    sys.exit(
        f"unknown shard/part {name!r}; choose from {sorted(PARTS)} {sorted(SHARDS)} or an ID list"
    )


def build_command(experiments: list[str], mode: str) -> tuple[dict, list[str]]:
    env = dict(os.environ)
    env["TRUSTFIN_RUN_MODE"] = mode
    env["TRUSTFIN_EXPERIMENTS"] = ",".join(experiments)
    env.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,garbage_collection_threshold:0.85",
    )
    # Training+eval share one cell; joint-sampled trainings on the secondary
    # model (Phi-4-mini, seq 768) can run 5-6h before eval, so allow 12h/cell
    # in full mode (E11 already blew through 4h and was killed mid-eval).
    cell_timeout = 43200 if mode == "full" else 3600
    cmd = [
        str(VENV_PY.parent / "jupyter"),
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        f"--ExecutePreprocessor.timeout={cell_timeout}",
        "--ExecutePreprocessor.kernel_name=trustfin",
        "--output",
        f"Phase3_{mode}_{'_'.join(experiments[:1])}.ipynb",
        "--output-dir",
        str(WORKSPACE),
        str(NOTEBOOK),
    ]
    return env, cmd


def preflight() -> list[str]:
    problems = []
    if not NOTEBOOK.exists():
        problems.append(f"notebook missing: {NOTEBOOK}")
    if not CONTRACT.exists():
        problems.append(f"phase2_1 contract missing: {CONTRACT}")
    if not (VENV_PY.exists() or shutil.which(str(VENV_PY))):
        problems.append(f"venv python missing: {VENV_PY}")
    cache = Path.home() / ".cache/huggingface/hub"
    for model in ("models--Qwen--Qwen3-1.7B", "models--microsoft--Phi-4-mini-instruct"):
        if not (cache / model).exists():
            problems.append(
                f"model not cached: {model} (huggingface-cli download <repo>)"
            )
    return problems


def cmd_plan(_args) -> None:
    print("Phase-3 multi-PC plan")
    print("=" * 72)
    print(f"workspace : {WORKSPACE}")
    print(f"notebook  : {NOTEBOOK.name}")
    print(f"contract  : {CONTRACT} {'[OK]' if CONTRACT.exists() else '[MISSING]'}")
    print(
        f"artifacts : {PHASE3_DIR} {'[exists]' if PHASE3_DIR.exists() else '[fresh run]'}"
    )
    problems = preflight()
    print("preflight :", "OK" if not problems else "PROBLEMS:")
    for p in problems:
        print("   -", p)
    print()
    print("Wave 1 — run in parallel, one shard per PC (RUN_MODE=full):")
    for name, exps in SHARDS.items():
        if name == "consolidate":
            continue
        deps = sorted({d for e in exps for d in DEPENDS.get(e, []) if d not in exps})
        print(f"  {name:<10} -> {','.join(exps)}")
        if deps:
            print(f"  {'':<10}    WARNING unmet in-shard deps: {deps}")
    print("Wave 2 — after merging wave-1 artifacts into ONE PC:")
    print(
        f"  {'consolidate':<10} -> {','.join(SHARDS['consolidate'])} (+ completion gates)"
    )
    print()
    print("Per-PC prerequisites:")
    print("  1. this workspace (notebook + artifacts/phase2_1 contract)")
    print("  2. venv: torch+cuda, transformers, peft, bitsandbytes, datasets, trl,")
    print("     pandas, scipy, scikit-learn, matplotlib, jupyter")
    print(
        "  3. models cached: Qwen/Qwen3-1.7B (~3.5 GB), Phi-4-mini-instruct (~7.2 GB)"
    )
    print(
        "  4. NVIDIA GPU >= 6 GB VRAM (seq 1024 measured peaks: Qwen 4.53 / Phi 4.88 GiB)"
    )
    print("  5. ~5 GB free disk per PC (adapters + predictions are only ~0.5 GB total)")
    print()
    print("Transfer between PCs: copy artifacts/phase3 (approx 0.3-0.5 GB, mostly")
    print("LoRA adapters) via scp/USB/cloud, then run `merge`. Nothing else moves.")
    print()
    print("Run `estimate` for timing, `verify` to see which shard still owes outputs.")


def cmd_env(args) -> None:
    exps = shard_experiments(args.shard)
    env, cmd = build_command(exps, args.mode)
    print(f"# shard {args.shard} ({args.mode} mode)")
    print(f"export TRUSTFIN_RUN_MODE={env['TRUSTFIN_RUN_MODE']}")
    print(f"export TRUSTFIN_EXPERIMENTS={env['TRUSTFIN_EXPERIMENTS']}")
    print(f"export PYTORCH_CUDA_ALLOC_CONF={env['PYTORCH_CUDA_ALLOC_CONF']}")
    print("# then, from the workspace root:")
    print(" ".join(cmd))


def cmd_launch(args) -> None:
    exps = shard_experiments(args.shard)
    env, cmd = build_command(exps, args.mode)
    problems = preflight()
    if problems:
        print("preflight failed:")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    print(f"shard {args.shard}: RUN_MODE={args.mode} EXPERIMENTS={','.join(exps)}")
    if not args.run:
        print("(dry run — pass --run to execute)")
        print(" ".join(cmd))
        return
    if getattr(args, "inhibit", False):
        cmd = [
            "systemd-inhibit",
            "--what=sleep:idle:handle-lid-switch",
            "--mode=block",
            f"--why=TrustFin Phase3 {args.shard} {args.mode}",
        ] + cmd
    log = WORKSPACE / f"phase3_{args.shard}_{args.mode}.log"
    import subprocess

    with log.open("w") as lf:
        proc = subprocess.Popen(
            cmd,
            cwd=WORKSPACE,
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(f"started pid={proc.pid} log={log}")


def cmd_merge(args) -> None:
    src = Path(args.src).expanduser()
    if src.name != "phase3":
        candidate = src / "phase3"
        if candidate.is_dir():
            src = candidate
    if not src.is_dir():
        sys.exit(f"not a directory: {src}")
    PHASE3_DIR.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        dst = PHASE3_DIR / rel
        if dst.exists():
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
        copied += 1
    print(f"merge from {src}: copied {copied}, already present {skipped}")
    print("run `verify` to see remaining gaps.")


def cmd_verify(_args) -> None:
    rows = []
    for exp in ALL_EXPERIMENTS:
        out = EXPECTED_OUTPUTS[exp]
        exists = out.exists()
        missing_deps = [
            d for d in DEPENDS.get(exp, []) if not EXPECTED_OUTPUTS[d].exists()
        ]
        rows.append((exp, exists, missing_deps))
    width = max(len(e) for e in ALL_EXPERIMENTS)
    owe = {}
    print(f"{'exp':<{width}}  status      notes")
    for exp, exists, missing_deps in rows:
        if exists:
            status = "DONE"
            note = ""
        else:
            status = "MISSING"
            note = f"needs {','.join(missing_deps)} first" if missing_deps else ""
            for shard, exps in SHARDS.items():
                if exp in exps:
                    owe.setdefault(shard, []).append(exp)
                    break
        print(f"{exp:<{width}}  {status:<10}  {note}")
    done = sum(1 for _, e, _ in rows if e)
    print(f"\n{done}/{len(ALL_EXPERIMENTS)} outputs present in {PHASE3_DIR}")
    if owe:
        print("outstanding by shard:")
        for shard, exps in owe.items():
            print(f"  {shard}: {','.join(exps)}")
    else:
        print("all E1-E16 outputs present. If RUN_MODE=full, the notebook's")
        print("completion gates should read PHASE-3 STATUS: PASS / READY TO FREEZE.")
    freeze = PHASE3_DIR / "manifests" / "phase3_freeze.json"
    if freeze.exists():
        try:
            status = json.loads(freeze.read_text()).get("status")
            print("freeze status:", status)
        except json.JSONDecodeError:
            print("freeze status: <unreadable>")


def cmd_parts(_args) -> None:
    print("Single-PC schedule — one ~3h session at a time (everything resumes):")
    print("=" * 68)
    est = {
        "part1": "~3h   3 baseline evals x 1000 records",
        "part2": "~3h   Phi few-shot eval + Qwen FinQA-only train+eval",
        "part3": "~2.5-3h Qwen FiQA-only train+eval",
        "part4": "~2.5-3h joint balanced train+eval",
        "part5": "~2.5-3h secondary joint train+eval",
        "part6": "~2.5-3h joint natural train+eval",
        "part7": "~3h   prompt ablations (needs part4's E5 adapter)",
        "part8": "~1h   analysis E12-E16 + completion gates + freeze",
    }
    for name, exps in PARTS.items():
        done = [e for e in exps if EXPECTED_OUTPUTS[e].exists()]
        mark = "DONE " if len(done) == len(exps) else ("partly" if done else "todo ")
        print(f"{mark} {name}: {','.join(exps):<28} {est[name]}")
    print()
    print("Run one per session:")
    print("  python3 tools/phase3_shard.py launch part1 --mode full --inhibit --run")
    print("Interrupted? Re-run the SAME command — finished experiments are skipped,")
    print("the interrupted one resumes from its cached pieces.")


def cmd_estimate(_args) -> None:
    print("Time estimates (RTX 4050 6 GB laptop; ±20%)")
    print("=" * 56)
    lo, hi = ESTIMATES["smoke_single_pc_min"]
    print(f"smoke, 1 PC (pipeline check, capped data)   : ~{lo}-{hi} min  [measured]")
    lo, hi = ESTIMATES["full_single_pc_h"]
    print(f"full, 1 PC (everything sequential)          : ~{lo}-{hi} h")
    print("full, 3 PCs in parallel (wave 1) + merge:")
    for key, label in (
        ("full_shard_a_h", "shard-a  E1 E2 E3 E4"),
        ("full_shard_b_h", "shard-b  E5 E9 E10 E11"),
        ("full_shard_c_h", "shard-c  E6 E7 E8"),
    ):
        lo, hi = ESTIMATES[key]
        print(f"    {label:<28}: ~{lo}-{hi} h")
    lo, hi = ESTIMATES["full_consolidate_h"]
    print(f"    {'merge + consolidate E12-E16':<28}: ~{lo}-{hi} h")
    lo, hi = ESTIMATES["full_3pc_wall_h"]
    print(f"-> wall clock on 3 PCs                  : ~{lo}-{hi} h")
    print("   (vs ~12-14 h single PC => roughly 2.5-3x faster)")
    print()
    print("Why not more speedup: each experiment is one GPU process; the notebook")
    print("is sequential per PC but experiments are independent across PCs. The")
    print("ceiling is the slowest shard (shard-a/b are inference-bound; a stronger")
    print("GPU shortens them; more PCs past 3 help little because only 3 wave-1")
    print("shards exist and E9/E10 must stay with E5).")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("plan", help="allocation, prerequisites, estimates")

    p = sub.add_parser("env", help="print env + command for a shard/part")
    p.add_argument("shard", help="part1..part8 | shard-a/b/c | consolidate | E1,E2,...")
    p.add_argument("--mode", default="full", choices=["smoke", "full"])

    p = sub.add_parser("launch", help="launch a shard/part")
    p.add_argument("shard", help="part1..part8 | shard-a/b/c | consolidate | E1,E2,...")
    p.add_argument("--mode", default="full", choices=["smoke", "full"])
    p.add_argument(
        "--run", action="store_true", help="actually execute (default: dry run)"
    )
    p.add_argument(
        "--inhibit",
        action="store_true",
        help="wrap with systemd-inhibit (blocks sleep during the run)",
    )

    p = sub.add_parser(
        "merge", help="fold another PC's artifacts/phase3 into this workspace"
    )
    p.add_argument("src", help="path to the received phase3 directory (or its parent)")

    sub.add_parser("parts", help="show the single-PC ~3h session schedule")
    sub.add_parser("verify", help="show E1-E16 output status and outstanding shards")
    sub.add_parser("estimate", help="time estimates for 1 PC vs 3 PCs")

    args = ap.parse_args()
    {
        "plan": cmd_plan,
        "env": cmd_env,
        "launch": cmd_launch,
        "merge": cmd_merge,
        "parts": cmd_parts,
        "verify": cmd_verify,
        "estimate": cmd_estimate,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
