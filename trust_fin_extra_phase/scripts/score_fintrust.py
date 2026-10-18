#!/usr/bin/env python3
"""score_fintrust.py — P4-E9 judge-later scoring.

Runs the *official* FinTrust `postprocess_response.py` scripts unchanged inside a
sandbox directory, pointed at our stored generations (`P4-E9_{dim}.json`). This keeps
the official judge prompts/metrics verbatim instead of re-implementing them.

Fairness dimensions are already rule-scored at generation time (no judge); they are not
handled here.

Requirements:
  * `openai` installed in the active interpreter:  pip install openai
  * OPENAI_API_KEY in the environment (the official scripts use gpt-4.1-mini).

Usage:
  python3 scripts/score_fintrust.py --dry-run
  OPENAI_API_KEY=... python3 scripts/score_fintrust.py --dims privacy robustness
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OUTER = HERE.parent
SAMPLES = OUTER / "data/raw/fintrust_eval_code"
P4 = HERE / "artifacts" / "phase4"
SCORING = P4 / "scoring"
METRIC = P4 / "metrics" / "P4-E9.json"

# dimension -> (official postprocess script relpath, our predictions file, scoring key)
DIM_SCORERS = {
    "privacy": ("privacy/postprocess_response.py", "P4-E9_privacy.json", "privacy"),
    "robustness": ("robustness/post_process_response.py", "P4-E9_robustness.json", "robustness"),
    "transparency": ("transparency/post_process_response.py", "P4-E9_transparency.json", "transparency"),
    "knowledge_discovery": ("knowledge-discovery/post_process_response.py",
                            "P4-E9_knowledge_discovery.json", "knowledge_discovery"),
    "safety_deep_inception": ("safety/deep-inception/post_process_response.py",
                              "P4-E9_safety_deep_inception.json", "safety_deep_inception"),
    "safety_in_context": ("safety/in-context attack/postprocess_response.py",
                          "P4-E9_safety_in_context.json", "safety_in_context"),
    "safety_language_barrier": ("safety/language_barrier/postprocess_response.py",
                                "P4-E9_safety_language_barrier.json", "safety_language_barrier"),
    "safety_plain_question": ("safety/plain_question/postprocess_response.py",
                              "P4-E9_safety_plain_question.json", "safety_plain_question"),
    "trustfulness_hallucination_certainty": (
        "trustfulness/hallucination_certainty/postprocess_response.py",
        "P4-E9_trustfulness_hallucination_certainty.json", "trustfulness_hallucination_certainty"),
    "trustfulness_number_informativeness": (
        "trustfulness/number_informativeness/postprocess_response.py",
        "P4-E9_trustfulness_number_informativeness.json", "trustfulness_number_informativeness"),
}


def extract_input_path(script_text: str) -> str | None:
    for var in ("file_path", "input_path"):
        m = re.search(rf"^\s*{var}\s*=\s*['\"]([^'\"]+)['\"]", script_text, flags=re.M)
        if m:
            return m.group(1)
    return None


def plan_one(dim: str) -> dict:
    script_rel, pred_name, _ = DIM_SCORERS[dim]
    script = SAMPLES / script_rel
    pred = P4 / "predictions" / pred_name
    info = {"dim": dim, "script": str(script), "predictions": str(pred)}
    if not script.exists():
        info["error"] = "official script missing"
        return info
    if not pred.exists():
        info["error"] = "predictions missing (generation incomplete?)"
        return info
    text = script.read_text(encoding="utf-8")
    fp = extract_input_path(text)
    info["expected_input_path"] = fp
    if not fp:
        info["error"] = "could not find file_path/input_path assignment"
        return info
    # sanity: our records vs expected
    try:
        n = len(json.loads(pred.read_text(encoding="utf-8")))
        info["n_records"] = n
    except Exception as e:
        info["error"] = f"predictions not readable: {e}"
    info["needs_judge"] = "gpt-4.1-mini" in text
    return info


def run_one(dim: str, sandbox: Path, judge_model: str | None, limit: int | None = None) -> dict:
    info = plan_one(dim)
    if info.get("error"):
        return info
    script_rel, pred_name, key = DIM_SCORERS[dim]
    script = SAMPLES / script_rel
    pred = P4 / "predictions" / pred_name
    rel_input = info["expected_input_path"]

    sb = sandbox / dim
    if sb.exists():
        shutil.rmtree(sb)
    (sb / Path(rel_input).parent).mkdir(parents=True, exist_ok=True)
    # official script expects the input at file_path (relative to cwd)
    if limit:
        rows = json.loads(pred.read_text(encoding="utf-8"))[:limit]
        (sb / rel_input).write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        info["limit"] = limit
    else:
        shutil.copy(pred, sb / rel_input)
    # place the official postprocess script next to its natural home and run with cwd=sandbox
    dest_script = sb / Path(script_rel).name
    dest_script.write_text(script.read_text(encoding="utf-8"), encoding="utf-8")

    env = dict(os.environ)
    if judge_model:
        env["MODEL_KEY"] = judge_model
    before = {p for p in sb.rglob("*") if p.is_file()}
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(dest_script)], cwd=sb, env=env,
                          capture_output=True, text=True)
    dt = time.time() - t0
    after = {p for p in sb.rglob("*") if p.is_file()}
    new_files = sorted(str(p.relative_to(sb)) for p in (after - before))
    info.update({"returncode": proc.returncode, "seconds": round(dt, 1),
                 "stdout_tail": proc.stdout[-1500:], "stderr_tail": proc.stderr[-1500:],
                 "new_files": new_files})
    # move any produced json back into artifacts/phase4/scoring
    SCORING.mkdir(parents=True, exist_ok=True)
    for nf in new_files:
        src = sb / nf
        if src.suffix == ".json":
            shutil.copy(src, SCORING / f"{key}__{Path(nf).name}")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", nargs="+", default=["all"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sandbox", default=str(P4 / "scoring" / "_sandbox"))
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke-test: judge only the first N records per dimension")
    args = ap.parse_args()

    dims = list(DIM_SCORERS) if args.dims == ["all"] else args.dims
    bad = [d for d in dims if d not in DIM_SCORERS]
    assert not bad, f"unknown dims: {bad}"

    if args.dry_run:
        for d in dims:
            print(json.dumps(plan_one(d), indent=1))
        return

    key_file = os.getenv("OPENAI_API_KEY_FILE", "/tmp/opencode/openai.key")
    if not os.getenv("OPENAI_API_KEY") and Path(key_file).exists():
        os.environ["OPENAI_API_KEY"] = Path(key_file).read_text().strip()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set — cannot run the official judge. "
                 "Set the env var, or write the key to OPENAI_API_KEY_FILE "
                 f"(default {key_file}). (Use --dry-run to validate wiring without a key.)")
    try:
        import openai  # noqa: F401
    except Exception:
        sys.exit("The 'openai' package is not installed in this interpreter. "
                 "Install it (pip install openai) then re-run.")

    results = {}
    for d in dims:
        print(f"[score] {d} ...", flush=True)
        results[d] = run_one(d, Path(args.sandbox), args.judge_model, args.limit)
        print(json.dumps({k: results[d].get(k) for k in
                          ("returncode", "seconds", "new_files", "error")}, indent=1))
    if METRIC.exists():
        m = json.loads(METRIC.read_text())
        m.setdefault("judge_scoring_runs", {})
        m["judge_scoring_runs"][time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())] = results
        METRIC.write_text(json.dumps(m, indent=2, ensure_ascii=False, default=str))
    print("done")


if __name__ == "__main__":
    main()