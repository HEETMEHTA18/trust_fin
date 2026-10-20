"""Roadmap §4 / §5: protected-test access controls stay enforced offline.

The protected test may only run when the freeze manifest says FROZEN **and**
TRUSTFIN4_PROTECTED=1 is supplied (two-key gate in tools/phase4_shard.py).
These tests drive the real shard tool and assert it never launches a run
without both keys — in any environment, including containers without artifacts.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

WS = Path(__file__).resolve().parents[1]
REPO = WS.parent
SHARD = WS / "tools" / "phase4_shard.py"
FREEZE = WS / "artifacts" / "phase4" / "manifests" / "phase4_freeze.json"


def _run(args, extra_env=None):
    env = {k: v for k, v in os.environ.items()
           if k not in ("TRUSTFIN4_PROTECTED", "TRUSTFIN4_RUN_MODE")}
    env.update(extra_env or {})
    return subprocess.run([sys.executable, str(SHARD), *args],
                          capture_output=True, text=True, cwd=WS, env=env)


def test_protected_test_refused_without_second_key():
    proc = _run(["run", "P4TEST", "--mode", "full"])
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 1, combined
    assert "started pid=" not in combined, "protected run launched without key!"
    # every environment path (missing artifacts, preflight failure, frozen
    # manifest present) must end in an explicit refusal
    assert ("refusing" in combined) or ("preflight" in combined), combined
    if FREEZE.exists():
        assert "TRUSTFIN4_PROTECTED" in combined, combined


def test_protected_test_dry_run_allowed_with_key():
    if not FREEZE.exists():
        pytest.skip("freeze manifest absent (container image)")
    proc = _run(["run", "P4TEST", "--mode", "full"],
                extra_env={"TRUSTFIN4_PROTECTED": "1"})
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
    assert "(dry run" in combined          # --run was NOT passed: nothing executes
    assert "started pid=" not in combined


def test_freeze_manifest_reports_frozen():
    if not FREEZE.exists():
        pytest.skip("freeze manifest absent (container image)")
    status = json.loads(FREEZE.read_text())["status"]
    assert status == "FROZEN"
