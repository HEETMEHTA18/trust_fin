"""Roadmap §11 step 1: frozen Phase-2.1/3/4 artifacts match their recorded hashes.

Skipped when the frozen trees are absent (e.g. the slim Docker test image,
which excludes artifacts/ — run `make verify-hashes` on the workstation).
"""
import subprocess
import sys
from pathlib import Path

import pytest

WS = Path(__file__).resolve().parents[1]
REPO = WS.parent
TOOL = WS / "tools" / "record_frozen_hashes.py"
HASHES = WS / "artifacts" / "phase5" / "manifests" / "frozen_artifact_hashes.json"

pytestmark = pytest.mark.skipif(
    not HASHES.exists() or not (WS / "artifacts" / "phase4").exists()
    or not (REPO / "artifacts" / "phase3").exists(),
    reason="frozen artifacts not present in this environment (full check: make verify-hashes)")


def test_frozen_artifacts_match_recorded_hashes():
    proc = subprocess.run([sys.executable, str(TOOL), "--check"],
                          capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, (
        f"frozen artifact drift detected:\n{proc.stdout}\n{proc.stderr}")


def test_release_manifest_is_complete():
    import json
    rel = json.loads(
        (WS / "artifacts" / "phase5" / "manifests" / "release_manifest.json").read_text())
    assert rel["model"]["model_id"] == "Qwen/Qwen3-1.7B"
    assert rel["model"]["adapter_dir_digest_sha256"]
    assert rel["training"]["seed"] == 42
    assert rel["training"]["quantization"] == "4-bit-nf4"
    assert rel["datasets"]["train_sha256"] and rel["datasets"]["validation_sha256"]
    thr = rel["abstention_thresholds"]["finqa"]["coverage_0.80"]["threshold"]
    assert 0 < thr < 1
