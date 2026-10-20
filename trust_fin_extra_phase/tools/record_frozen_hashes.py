#!/usr/bin/env python3
"""record_frozen_hashes.py — Phase-5A frozen-artifact preservation + release manifest.

Implements roadmap step 1 (docs/next_phases_roadmap.md §11): fingerprint every
frozen Phase-2.1 / Phase-3 / Phase-4 artifact with SHA-256, verify the dataset
hashes declared in the Phase-2.1 registry, and emit a release manifest that pins
model ID, adapter digest, dataset hashes, calibration policy, abstention
thresholds, code commit, seed and quantization.

The tool only READS frozen trees and WRITES under artifacts/phase5/manifests/:
  artifacts/phase5/manifests/frozen_artifact_hashes.json
  artifacts/phase5/manifests/release_manifest.json

Usage
  python3 tools/record_frozen_hashes.py            # record (write manifests)
  python3 tools/record_frozen_hashes.py --check    # recompute & compare (exit 1 on drift)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import find_repo_root, resolve_recorded  # noqa: E402

REPO = find_repo_root()
WS = REPO / "trust_fin_extra_phase"
OUT_DIR = WS / "artifacts" / "phase5" / "manifests"
HASHES_FILE = OUT_DIR / "frozen_artifact_hashes.json"
RELEASE_FILE = OUT_DIR / "release_manifest.json"

# Individual frozen files (repo-root-relative, POSIX).
SCOPE_FILES = [
    "artifacts/phase2_1/manifests/dataset_registry.json",
    "artifacts/phase2_1/manifests/phase3_data_config.json",
    "artifacts/phase3/manifests/phase3_freeze.json",
    "artifacts/phase3/manifests/phase3_completion_report.json",
    "artifacts/phase3/manifests/environment.json",
]

# Whole trees (dir digest = sha256 over sorted "relpath\\0filesha256" lines).
SCOPE_DIRS = [
    "artifacts/phase3/checkpoints/E5",                    # frozen QLoRA adapter
    "trust_fin_extra_phase/artifacts/phase4",             # frozen Phase-4 deliverable
    "trust_fin_extra_phase/artifacts/phase4_2",           # derived analysis (read-only)
]

CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def dir_digest(root: Path) -> tuple[str, int, int]:
    """Deterministic digest of a directory: sha256 over sorted relpath+filesha lines."""
    h = hashlib.sha256()
    n_files = n_bytes = 0
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        rel = p.relative_to(root).as_posix()
        fsha = sha256_file(p)
        h.update(f"{rel}\0{fsha}\n".encode())
        n_files += 1
        n_bytes += p.stat().st_size
    return h.hexdigest(), n_files, n_bytes


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def registry_declared_datasets(warnings: list[str]) -> dict:
    """Verify + hash every dataset/audit file declared by the Phase-2.1 registry.

    The registry (schema trustfin-phase2.1-v1.0) declares each contract file under
    ``artifacts.<name>.{path,sha256}``: train, validation, the protected eval
    splits, the FinTrust registry and the integrity audits.
    """
    reg_path = REPO / "artifacts/phase2_1/manifests/dataset_registry.json"
    out: dict[str, dict] = {}
    if not reg_path.exists():
        warnings.append("phase2_1 dataset registry missing — dataset hash check skipped")
        return out
    try:
        reg = json.loads(reg_path.read_text())
    except json.JSONDecodeError:
        warnings.append("phase2_1 dataset registry unreadable — dataset hash check skipped")
        return out
    for name, entry in sorted(reg.get("artifacts", {}).items()):
        p_raw, declared = entry.get("path"), entry.get("sha256")
        if not p_raw or not declared:
            warnings.append(f"registry entry {name!r} lacks path+sha256")
            continue
        p = resolve_recorded(p_raw)
        if not p.exists():
            warnings.append(f"registry artifact missing on this machine: {name} ({p_raw})")
            continue
        actual = sha256_file(p)
        rel = p.relative_to(REPO).as_posix() if p.is_relative_to(REPO) else str(p)
        out[rel] = {"sha256": actual, "bytes": p.stat().st_size,
                    "role": entry.get("role"), "matches_registry": actual == declared}
        if actual != declared:
            warnings.append(f"registry artifact SHA-256 MISMATCH: {name}")
    return out


def collect() -> tuple[dict, dict, list[str]]:
    warnings: list[str] = []
    files: dict[str, dict] = {}
    for rel in SCOPE_FILES:
        p = REPO / rel
        if p.exists():
            files[rel] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        else:
            warnings.append(f"frozen file missing: {rel}")
    files.update(registry_declared_datasets(warnings))
    dirs: dict[str, dict] = {}
    for rel in SCOPE_DIRS:
        root = REPO / rel
        if root.is_dir():
            digest, n, nbytes = dir_digest(root)
            dirs[rel] = {"sha256": digest, "files": n, "bytes": nbytes}
        else:
            warnings.append(f"frozen tree missing: {rel}")
    return files, dirs, warnings


def load_json(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def record() -> int:
    files, dirs, warnings = collect()
    commit = git_commit()
    now = datetime.now(timezone.utc).isoformat()

    hashes = {
        "schema": "trustfin-frozen-hashes-v1",
        "created_at_utc": now,
        "git_commit": commit,
        "files": files,
        "dir_digests": dirs,
        "notes": [
            "SHA-256 fingerprints of the frozen Phase-2.1/3/4 contract.",
            "verify with: python3 trust_fin_extra_phase/tools/record_frozen_hashes.py --check",
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HASHES_FILE.write_text(json.dumps(hashes, indent=2) + "\n")

    # ---- release manifest -------------------------------------------------
    freeze = load_json(REPO / "artifacts/phase3/manifests/phase3_freeze.json")
    p4_freeze = load_json(WS / "artifacts/phase4/manifests/phase4_freeze.json")
    env3 = load_json(REPO / "artifacts/phase3/manifests/environment.json")
    reg = load_json(REPO / "artifacts/phase2_1/manifests/dataset_registry.json")
    backbone = freeze.get("winner", {})
    release = {
        "schema": "trustfin-release-manifest-v1",
        "created_at_utc": now,
        "git_commit": commit,
        "phase": "5A-hardening",
        "model": {
            "model_id": backbone.get("model_id"),
            "base_experiment": backbone.get("experiment"),
            "adapter_dir": "artifacts/phase3/checkpoints/E5",
            "adapter_dir_digest_sha256": dirs.get(
                "artifacts/phase3/checkpoints/E5", {}).get("sha256"),
            "adapter_selection_score": backbone.get("selection_score"),
            "backbone_changed_by_phase4": p4_freeze.get("backbone", {}).get("changed_by_phase4"),
        },
        "training": {
            "method": "QLoRA (4-bit NF4, PEFT LoRA)",
            "quantization": "4-bit-nf4",
            "seed": env3.get("seed") or reg.get("seed"),
            "epochs": env3.get("training", {}).get("epochs", 2.0),
            "train_max_seq_length": env3.get("training", {}).get("max_seq_length"),
            "lora_r": env3.get("training", {}).get("lora_r"),
            "lora_alpha": env3.get("training", {}).get("lora_alpha"),
        },
        "datasets": {
            "registry_sha256": files.get(
                "artifacts/phase2_1/manifests/dataset_registry.json", {}).get("sha256"),
            "train_sha256": (reg.get("artifacts", {}).get("train") or {}).get("sha256"),
            "validation_sha256": (reg.get("artifacts", {}).get("validation") or {}).get("sha256"),
        },
        "calibration": {
            "split": "validation-only",
            "folds": p4_freeze.get("calibration", {}).get("folds"),
            "selection_rule": p4_freeze.get("calibration", {}).get("selection_rule"),
            "random_state": 42,
        },
        "abstention_thresholds": p4_freeze.get("operating_points"),
        "prompt_policy": p4_freeze.get("prompt_policy"),
        "evaluation": {
            "protected_test_note": (
                "Protected-test results are frozen evidence "
                "(trust_fin_extra_phase/artifacts/phase4/manifests/"
                "phase4_completion_report.json). Never tune on them."),
            "headline": {
                "finqa_protected_exec_accuracy_greedy": 0.6574,
                "finqa_error_reduction_at_80pct_coverage": 0.3103,
            },
        },
        "compatibility": {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "lockfile": "trust_fin_extra_phase/requirements.lock.txt",
            "test_command": "make test",
        },
    }
    RELEASE_FILE.write_text(json.dumps(release, indent=2) + "\n")

    total_bytes = sum(v["bytes"] for v in files.values()) + \
        sum(v["bytes"] for v in dirs.values())
    print(f"recorded {len(files)} files + {len(dirs)} trees "
          f"({total_bytes / 1e6:.1f} MB) at commit {commit[:12]}")
    print(f"  -> {HASHES_FILE.relative_to(REPO)}")
    print(f"  -> {RELEASE_FILE.relative_to(REPO)}")
    for w in warnings:
        print(f"  WARNING: {w}")
    return 0 if not any("MISMATCH" in w for w in warnings) else 1


def check() -> int:
    if not HASHES_FILE.exists():
        print("no stored manifest — run record first")
        return 1
    stored = json.loads(HASHES_FILE.read_text())
    files, dirs, warnings = collect()
    problems: list[str] = []
    for rel, meta in stored.get("files", {}).items():
        if rel not in files:
            problems.append(f"MISSING: {rel}")
        elif files[rel]["sha256"] != meta["sha256"]:
            problems.append(f"CHANGED: {rel}")
    for rel in files:
        if rel not in stored.get("files", {}):
            problems.append(f"UNTRACKED (not in manifest): {rel}")
    for rel, meta in stored.get("dir_digests", {}).items():
        if rel not in dirs:
            problems.append(f"MISSING TREE: {rel}")
        elif dirs[rel]["sha256"] != meta["sha256"]:
            problems.append(f"CHANGED TREE: {rel} "
                            f"({meta['files']} files recorded, {dirs[rel]['files']} found)")
    if stored.get("git_commit") != git_commit():
        print(f"note: commit moved ({stored.get('git_commit', '?')[:12]} -> "
              f"{git_commit()[:12]}); artifact comparison is what matters")
    for w in warnings:
        print(f"  WARNING: {w}")
    if problems:
        print(f"FROZEN ARTIFACT DRIFT: {len(problems)} problem(s)")
        for p in problems[:20]:
            print("  -", p)
        return 1
    n_files = len(stored.get("files", {}))
    print(f"OK: {n_files} files + {len(stored.get('dir_digests', {}))} trees "
          "match the recorded frozen manifest")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="recompute and compare against the stored manifest")
    a = ap.parse_args()
    sys.exit(check() if a.check else record())


if __name__ == "__main__":
    main()
