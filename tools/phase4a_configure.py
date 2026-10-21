#!/usr/bin/env python3
"""Configure the Phase-4A notebook for a local run.

Idempotent: re-running this script always yields the same configured notebook.
It performs exactly two edits to
``notebooks/Phase4A_TrustFin_Execution_Verification_and_Financial_Grounding.ipynb``:

1. **Configuration cell (cell 3)** — replaces the environment-only defaults with
   absolute, repo-relative paths for the Phase-3 ZIP, the Phase-3.1 ZIP and the
   original Phase-2.1 ``validation.jsonl``; makes ``RUN_MODE`` default to
   ``smoke``; and makes ``ROOT`` mode-aware so a smoke run can never contaminate
   the full-run review package.

2. **Evidence-loading cell (cell 11)** — broadens the validation-split guard from
   ``{'dev','validation'}`` to ``{'dev','valid','validation'}``.

   Why this is required and safe: in the frozen Phase-2.1 validation export the
   FinQA rows carry ``split="dev"`` while the FiQA rows carry ``split="valid"``
   (FiQA's official HuggingFace split name; see
   ``artifacts/phase2_1/manifests/source_fingerprints.json`` and the passing
   "FiQA valid count" check in
   ``artifacts/phase2_1/manifests/phase2_1_validation_report.json``).
   The original guard therefore raised ``AssertionError`` on all 117 FiQA rows
   and blocked the notebook's own *recommended* configuration (mounting the
   original Phase-2.1 validation file). The guard still rejects every
   test / protected / FOMC / FinTrust split token, so the validation-only
   intent is preserved exactly.

Usage:
    python tools/phase4a_configure.py            # default mode: smoke
    python tools/phase4a_configure.py --mode full
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NB = REPO / "notebooks" / "Phase4A_TrustFin_Execution_Verification_and_Financial_Grounding.ipynb"

CONFIG_CELL_TEMPLATE = '''from __future__ import annotations
import os, io, json, re, math, csv, zipfile, hashlib, shutil, sys, platform, statistics
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# RUN MODE  ->  'smoke' (35 FinQA + 15 FiQA subset)  or  'full' (883 + 117)
# The smoke run has been validated (SMOKE PASS). This notebook is now pinned
# to 'full'. The env var TRUSTFIN_PHASE4A_MODE still overrides this value.
# ---------------------------------------------------------------------------
RUN_MODE = os.environ.get('TRUSTFIN_PHASE4A_MODE', '{mode}').strip().lower()  # smoke | full
SEED = 42

def _discover_repo(start: Path) -> Path:
    """Walk up from the launch directory until the TrustFin repo root is found."""
    for cand in [start, *start.parents]:
        if ((cand / 'artifacts' / 'phase2_1' / 'datasets' / 'validation.jsonl').is_file()
                and (cand / 'deliverables' / 'Phase3_TrustFin_Deliverable.zip').is_file()):
            return cand
    return Path('/home/heet18/Downloads/Research Papers')

REPO = Path(os.environ.get('TRUSTFIN_PHASE4A_REPO', str(_discover_repo(Path.cwd())))).expanduser()
BASE = Path(os.environ.get('TRUSTFIN_PHASE4A_INPUT_DIR', str(REPO))).expanduser()
# Smoke writes to its own root so it can never contaminate the full review package.
ROOT = Path(os.environ.get('TRUSTFIN_PHASE4A_OUTPUT',
                           str(BASE / ('artifacts/phase4a_smoke' if RUN_MODE == 'smoke' else 'artifacts/phase4a'))
                           )).expanduser()
PHASE3_ZIP = Path(os.environ.get('TRUSTFIN_PHASE3_ZIP',
                                 str(BASE / 'deliverables/Phase3_TrustFin_Deliverable.zip'))).expanduser()
PHASE3_1_ZIP = Path(os.environ.get('TRUSTFIN_PHASE3_1_ZIP',
                                   str(BASE / 'artifacts/phase3_1/Phase3_1_Review_Package.zip'))).expanduser()
# Original Phase-2.1 validation.jsonl -> upgrades evidence provenance to RECOMPUTED.
PHASE2_VAL_OVERRIDE = os.environ.get('TRUSTFIN_PHASE2_VAL',
                                      str(BASE / 'artifacts/phase2_1/datasets/validation.jsonl'))
OFFICIAL_EVALUATOR_OVERRIDE = os.environ.get('TRUSTFIN_FINQA_EVALUATOR','').strip()
FINQA_DEV_OVERRIDE = os.environ.get('TRUSTFIN_FINQA_DEV_JSON','').strip()
E5_ADAPTER_OVERRIDE = os.environ.get('TRUSTFIN_E5_ADAPTER','').strip()

assert RUN_MODE in {'smoke','full'}
for p in (PHASE3_ZIP,PHASE3_1_ZIP):
    assert p.is_file(), f'Missing ZIP: {p}. Set input path in configuration.'
for part in ['tables','analysis','features','labels','manifests','plots','phase4b_interface']:
    (ROOT/part).mkdir(parents=True, exist_ok=True)
print('Mode:',RUN_MODE)
print('Repo:',REPO)
print('Phase-3:',PHASE3_ZIP)
print('Phase-3.1:',PHASE3_1_ZIP)
print('Phase-2.1 validation:',PHASE2_VAL_OVERRIDE)
print('Output:',ROOT)
'''

SPLIT_GUARD_OLD = "    assert all(str(ORIGINAL[k]['split']).lower() in {'dev','validation'} for k in frozen.record_id)"
SPLIT_GUARD_NEW = (
    "    # Phase-2.1 validation export: FinQA rows carry split='dev', FiQA rows carry\n"
    "    # split='valid' (FiQA's official HF validation split). Both are validation\n"
    "    # splits; every test/protected/FOMC/FinTrust token remains rejected.\n"
    "    assert all(str(ORIGINAL[k]['split']).lower() in {'dev','valid','validation'} for k in frozen.record_id)"
)

# Cell 31 builds the artifact inventory twice. Both passes listed
# `manifests/artifact_inventory.json` itself, so the manifest recorded a hash of
# its own *previous* write and the final rewrite invalidated that hash. Any
# auditor verifying the review package therefore hit exactly one spurious
# mismatch. The review ZIP was already excluded for the same reason; the
# inventory file is now excluded from its own listing as well, so every entry
# in `artifact_inventory.json` verifies byte-for-byte.
INVENTORY_EXCLUDE_OLD = "    if path.is_file() and path.name!='Phase4A_Review_Package.zip':"
INVENTORY_EXCLUDE_NEW = ("    if (path.is_file() and path.name not in ('Phase4A_Review_Package.zip',\n"
                          "                                                'artifact_inventory.json')):")


def _set_source(cell, text: str) -> None:
    lines = text.splitlines(keepends=True)
    cell["source"] = lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke",
                    help="value written into the notebook's RUN_MODE default")
    ap.add_argument("--notebook", default=str(NB), help="notebook to configure")
    args = ap.parse_args()

    nb_path = Path(args.notebook).expanduser()
    if not nb_path.is_file():
        print(f"Notebook not found: {nb_path}", file=sys.stderr)
        return 1

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb["cells"]

    config_idx = None
    evidence_idx = None
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if c.get("cell_type") != "code":
            continue
        if "TRUSTFIN_PHASE4A_MODE" in src and "PHASE3_1_ZIP" in src:
            config_idx = i
        if "ORIGINAL[k]['split']" in src:
            evidence_idx = i

    if config_idx is None or evidence_idx is None:
        print("Could not locate the configuration / evidence cells.", file=sys.stderr)
        return 1

    _set_source(cells[config_idx], CONFIG_CELL_TEMPLATE.replace("{mode}", args.mode))

    ev = cells[evidence_idx]
    src = "".join(ev["source"])
    if SPLIT_GUARD_OLD in src:
        src = src.replace(SPLIT_GUARD_OLD, SPLIT_GUARD_NEW)
    elif SPLIT_GUARD_NEW.split("\n")[3] not in src:
        print("Split guard line not found in evidence cell.", file=sys.stderr)
        return 1
    _set_source(ev, src)

    # --- edit 3: self-consistent artifact inventory (cell 31) -----------------
    inv_idx = None
    for i, c in enumerate(cells):
        if c.get("cell_type") != "code":
            continue
        s = "".join(c.get("source", []))
        if INVENTORY_EXCLUDE_OLD in s or INVENTORY_EXCLUDE_NEW in s:
            inv_idx = i
    if inv_idx is None:
        print("Artifact-inventory loop not found.", file=sys.stderr)
        return 1
    inv_src = "".join(cells[inv_idx]["source"])
    n_replaced = inv_src.count(INVENTORY_EXCLUDE_OLD)
    inv_src = inv_src.replace(INVENTORY_EXCLUDE_OLD, INVENTORY_EXCLUDE_NEW)
    _set_source(cells[inv_idx], inv_src)

    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Configured: {nb_path}")
    print(f"  configuration cell  -> index {config_idx} (RUN_MODE default = {args.mode})")
    print(f"  split-guard cell    -> index {evidence_idx}")
    print(f"  inventory cell      -> index {inv_idx} ({n_replaced} loop(s) patched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
