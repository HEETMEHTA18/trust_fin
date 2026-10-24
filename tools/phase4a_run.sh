#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Phase-4A sequential driver — TrustFin-SLM
#
# Reproduces the full documented Phase-4A sequence from a clean state:
#   1. configure input paths (Phase-3 ZIP, Phase-3.1 ZIP, Phase-2.1 validation)
#   2. RUN_MODE="smoke"  -> expect SMOKE PASS
#   3. RUN_MODE="full"   -> expect PASS + Phase4A_Review_Package.zip
#
# Usage:
#   bash tools/phase4a_run.sh            # smoke, then full
#   bash tools/phase4a_run.sh smoke      # smoke only
#   bash tools/phase4a_run.sh full       # full only
#
# Every run uses a FRESH kernel (separate nbconvert invocation), which is the
# "restart the kernel" step required by the notebook instructions.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NB="Phase4A_TrustFin_Execution_Verification_and_Financial_Grounding.ipynb"
NBEXEC="Phase4A_TrustFin_Execution_Verification_and_Financial_Grounding_executed.ipynb"
KERNEL="${TRUSTFIN_PHASE4A_KERNEL:-trustfin}"
MODE="${1:-both}"

cd "$REPO/notebooks"

run_mode () {
    local mode="$1"
    echo
    echo "======================================================================"
    echo " PHASE-4A RUN MODE = $mode"
    echo "======================================================================"

    python3 "$REPO/tools/phase4a_configure.py" --mode "$mode"

    # clean the output root for this mode so nothing stale can survive
    if [ "$mode" = "smoke" ]; then
        rm -rf "$REPO/artifacts/phase4a_smoke"
    else
        rm -rf "$REPO/artifacts/phase4a"
    fi

    # fresh kernel; the env var pins the mode so the notebook's own default
    # (full) can never be overridden accidentally
    env TRUSTFIN_PHASE4A_MODE="$mode" MPLBACKEND=Agg \
        jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.kernel_name="$KERNEL" \
        --ExecutePreprocessor.timeout=3600 \
        --output "$NBEXEC" "$NB"

    echo
    echo "---- readiness report ($mode) ----"
    local root="$REPO/artifacts/phase4a_smoke"
    [ "$mode" = "full" ] && root="$REPO/artifacts/phase4a"
    python3 - "$root" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rd = json.load(open(root / "manifests/phase4a_readiness.json"))
print(json.dumps({k: v for k, v in rd.items() if k != "source_fingerprint"}, indent=2))
print()
print("review package:", (root / "Phase4A_Review_Package.zip").stat().st_size, "bytes")
print("artifact files:", sum(1 for p in root.rglob("*") if p.is_file()))
PY
}

if [ "$MODE" = "smoke" ] || [ "$MODE" = "both" ]; then run_mode smoke; fi
if [ "$MODE" = "full" ] || [ "$MODE" = "both" ]; then run_mode full; fi

echo
echo "======================================================================"
echo " PHASE-4A SEQUENCE COMPLETE"
echo "======================================================================"
