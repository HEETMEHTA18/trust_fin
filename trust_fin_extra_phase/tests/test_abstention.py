"""Roadmap §4/§5: calibrated abstention uses the FROZEN Phase-4 thresholds and
only for the family where validation proved it works (FinQA). No transfer."""
import json
from pathlib import Path

import pytest

from runtime import policy

WS = Path(__file__).resolve().parents[1]
REPO = WS.parent
FREEZE = WS / "artifacts" / "phase4" / "manifests" / "phase4_freeze.json"

pytestmark = pytest.mark.skipif(
    not FREEZE.exists(), reason="frozen Phase-4 manifest absent in this environment")


def test_thresholds_read_from_freeze_exactly():
    stored = json.loads(FREEZE.read_text())["operating_points"]
    loaded = policy.load_frozen_thresholds()
    for fam in ("finqa", "fiqa"):
        for target in ("coverage_0.80", "coverage_0.90"):
            assert loaded[fam][target] == stored[fam][target]["threshold"]
    assert loaded["finqa"]["coverage_0.80"] == pytest.approx(0.4090, abs=1e-3)
    assert loaded["finqa"]["coverage_0.90"] == pytest.approx(0.2211, abs=1e-3)


def test_below_threshold_abstains_above_answers():
    text = "Calculate the EBITDA margin from the quarterly filing."
    below = policy.decide(text, verified_result=150, confidence=0.15,
                          task_family="finqa")
    assert below["status"] == "abstain"
    assert "confidence-below-frozen-validation-threshold" in below["limitations"]
    above = policy.decide(text, verified_result=150, confidence=0.90,
                          task_family="finqa")
    assert above["status"] == "answered"
    assert above["answer"] == "150"


def test_high_raw_confidence_outside_finqa_never_auto_answers():
    # roadmap: do NOT copy the FinQA calibrator to other families
    r = policy.decide("Classify the filing stance.", draft_answer="neutral",
                      confidence=0.95, task_family="classification")
    assert r["status"] == "abstain"
    assert "no-validated-abstention-for-family" in r["limitations"]


def test_fiqa_threshold_not_auto_enabled_despite_being_frozen():
    assert "fiqa" not in policy.ENABLED_ABSTENTION_FAMILIES
    r = policy.decide("What does this filing say about capex?", draft_answer="x",
                      confidence=0.99, task_family="fiqa")
    assert r["status"] == "abstain"


def test_no_confidence_no_gate_without_calibrator():
    # answered paths without confidence still work (Phase-5A stub answers)
    r = policy.decide("According to the report, what was the operating margin?",
                      draft_answer="18.4%")
    assert r["status"] == "answered"
    assert r["limitations"] == ["no-retrieval-attached"]
