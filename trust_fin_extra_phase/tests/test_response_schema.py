"""Roadmap §11 step 4: JSON schemas + offline validation of the response contract."""
import json
from pathlib import Path

import pytest

from runtime import response

WS = Path(__file__).resolve().parents[1]
SAFETY_SCHEMA = json.loads(
    (WS / "runtime" / "schemas" / "safety_record.schema.json").read_text())
SAFETY_SET = WS / "data" / "safety" / "synthetic_safety_regression.jsonl"


def _valid(**overrides) -> dict:
    base = response.make_response(
        "answered", answer="150", confidence=0.92,
        calculation={"inputs": [], "operation": "a*b", "verified_result": 150,
                     "rendered": "150", "source": "verified_tool"},
        limitations=[], model_version="E5", policy_version="5A")
    base.update(overrides)
    return base


def test_valid_answered_response_passes():
    assert response.validate_response(_valid()) == []


def test_every_non_answered_status_validates():
    for st in ("clarify", "abstain", "refused", "escalate"):
        assert response.validate_response(_valid(status=st, answer=None)) == [], st


def test_unknown_status_rejected():
    errs = response.validate_response(_valid(status="maybe"))
    assert errs and "status" in errs[0]


def test_answered_without_answer_rejected():
    errs = response.validate_response(_valid(answer=None))
    assert any("answer" in e for e in errs)


def test_missing_required_field_rejected():
    bad = _valid()
    del bad["policy_version"]
    assert response.validate_response(bad)


def test_extra_property_rejected():
    errs = response.validate_response(_valid(hallucinated_field=1))
    assert errs


def test_confidence_out_of_range_rejected():
    assert response.validate_response(_valid(confidence=1.7))


def test_safety_records_match_schema_and_ids_unique():
    import jsonschema
    records = [json.loads(line) for line in
               SAFETY_SET.read_text().splitlines() if line.strip()]
    assert len(records) >= 30, "safety regression set too small"
    validator = jsonschema.Draft7Validator(SAFETY_SCHEMA)
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids)), "duplicate safety record ids"
    for r in records:
        errs = [f"{'/'.join(map(str, e.path))}: {e.message}"
                for e in validator.iter_errors(r)]
        assert not errs, f"{r['id']}: {errs}"
    # roadmap §6: every record carries provenance + expected action labels
    for r in records:
        assert r["source_license"].startswith("synthetic")
        assert r["split"] in ("red_team", "release_test", "train", "validation")
        assert r["pii_status"] in ("none", "synthetic", "redacted")
