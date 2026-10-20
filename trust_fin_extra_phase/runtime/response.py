"""response.py — build and validate the structured runtime response (roadmap §8)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / \
    "verified_calculation_response.schema.json"

STATUS_VALUES = ("answered", "clarify", "abstain", "refused", "escalate")


def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def make_response(
    status: str,
    *,
    answer: str | None = None,
    evidence: list[dict] | None = None,
    calculation: dict | None = None,
    confidence: float | None = None,
    limitations: list[str] | None = None,
    risk_flags: list[str] | None = None,
    model_version: str,
    policy_version: str,
) -> dict[str, Any]:
    """Assemble a schema-shaped response dict (does not validate)."""
    if status not in STATUS_VALUES:
        raise ValueError(f"invalid status {status!r}; expected one of {STATUS_VALUES}")
    return {
        "status": status,
        "answer": answer,
        "evidence": evidence or [],
        "calculation": calculation,
        "confidence": confidence,
        "limitations": limitations or [],
        "risk_flags": risk_flags or [],
        "model_version": model_version,
        "policy_version": policy_version,
    }


def validate_response(obj: dict) -> list[str]:
    """Return a list of schema violations (empty list == valid)."""
    import jsonschema  # imported lazily so the module loads without the dep
    errors = sorted(
        jsonschema.Draft7Validator(_schema()).iter_errors(obj),
        key=lambda e: list(e.path))
    out = []
    for e in errors:
        where = "/".join(str(p) for p in e.path) or "<root>"
        out.append(f"{where}: {e.message}")
    return out


def assert_valid(obj: dict) -> None:
    errs = validate_response(obj)
    if errs:
        raise ValueError("response violates schema:\n  " + "\n  ".join(errs))
