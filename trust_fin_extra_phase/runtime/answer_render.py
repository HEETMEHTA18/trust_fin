"""answer_render.py — render the user-visible number from the verified result.

Roadmap §4 / Phase-4 finding R1 (docs/eval_rigor.md §3): on 31 % of FinQA
protected-test records the model computed the correct value yet printed a
different answer string. The runtime therefore NEVER displays the model's
printed number when an executor-verified result exists; the displayed string is
rendered from the verified value, and printed-vs-verified mismatches are
detectable with :func:`printed_matches_verified`.
"""
from __future__ import annotations

import math
import re


_ANSWER_LINE = re.compile(r"(?i)^answer\s*:")


def strip_program_answer(text: str) -> str:
    """Collapse a leaked FinQA ``PROGRAM/ANSWER`` transcript to its payload.

    Phase-4 failure mode (roadmap §5): the model stays locked to the FinQA
    ``PROGRAM/ANSWER`` format on unrelated tasks. When a raw model draft still
    carries that transcript, the runtime displays only the value after the
    last ``ANSWER:`` line — never the program block.
    """
    lines = text.splitlines()
    for line in reversed(lines):
        s = line.strip()
        if _ANSWER_LINE.match(s):
            payload = s.split(":", 1)[1].strip()
            if payload:
                return payload
            break
    return text


def format_number(value: int | float) -> str:
    """Deterministic, locale-free rendering of a numeric verified result."""
    if isinstance(value, bool):
        raise TypeError("bool is not a numeric result")
    if isinstance(value, int):
        return str(value)
    f = float(value)
    if not math.isfinite(f):
        raise ValueError(f"non-finite verified result: {value!r}")
    return f"{f:.12g}"


def render_verified(calculation: dict) -> str:
    """Render the display string from a calculation's verified_result."""
    vr = calculation.get("verified_result")
    if vr is None:
        raise ValueError("no verified_result to render")
    return vr if isinstance(vr, str) else format_number(vr)


def _parse_number(text) -> float | None:
    try:
        s = str(text).strip().rstrip("%").replace(",", "").replace("$", "")
        return float(s)
    except (TypeError, ValueError):
        return None


def printed_matches_verified(
    printed, verified, *, rel_tol: float = 1e-9, abs_tol: float = 1e-9,
) -> bool:
    """True only when the model's printed string equals the verified value.

    Unparseable input counts as a mismatch (the printed string cannot be shown
    as the verified answer).
    """
    pv = _parse_number(printed)
    vv = _parse_number(verified)
    if pv is None or vv is None:
        return False
    return math.isclose(pv, vv, rel_tol=rel_tol, abs_tol=abs_tol)
