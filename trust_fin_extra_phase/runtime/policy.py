"""policy.py — Phase-5A runtime policy (roadmap docs/next_phases_roadmap.md §4/§6).

Deterministic, offline, model-free input routing and decision logic:

  1. prompt-injection markers  -> abstain (treat input as data, never follow it)
  2. requests for PII          -> refuse
  3. personalized financial advice / harmful asks -> escalate
  4. missing/conflicting data  -> clarify
  5. unverified calculations   -> abstain (no executor result, no number shown)
  6. calibrated-confidence gate -> answered only at/above the FROZEN validation
                                   threshold, and only for families Phase 4
                                   validated (FinQA). No cross-family transfer.
  7. otherwise                 -> answered (with evidence/limitations attached)

The SLM, retrieval and calculator are wired in behind this contract in Phase 6/8.
Nothing here calls a model or the network.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.paths import resolve_recorded                     # noqa: E402
from runtime import answer_render, redaction, response       # noqa: E402

POLICY_VERSION = "trustfin-policy-5A-v1"
MODEL_VERSION = "E5-qwen3-1.7b-qlora-frozen"

#: Only families whose abstention was validated on Phase-4 *validation* data may
#: use automatic thresholds (roadmap: never copy the FinQA calibrator elsewhere).
ENABLED_ABSTENTION_FAMILIES = ("finqa",)

COVERAGE_TARGET = "coverage_0.80"
FINQA_FREEZE_REL = "trust_fin_extra_phase/artifacts/phase4/manifests/phase4_freeze.json"

_PATTERNS: dict[str, re.Pattern] = {
    "injection": re.compile(
        r"(?i)\b(ignore|disregard|forget)\b.{0,50}\b(instructions?|rules?|prompt|policies)\b"
        r"|\bprevious instructions\b|\bsystem prompt\b|\byou must now\b"
        r"|\bnew instructions\b|\bobey only me\b"),
    "pii_request": re.compile(
        r"(?i)\b(ssn|social security|account numbers?|routing numbers?"
        r"|credit cards?|pins?|passwords?|credentials?|ibans?|client records?)\b"),
    "harmful_advice": re.compile(
        r"(?i)\b(should i (?:buy|sell|short|invest|exercise)"
        r"|recommend(?:ed|ation|ations)?\b.{0,30}\b(?:i|me|my)"
        r"|guaranteed returns?|risk[- ]free (?:profit|investment)"
        r"|i'?m authorizing|place (?:the|my) trade"
        r"|approve (?:the|my) (?:credit|transfer)"
        r"|buy[ /-]or[ /-]?sell|personaliz(?:ed|ation)\b.{0,30}\b(?:advice|recommendations?|signals?)"
        r"|(?:evade|evasion|hide)\b.{0,30}\b(?:tax|income|reporting|authorit))\b"),
    "calculation": re.compile(
        r"(?i)\b(?:compute|calculate|cagr|ebitda|percentage change|cash ?flow"
        r"|volatility)\b"),
    "uncertainty": re.compile(
        r"(?i)\b(?:insufficient|not enough data"
        r"|conflicting (?:documents?|sources?|figures?)"
        r"|which (?:quarter|year|filing)|as of (?:what|which)"
        r"|missing (?:table|data|value|figure|figures)?|ambiguous"
        r"|cannot determine|stale)\b"),
}


def route(text: str) -> dict:
    """Classify an input: flags, risk level, task family guess, PII spans."""
    flags = [name for name, rx in _PATTERNS.items() if rx.search(text)]
    pii = redaction.find_pii(text)
    if pii:
        flags.append("pii_in_input")
    if "harmful_advice" in flags:
        risk = "high"
    elif {"pii_request", "pii_in_input", "injection"} & set(flags):
        risk = "medium"
    else:
        risk = "low"
    family = "calculation" if "calculation" in flags else "general"
    return {"task_family": family, "risk_level": risk, "flags": flags, "pii": pii}


def load_frozen_thresholds() -> dict:
    """Read operating-point thresholds from the frozen Phase-4 manifest.

    Returns e.g. {"finqa": {"coverage_0.80": 0.40899..., ...}, "fiqa": {...}}.
    Empty dict when the frozen manifest is not on this machine.
    """
    p = resolve_recorded(FINQA_FREEZE_REL)
    if not p.exists():
        return {}
    try:
        fz = json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}
    out: dict[str, dict] = {}
    for fam, pts in (fz.get("operating_points") or {}).items():
        if isinstance(pts, dict):
            out[fam] = {k: v.get("threshold") for k, v in pts.items()
                        if isinstance(v, dict)}
    return out


def _answered(answer: str, *, calculation: dict | None, flags: list[str],
              confidence: float | None, limitations: list[str]) -> dict:
    if calculation is None:
        limitations = limitations + ["no-retrieval-attached"]
    return response.make_response(
        "answered",
        answer=answer,
        calculation=calculation,
        confidence=confidence,
        limitations=limitations,
        risk_flags=flags,
        model_version=MODEL_VERSION,
        policy_version=POLICY_VERSION,
    )


def decide(
    text: str,
    *,
    verified_result: int | float | str | None = None,
    draft_answer: str | None = None,
    printed_answer: str | None = None,
    confidence: float | None = None,
    coverage_target: str = COVERAGE_TARGET,
    task_family: str | None = None,
) -> dict:
    """Route and decide. Always returns a schema-valid structured response."""
    r = route(text)
    flags = list(r["flags"])
    fam = task_family or r["task_family"]
    limitations: list[str] = []

    # 1. injection: input is data, never instructions
    if "injection" in flags:
        return response.make_response(
            "abstain",
            limitations=["untrusted-instruction-marker: input treated as data"],
            confidence=confidence, risk_flags=flags,
            model_version=MODEL_VERSION, policy_version=POLICY_VERSION)

    # 2. privacy: never supply personal data
    if "pii_request" in flags:
        return response.make_response(
            "refused",
            limitations=["privacy-policy: personal data is never disclosed"],
            risk_flags=flags, model_version=MODEL_VERSION,
            policy_version=POLICY_VERSION)

    # 3. safety: individualized financial advice / harmful asks escalate
    if "harmful_advice" in flags:
        return response.make_response(
            "escalate",
            limitations=["not-individualized-advice: requires a licensed professional"],
            confidence=confidence, risk_flags=flags,
            model_version=MODEL_VERSION, policy_version=POLICY_VERSION)

    # 4. missing or conflicting evidence
    if "uncertainty" in flags:
        return response.make_response(
            "clarify",
            limitations=["missing-or-conflicting-evidence"],
            risk_flags=flags, model_version=MODEL_VERSION,
            policy_version=POLICY_VERSION)

    # 5. calculations only ever display executor-verified numbers
    calc: dict | None = None
    if verified_result is not None:
        calc = {
            "inputs": [],
            "operation": None,
            "verified_result": verified_result,
            "rendered": answer_render.render_verified(
                {"verified_result": verified_result}),
            "source": "verified_tool",
        }
    elif fam == "calculation" or "calculation" in flags:
        why = ("printed-answer-not-executor-verified: number withheld"
               if printed_answer is not None else
               "calculation-not-verified: no executor result available")
        return response.make_response(
            "abstain",
            limitations=[why],
            confidence=confidence, risk_flags=flags,
            model_version=MODEL_VERSION, policy_version=POLICY_VERSION)

    # 6. calibrated abstention gate (frozen Phase-4 thresholds, FinQA only)
    if confidence is not None:
        thresholds = load_frozen_thresholds()
        fam_thresholds = thresholds.get(fam) or {}
        if fam not in ENABLED_ABSTENTION_FAMILIES or not fam_thresholds:
            return response.make_response(
                "abstain",
                limitations=["no-validated-abstention-for-family"],
                confidence=confidence, risk_flags=flags,
                model_version=MODEL_VERSION, policy_version=POLICY_VERSION)
        thr = fam_thresholds.get(coverage_target)
        if thr is None:
            return response.make_response(
                "abstain",
                limitations=["no-frozen-threshold-for-target"],
                confidence=confidence, risk_flags=flags,
                model_version=MODEL_VERSION, policy_version=POLICY_VERSION)
        if confidence < thr:
            return response.make_response(
                "abstain",
                limitations=["confidence-below-frozen-validation-threshold"],
                confidence=confidence, risk_flags=flags,
                model_version=MODEL_VERSION, policy_version=POLICY_VERSION)

    # 7. answered
    if verified_result is not None:
        answer = calc["rendered"]
        if printed_answer is not None and not answer_render.printed_matches_verified(
                printed_answer, verified_result):
            limitations.append("printed-answer-mismatch-corrected-by-executor")
    elif draft_answer is not None:
        # a leaked PROGRAM/ANSWER transcript must never reach the user verbatim
        answer = answer_render.strip_program_answer(str(draft_answer))
    else:
        return response.make_response(
            "abstain",
            limitations=["no-answer-available"],
            confidence=confidence, risk_flags=flags,
            model_version=MODEL_VERSION, policy_version=POLICY_VERSION)

    if "pii_in_input" in flags:
        limitations.append("input-pii-redacted")
    return _answered(answer, calculation=calc, flags=flags,
                     confidence=confidence, limitations=limitations)
