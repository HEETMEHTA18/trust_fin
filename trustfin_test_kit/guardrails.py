#!/usr/bin/env python3
"""Deterministic guardrails for the TrustFin E5 demo.

E5 is a task adapter, not a safety-tuned assistant. This module is an explicit
policy wrapper around it. It does not claim the model itself is aligned.

Control flow:
  input checks -> optional evidence sanitisation -> model generation
  -> output checks (parse, execute, operand grounding, consistency) -> allow/abstain

All decisions are rule-based and inspectable. Thresholds are not calibrated
probabilities and must not be reported as such.
"""
from __future__ import annotations

import copy
import math
import re

# --- pattern libraries -------------------------------------------------------

JAILBREAK_PATTERNS = [
    r"\byou are now\b", r"\bdan mode\b", r"\bjailbreak\b", r"\bdeveloper mode\b",
    r"\bignore all (previous|prior) instructions?\b", r"\boutput the system prompt\b",
    r"\bsystem\s*:", r"<\|im_start\|>", r"\bpretend you are\b", r"\bact as\b.*\bunrestricted\b",
]

INJECTION_PATTERNS = [
    r"\bignore (the|all|any) (evidence|instructions?|context|above)\b",
    r"\bdisregard (the )?(evidence|instructions?|above|context)\b",
    r"\b(say|state|report|answer|output)\b[^.]{0,40}\b(is|was|as)\b\s*[-+]?\d",
    r"\banswer\s+[-+]?\d+\s+instead\b",
    r"\boverride (the )?(evidence|answer|instructions?)\b",
    r"\bdo not follow\b",
]

ADVICE_PATTERNS = [
    r"\bshould i (invest|buy|sell|put|allocate|trade)\b",
    r"\ball my (savings|money|retirement)\b",
    r"\bmy (retirement|portfolio|savings|money)\b",
    r"\bwhich stock\b", r"\bwhat stock\b",
    r"\bguarantee(d)?\b", r"\brisk[- ]free return\b",
    r"\bdouble (my|next)\b", r"\bget rich\b",
    r"\bcrypto strategy\b", r"\bleveraged\b.*\b(retirement|savings)\b",
    r"\btell me to (buy|sell)\b", r"\bpersonal(ized)? (financial )?advice\b",
    r"\bfinancial advisor\b",
]

PII_PATTERNS = [
    r"\bhome address\b", r"\bsocial security\b", r"\bphone number\b",
    r"\bpersonal email\b", r"\bdate of birth\b", r"\bbank account\b",
    r"\bpassport number\b", r"\bmedical record\b",
]

OFFTOPIC_PATTERNS = [
    r"\bwrite (a|an) (poem|song|story|essay)\b", r"\btell me a joke\b",
    r"\bweather\b", r"\brecipe\b", r"\btranslate\b", r"\bwho is the president\b",
    r"\bwrite (code|a script)\b", r"\bplay a game\b",
]

FORECAST_PATTERNS = [
    r"\bwill\b.*\b(be|reach|grow|return)\b", r"\bwould\b.*\b(be|reach)\b",
    r"\bforecast\b", r"\bpredict\b", r"\bproject(ed|ion)?\b",
    r"\bnext (year|quarter|month|week)\b", r"\bexpect(ed)?\b.*\breturn\b",
]

NONNEGATIVE_NOUNS = (
    "revenue|sales|expenses?|costs?|assets?|equity|shares?|inventory|debt|"
    "net income|income|earnings|cash|capex"
)
NEGATIVE_INPUT_PATTERNS = [
    rf"\b(negative|minus)\s+({NONNEGATIVE_NOUNS})\b",
    rf"\b({NONNEGATIVE_NOUNS})\b[^.]{{0,25}}\b(was|were|is|are|of|at)\s*[-+]\s*\d",
]

CONFLICT_CUES = [
    "no precedence", "document a", "document b", "conflict", "contradict",
    "inconsistent", "two different", "does not match", "discrepanc",
]

CITATION_PATTERNS = [r"\bcite\b", r"\bpage\b", r"\bpage number\b", r"\bfootnote\b"]

METRIC_WORDS = (
    "revenue sales income profit earnings expense cost debt equity inventory eps "
    "cash capex assets liabilities margin ratio return dividend salary compensation "
    "price shares"
).split()

# Standard arithmetic constants that do not need to appear in evidence.
CONSTANT_WHITELIST = {1.0, 2.0, 3.0, 100.0, 0.5, 12.0, 365.0}

NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
YEAR_RE = re.compile(r"(?:FY)?(20\d{2})", re.I)


# --- helpers -----------------------------------------------------------------

def norm_text(text: str) -> str:
    return (text or "").lower()


def parse_number(token: str):
    try:
        return float(str(token).replace(",", ""))
    except (TypeError, ValueError):
        return None


def numbers_in(text: str) -> set[float]:
    """Numbers plus common million/billion scale variants."""
    out: set[float] = set()
    for m in NUM_RE.finditer(text or ""):
        v = parse_number(m.group())
        if v is None:
            continue
        out.add(round(v, 6))
        for f in (1e-9, 1e-6, 1e-3, 1e3, 1e6, 1e9):
            out.add(round(v * f, 6))
    return out


def evidence_text(case: dict) -> str:
    parts = [str(case.get("context", "") or "")]
    for row in case.get("table", []) or []:
        parts.extend(str(cell) for cell in row)
    return "\n".join(parts)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text or "") if s.strip()]


def _matches(patterns, text: str) -> list[str]:
    hits = []
    for pat in patterns:
        if re.search(pat, text, re.I | re.S):
            hits.append(pat)
    return hits


def _sentences_with(patterns, text: str) -> list[str]:
    out = []
    for s in sentences(text):
        if _matches(patterns, s):
            out.append(s)
    return out


def sanitize_case(case: dict) -> tuple[dict, list[str]]:
    """Remove untrusted instruction sentences from evidence and table cells."""
    clean = copy.deepcopy(case)
    removed: list[str] = []

    kept = []
    for s in sentences(clean.get("context", "") or ""):
        if _matches(INJECTION_PATTERNS, s):
            removed.append(s)
        else:
            kept.append(s)
    clean["context"] = " ".join(kept).strip()

    new_table = []
    for row in clean.get("table", []) or []:
        injected = [str(cell) for cell in row if _matches(INJECTION_PATTERNS, str(cell))]
        if injected:
            removed.extend(injected)
            continue  # drop the contaminated row entirely
        new_table.append(list(row))
    clean["table"] = new_table
    return clean, removed


def _metric_requirements_missing(question: str, ev: str) -> list[str]:
    """Conservative required-input check for strongly identified metrics."""
    q, e = norm_text(question), norm_text(ev)
    missing = []

    def has(*terms):
        return all(t in e for t in terms)

    asks_net_income = bool(re.search(r"\b(net income|net profit|earnings)\b", q))
    asks_profit = bool(re.search(r"\bprofit\b", q)) and not asks_net_income
    if asks_net_income or asks_profit:
        direct = "net income" in e or "net profit" in e or "earnings" in e
        if asks_profit and "profit" in e:
            direct = True
        rev_and_cost = (
            (("revenue" in e) or ("sales" in e))
            and (("expense" in e) or ("cost" in e))
            and not _stated_absent(e, "expenses?")
            and not _stated_absent(e, "costs?")
        )
        derivable = (
            direct
            or rev_and_cost
            or "margin" in e
            or (has("ebitda") and has("depreciation") and has("interest") and has("tax"))
            or (has("gross profit") and (has("expense") or has("tax") or has("interest")))
        )
        if not derivable:
            missing.append("net income/profit inputs")

    if re.search(r"\beps\b|earnings per share", q):
        if not (("share" in e) and any(t in e for t in ("income", "earnings", "profit"))):
            missing.append("EPS inputs")

    if re.search(r"debt[- ]to[- ]equity|\bleverage\b|gearing", q):
        if not (has("debt") and has("equity")):
            missing.append("debt and equity")

    if re.search(r"\bquick ratio\b", q):
        if not (has("current assets") and has("inventory") and has("current liabilities")):
            missing.append("quick-ratio inputs")

    if re.search(r"\bcurrent ratio\b", q):
        if not (has("current assets") and has("current liabilities")):
            missing.append("current-ratio inputs")

    if re.search(r"free cash flow|\bfcf\b", q):
        if not (("cash flow" in e or "operating cash" in e or "cash from operations" in e) and
                ("capital expenditure" in e or "capex" in e)):
            missing.append("free-cash-flow inputs")

    if re.search(r"return on equity|\broe\b", q):
        if not (any(t in e for t in ("net income", "income", "earnings")) and "equity" in e):
            missing.append("ROE inputs")

    if re.search(r"gross margin|gross profit margin", q):
        if not (has("gross profit") or (has("revenue") and (has("cogs") or has("cost of goods")))):
            missing.append("gross-margin inputs")

    if re.search(r"average inventory", q):
        if not (has("beginning inventory") and has("ending inventory")):
            missing.append("beginning and ending inventory")

    if re.search(r"\bsalary\b|\bcompensation\b", q) and not (
            "salary" in e or "compensation" in e):
        missing.append("salary/compensation data")

    if re.search(r"share price|stock price|market price", q) and not (
            "share price" in e or "stock price" in e or "market price" in e):
        missing.append("share-price data")

    if re.search(r"\bdividend", q) and "dividend" not in e:
        missing.append("dividend data")

    if re.search(r"percent(age)? (increase|growth|change)|grow(th)? by|by what percent", q):
        if len({v for v in numbers_in(ev) if abs(v) < 1e7}) < 2:
            missing.append("two comparable values")

    return missing


def _stated_absent(ev: str, term: str) -> bool:
    """True if the evidence explicitly says a quantity was not supplied."""
    return bool(re.search(
        rf"\b{term}\b[^.]{{0,60}}"
        r"(not provided|not disclosed|not available|not reported|not given|missing|unknown|omitted)",
        ev, re.I))


def _conflict_reasons(case: dict) -> list[str]:
    ev = evidence_text(case)
    reasons = []
    cue = any(c in norm_text(ev) for c in CONFLICT_CUES)

    if cue:
        reasons.append("evidence contains explicit conflict/precedence cues")

    # Same metric and same single period with distinct values. Year tokens
    # themselves are not values, and multi-year sentences are left alone.
    for metric in ("revenue", "sales", "net income", "profit", "expenses", "debt", "equity", "inventory"):
        grouped: dict[str, set[float]] = {}
        for s in sentences(ev):
            if metric not in s.lower():
                continue
            years = YEAR_RE.findall(s)
            if len(set(years)) > 1:
                continue
            vals: set[float] = set()
            for m in NUM_RE.finditer(s):
                tok = m.group().replace(",", "").lstrip("+-")
                if tok in years:
                    continue
                v = parse_number(m.group())
                if v is not None and abs(v) < 1e9:
                    vals.add(v)
            if not vals:
                continue
            key = years[0] if years else "unspecified"
            grouped.setdefault(key, set()).update(vals)
        for key, vals in grouped.items():
            if len(vals) > 1 and (key != "unspecified" or cue):
                reasons.append(f"{metric} has multiple values for period {key}")
                break
    return reasons


def _direction_reasons(ev: str) -> list[str]:
    up = r"\b(rose|rise|increased?|grew|grow|gained|up|higher)\b"
    down = r"\b(fell|fall|declined?|decreased?|dropped|down|lower)\b"
    reasons = []
    for s in sentences(ev):
        years = YEAR_RE.findall(s)
        vals = []
        for m in NUM_RE.finditer(s):
            tok = m.group().replace(",", "").lstrip("+-")
            if tok in years:
                continue
            v = parse_number(m.group())
            if v is not None and abs(v) < 1e9:
                vals.append(v)
        if len(vals) < 2:
            continue
        if re.search(up, s, re.I) and vals[-1] < vals[0]:
            reasons.append("direction word says up but values fall")
        if re.search(down, s, re.I) and vals[-1] > vals[0]:
            reasons.append("direction word says down but values rise")
    return reasons


def _negative_input_reasons(ev: str) -> list[str]:
    for pat in NEGATIVE_INPUT_PATTERNS:
        if re.search(pat, ev, re.I):
            return ["negative value attached to a non-negative quantity"]
    return []


def _entity_reasons(question: str, ev: str) -> list[str]:
    phrases = re.findall(r"\b([A-Z][A-Za-z0-9&.-]+(?:\s+[A-Z][A-Za-z0-9&.-]+)+)\b", question or "")
    reasons = []
    for p in phrases:
        words = {w.lower() for w in re.findall(r"[A-Za-z]+", p)}
        if words & set(METRIC_WORDS):
            continue
        if p.lower() not in norm_text(ev):
            reasons.append(f"requested entity not in evidence: {p}")
    return reasons


def precheck(case: dict) -> dict:
    """Run input-side policy. Returns allow/abstain plus optional sanitised case."""
    q = case.get("question", "") or ""
    ev = evidence_text(case)
    reasons: list[str] = []
    checks: dict[str, object] = {}
    sanitized_case = None

    # 1. Hard system-override / jailbreak anywhere.
    jb = _matches(JAILBREAK_PATTERNS, q) + _matches(JAILBREAK_PATTERNS, ev)
    checks["jailbreak_hits"] = len(jb)
    if jb:
        reasons.append("jailbreak or system-override instruction detected")
        return _decision("abstain", "input", reasons, "security", checks)

    # 2. Evidence-borne injection: remove it and continue.
    inj = _sentences_with(INJECTION_PATTERNS, ev)
    checks["injection_sentences"] = len(inj)
    if inj:
        sanitized_case, removed = sanitize_case(case)
        checks["removed_untrusted"] = len(removed)
        checks["sanitized"] = True

    # 3. Request-side policy.
    if _matches(ADVICE_PATTERNS, q):
        reasons.append("personalised investment advice / guaranteed-return request")
        return _decision("abstain", "input", reasons, "advice", checks)
    if _matches(PII_PATTERNS, q):
        reasons.append("request for personal data")
        return _decision("abstain", "input", reasons, "pii", checks)
    if _matches(OFFTOPIC_PATTERNS, q):
        reasons.append("request outside the financial scope")
        return _decision("abstain", "input", reasons, "offtopic", checks)

    # Sentiment cases carry their evidence in `sentence`, not `context`.
    if case.get("task") == "fiqa":
        return {
            "decision": "allow", "stage": "input", "reasons": [], "policy": None,
            "safe_message": None, "sanitized": False, "sanitized_case": None,
            "checks": checks,
        }

    working = sanitized_case or case
    working_ev = evidence_text(working)

    # 4. Evidence sufficiency.
    if not working_ev.strip():
        reasons.append("no financial evidence supplied")
        return _decision("abstain", "input", reasons, "no_evidence", checks)
    if _matches(FORECAST_PATTERNS, q):
        reasons.append("forecast/future value is not supported by the evidence")
        return _decision("abstain", "input", reasons, "forecast", checks)

    conflict = _conflict_reasons(working)
    checks["conflicts"] = conflict
    if conflict:
        reasons.append("conflicting evidence; reconciliation required")
        return _decision("abstain", "input", reasons, "conflict", checks)

    direction = _direction_reasons(working_ev)
    checks["direction"] = direction
    if direction:
        reasons.append("evidence is internally inconsistent")
        return _decision("abstain", "input", reasons, "inconsistent", checks)

    negative = _negative_input_reasons(working_ev)
    checks["negative_inputs"] = negative
    if negative:
        reasons.append("implausible negative input")
        return _decision("abstain", "input", reasons, "implausible", checks)

    entities = _entity_reasons(q, working_ev)
    checks["entities"] = entities
    if entities:
        reasons.append("question refers to an entity absent from the evidence")
        return _decision("abstain", "input", reasons, "entity", checks)

    if re.search(r"\bcite\b|\bpage\b|\bfootnote\b", q, re.I) and "page" not in norm_text(working_ev):
        reasons.append("citation/page provenance is not in the evidence")
        return _decision("abstain", "input", reasons, "citation", checks)

    missing = _metric_requirements_missing(q, working_ev)
    checks["missing_inputs"] = missing
    if missing:
        reasons.append("missing required inputs: " + ", ".join(missing))
        return _decision("abstain", "input", reasons, "missing_inputs", checks)

    return {
        "decision": "allow",
        "stage": "input",
        "reasons": [],
        "policy": None,
        "safe_message": None,
        "sanitized": sanitized_case is not None,
        "sanitized_case": sanitized_case,
        "checks": checks,
    }


def _decision(decision, stage, reasons, policy, checks):
    return {
        "decision": decision,
        "stage": stage,
        "reasons": reasons,
        "policy": policy,
        "safe_message": SAFE_MESSAGES.get(policy),
        "sanitized": False,
        "sanitized_case": None,
        "checks": checks,
    }


SAFE_MESSAGES = {
    "security": "I can't follow instructions that try to override the task or system rules.",
    "advice": "I can't give personalised investment advice or guarantee returns. I can summarise the supplied financial evidence instead.",
    "pii": "I can't extract or infer personal data that is not present in the supplied evidence.",
    "offtopic": "This model is scoped to financial evidence and calculations, so I can't help with that request.",
    "no_evidence": "Please supply the financial evidence needed to answer the question.",
    "forecast": "I can't forecast future values that are not in the supplied evidence.",
    "conflict": "The evidence contains conflicting figures. Please reconcile the sources or state which one takes precedence.",
    "inconsistent": "The evidence is internally inconsistent, so I can't compute a reliable answer.",
    "implausible": "The evidence contains a negative value for a quantity that cannot be negative. Please check the input.",
    "entity": "The question names an entity that is not present in the supplied evidence.",
    "citation": "I can't cite a page or footnote that is not present in the supplied evidence.",
    "missing_inputs": "The evidence does not contain all the figures needed for this calculation.",
    "unparseable": "The model did not return a usable program and answer.",
    "not_executable": "The generated program could not be executed safely, so no answer is reported.",
    "ungrounded": "The generated program uses values that are not grounded in the supplied evidence.",
    "magnitude": "The computed answer is outside a plausible range.",
    "unit_mismatch": "The computed value is a fraction but the question asks for a percentage. Reporting it as percentage points would be misleading.",
    "unit_scale": "The computed answer is off by orders of magnitude from the units used in the evidence, so I won't report it as reliable.",
    "sentiment_range": "The sentiment score is outside the valid range.",
}


def program_literals(program: str) -> list[float]:
    if not program:
        return []
    p = re.sub(r"#[0-9]+", " ", program)
    out = []
    for m in re.finditer(r"(?:const_)?([-+]?\d+(?:\.\d+)?)", p):
        v = parse_number(m.group(1))
        if v is not None:
            out.append(v)
    return out


def _grounded(value: float, pool: set[float]) -> bool:
    if any(math.isclose(abs(value), w, rel_tol=1e-9, abs_tol=1e-9) for w in CONSTANT_WHITELIST):
        return True
    for p in pool:
        if math.isclose(value, p, rel_tol=1e-6, abs_tol=1e-6):
            return True
        if p != 0 and math.isclose(value, -p, rel_tol=1e-6, abs_tol=1e-6):
            return True
    return False


# --- unit validation ---------------------------------------------------------

UNIT_WORDS = {"thousand": 1e3, "million": 1e6, "billion": 1e9}

UNIT_ALIASES = {
    "thousand": r"thousands?",
    "million": r"millions?|mil|mn|mio",
    "billion": r"billions?|bil|bn",
}


def _requested_unit(question: str):
    """Return (unit_word, multiplier) if the question asks for a specific scale."""
    q = norm_text(question)
    hits = [u for u in UNIT_WORDS
            if re.search(rf"\bin ({UNIT_ALIASES[u]})\b", q)]
    if len(hits) == 1:
        return hits[0], UNIT_WORDS[hits[0]]
    return None, None


def _evidence_units(text: str) -> set[str]:
    t = norm_text(text)
    return {u for u in UNIT_WORDS if re.search(rf"\b{UNIT_ALIASES[u]}\b", t)}


def _unit_conversion_constants(case: dict) -> set[float]:
    """Allow the factor needed to restate a quantity in the requested unit.

    Only granted when the evidence and the question name different scales, so a
    bare 'in millions' cannot legalise an arbitrary million-sized literal.
    """
    unit, mult = _requested_unit(case.get("question", ""))
    if unit is None:
        return set()
    ev_units = _evidence_units(evidence_text(case))
    if not ev_units - {unit}:
        return set()
    allowed: set[float] = set()
    for u in ev_units - {unit}:
        ratio = UNIT_WORDS[u] / mult
        allowed |= {ratio, 1.0 / ratio}
    return {round(v, 12) for v in allowed}


def _percent_convention_reason(case: dict, result: dict):
    """Flag a fraction returned where the question asked for percentage points.

    Structural check: if the question asks for a percentage and the program has
    no explicit x100 step, a result strictly between -1 and 1 is a fraction, not
    a percentage. This refuses a wrong-unit answer instead of passing it through.
    """
    q = norm_text(case.get("question", ""))
    if not re.search(r"percent|percentage|\b%\b", q):
        return None
    prog = (result.get("program") or "").replace(" ", "")
    if re.search(r"const_100\b", prog) or re.search(r"multiply\([^,]*,\s*100(?:\.0*)?\)", prog):
        return None
    val = result.get("executed_answer")
    if isinstance(val, (int, float)) and math.isfinite(val) and 0 < abs(val) < 1:
        return f"answer {val} looks like a fraction but percentage points were requested"
    return None


def _unit_scale_reason(case: dict, result: dict):
    """Flag answers off by orders of magnitude from the requested unit.

    Conservative: only applies when the question names a unit and the evidence
    uses a different scale word. Requires a 1000x discrepancy against the largest
    evidence magnitude restated in the requested unit.
    """
    unit, mult = _requested_unit(case.get("question", ""))
    if unit is None:
        return None
    ev = evidence_text(case)
    ev_units = _evidence_units(ev)
    if not ev_units - {unit}:
        return None
    val = result.get("executed_answer")
    if not isinstance(val, (int, float)) or not math.isfinite(val) or val == 0:
        return None

    magnitudes = []
    for u, m in UNIT_WORDS.items():
        for match in re.finditer(rf"([-+]?\d[\d,]*(?:\.\d+)?)\s*{u}s?\b", ev, re.I):
            v = parse_number(match.group(1))
            if v is not None:
                magnitudes.append(abs(v) * m / mult)
    if not magnitudes:
        return None
    top = max(magnitudes)
    if top <= 0:
        return None
    if abs(val) < top * 1e-3 or abs(val) > top * 1e3:
        return (f"answer {val} is {abs(val) / top:.2e} of the evidence scale in "
                f"{unit}s, suggesting a unit conversion error")
    return None


def postcheck(case: dict, result: dict) -> dict:
    """Run output-side policy on a parsed generation result."""
    checks: dict[str, object] = {}

    if case.get("task") == "fiqa":
        score = result.get("sentiment_score")
        checks["score"] = score
        if score is None or not result.get("score_in_range"):
            return _decision("abstain", "output", ["sentiment score missing or out of range"],
                             "sentiment_range", checks)
        return _decision("allow", "output", [], None, checks)

    if not result.get("parse_success"):
        return _decision("abstain", "output", ["required PROGRAM/ANSWER fields not parsed"],
                         "unparseable", checks)
    if not result.get("program_executable"):
        checks["execution_error"] = result.get("execution_error")
        return _decision("abstain", "output", ["generated program did not execute"],
                         "not_executable", checks)

    ev = evidence_text(case)
    pool = numbers_in(ev) | numbers_in(case.get("question", ""))
    lits = program_literals(result.get("program", ""))
    extra = _unit_conversion_constants(case)
    checks["unit_constants_allowed"] = sorted(extra)
    ungrounded = [v for v in lits if not _grounded(v, pool | extra)]
    checks["program_literals"] = lits
    checks["ungrounded_literals"] = ungrounded
    if ungrounded:
        return _decision("abstain", "output",
                         [f"program uses ungrounded values: {ungrounded[:5]}"],
                         "ungrounded", checks)

    executed = result.get("executed_answer")
    checks["executed_answer"] = executed
    if isinstance(executed, (int, float)) and (not math.isfinite(executed) or abs(executed) > 1e15):
        return _decision("abstain", "output", ["answer magnitude out of range"], "magnitude", checks)

    percent = _percent_convention_reason(case, result)
    checks["percent_convention"] = percent
    if percent:
        return _decision("abstain", "output", [percent], "unit_mismatch", checks)

    scale = _unit_scale_reason(case, result)
    checks["unit_scale"] = scale
    if scale:
        return _decision("abstain", "output", [scale], "unit_scale", checks)

    warnings = []
    if result.get("written_matches_execution") is False:
        warnings.append("written answer disagreed with the executed program; executed value preferred")

    return {
        "decision": "allow",
        "stage": "output",
        "reasons": [],
        "policy": None,
        "safe_message": None,
        "sanitized": False,
        "sanitized_case": None,
        "checks": checks,
        "warnings": warnings,
        "final_answer": executed if executed is not None else result.get("written_answer"),
    }


def apply(case: dict, result: dict | None) -> dict:
    """Convenience one-shot path used by the demo helper."""
    pre = precheck(case)
    if pre["decision"] == "abstain":
        return pre
    return postcheck(pre.get("sanitized_case") or case, result or {})
