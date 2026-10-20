"""redaction.py — deterministic PII detection and redaction (privacy gate).

Regex-only, stdlib-only, offline. Catches the PII classes named in the roadmap
and observed leaking in the Phase-4 FinTrust privacy run (docs/exp_4(d).md):
SSNs answered directly, account numbers, credentials. This is the *input* gate
of the runtime; Phase-6 safety tuning sits behind the same interface.
"""
from __future__ import annotations

import re

#: Ordered patterns; first match wins on overlapping spans.
PATTERNS: dict[str, re.Pattern] = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|"
                              r"3[47][0-9]{13})\b"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"),
    "account_number": re.compile(
        r"(?i)\b(?:account|routing|client|card|portfolio)\s*"
        r"(?:number|no\.?|#|id)?\s*[:#]?\s*\d{6,}\b"),
    "phone": re.compile(r"\+?\d{1,3}?[ (.-]?\d{3}[). -]?\d{3}[ .-]?\d{4}\b"),
}


def find_pii(text: str) -> list[dict]:
    """Return non-overlapping PII spans: [{type, start, end, masked}, ...]."""
    hits: list[dict] = []
    taken: list[tuple[int, int]] = []
    for pii_type, rx in PATTERNS.items():
        for m in rx.finditer(text):
            s, e = m.start(), m.end()
            if any(s < te and ts < e for ts, te in taken):
                continue
            taken.append((s, e))
            span = m.group(0)
            masked = span[:4] + "…" + span[-2:] if len(span) > 8 else "…"
            hits.append({"type": pii_type, "start": s, "end": e,
                         "masked": masked, "span": span})
    return sorted(hits, key=lambda h: h["start"])


def redact(text: str) -> str:
    """Replace every detected PII span with ``[REDACTED:<type>]``."""
    hits = find_pii(text)
    out, last = [], 0
    for h in hits:
        out.append(text[last:h["start"]])
        out.append(f"[REDACTED:{h['type']}]")
        last = h["end"]
    out.append(text[last:])
    return "".join(out)
