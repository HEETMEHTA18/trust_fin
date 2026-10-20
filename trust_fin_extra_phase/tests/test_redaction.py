"""Privacy gate: deterministic PII detection/redaction (roadmap §4, §6)."""
from runtime import redaction


def test_detects_ssn():
    hits = redaction.find_pii("His SSN is 123-45-6789 on file.")
    assert [h["type"] for h in hits] == ["ssn"]
    assert hits[0]["span"] == "123-45-6789"


def test_detects_card_iban_email():
    text = ("card 4111111111111111, iban GB82WEST12345698765432, "
            "reach me at jane.doe@example.com")
    types = {h["type"] for h in redaction.find_pii(text)}
    assert {"credit_card", "iban", "email"} <= types


def test_detects_account_number_phrasing():
    hits = redaction.find_pii("routing number 021000021 account number 9876543210")
    assert any(h["type"] == "account_number" for h in hits)


def test_clean_text_untouched():
    text = "Compute the EBITDA margin for FY2025 from the annual report."
    assert redaction.find_pii(text) == []
    assert redaction.redact(text) == text


def test_redact_replaces_all_spans():
    out = redaction.redact("SSN 123-45-6789 and account number 9876543210 end.")
    assert "123-45-6789" not in out
    assert "9876543210" not in out
    assert "[REDACTED:ssn]" in out
    assert "[REDACTED:account_number]" in out


def test_overlapping_spans_deduplicated():
    # account-number pattern overlapping its digits must not double-count
    hits = redaction.find_pii("account number 9876543210")
    spans = [(h["start"], h["end"]) for h in hits]
    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))
