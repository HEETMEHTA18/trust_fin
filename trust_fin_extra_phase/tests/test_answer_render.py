"""Roadmap §4 / Phase-4 finding R1: the displayed number comes from the
executor-verified result, and printed-vs-verified drift is detectable."""
import pytest

from runtime import answer_render


def test_format_number_deterministic():
    assert answer_render.format_number(150) == "150"
    assert answer_render.format_number(150.0) == "150"
    assert answer_render.format_number(0.298780487804878) == "0.298780487805"
    assert answer_render.format_number(-4200000) == "-4200000"


def test_format_number_rejects_bool_and_nan():
    with pytest.raises(TypeError):
        answer_render.format_number(True)
    with pytest.raises(ValueError):
        answer_render.format_number(float("nan"))


def test_render_verified_uses_verified_result():
    calc = {"verified_result": 95000}
    assert answer_render.render_verified(calc) == "95000"
    assert answer_render.render_verified({"verified_result": 150.5}) == "150.5"
    with pytest.raises(ValueError):
        answer_render.render_verified({"verified_result": None})


def test_scientific_notation_variants_match_verified_value():
    # the "31% channel": correct value, differently printed string
    assert answer_render.printed_matches_verified("1.5e2", 150)
    assert answer_render.printed_matches_verified("150.0", 150)
    assert answer_render.printed_matches_verified("$1,230.00", 1230)
    assert answer_render.printed_matches_verified("14.49115044%", 14.49115044)


def test_real_mismatch_detected():
    assert not answer_render.printed_matches_verified("1.51e2", 150)
    assert not answer_render.printed_matches_verified("2450000", 0.2987804878)
    assert not answer_render.printed_matches_verified("N/A", 150)   # unparseable
    assert not answer_render.printed_matches_verified(None, 150)
