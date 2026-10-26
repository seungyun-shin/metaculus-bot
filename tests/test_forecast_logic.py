import math
from datetime import date
from types import SimpleNamespace

import pytest

import forecast_logic as fl
from forecasting_tools import BinaryQuestion


def make_binary(**overrides):
    fields = dict(
        question_text="Will Country X hold a national election before 2027-01-01?",
        background_info="Country X postponed elections twice.",
        resolution_criteria="Resolves YES if a national election is held before 2027-01-01.",
        fine_print="Referendums do not count.",
        page_url="https://www.metaculus.com/questions/1/",
    )
    fields.update(overrides)
    return BinaryQuestion(**fields)


# ---------------------------------------------------------------- prompt

def test_binary_prompt_contains_question_fields_and_date():
    q = make_binary()
    prompt = fl.build_binary_prompt(q, "AskNews: an election date was announced.", date(2026, 10, 26))
    for text in (
        q.question_text,
        q.background_info,
        q.resolution_criteria,
        q.fine_print,
        "AskNews: an election date was announced.",
        "Today is 2026-10-26.",
    ):
        assert text in prompt


def test_binary_prompt_orders_base_rate_evidence_calibration():
    prompt = fl.build_binary_prompt(make_binary(), "", date(2026, 10, 26))
    i_base = prompt.index("Base rate (outside view)")
    i_evidence = prompt.index("Evidence (inside view)")
    i_calib = prompt.index("Calibration check")
    assert i_base < i_evidence < i_calib
    assert prompt.rstrip().endswith("Probability: ZZ%")


def test_binary_prompt_is_dedented_and_has_no_unfilled_placeholders():
    prompt = fl.build_binary_prompt(make_binary(), "", date(2026, 10, 26))
    assert not prompt.startswith(" ")
    assert "\n        " not in prompt  # template indentation removed
    for placeholder in ("{question_text}", "{research_block}", "{today}", "{criteria}"):
        assert placeholder not in prompt


def test_binary_prompt_without_research_says_so():
    prompt = fl.build_binary_prompt(make_binary(), "   ", date(2026, 10, 26))
    assert "No news search was available" in prompt


def test_binary_prompt_handles_missing_optional_fields():
    q = make_binary(background_info=None, fine_print=None)
    prompt = fl.build_binary_prompt(q, "", date(2026, 10, 26))
    assert prompt.count("(none given)") == 2


def test_binary_prompt_survives_braces_in_question_text():
    q = make_binary(question_text="Will the {curly} index exceed 5?")
    prompt = fl.build_binary_prompt(q, "research with {braces}", date(2026, 10, 26))
    assert "Will the {curly} index exceed 5?" in prompt
    assert "research with {braces}" in prompt


def test_conditional_disclaimer_only_for_conditional_children():
    assert fl.conditional_disclaimer(SimpleNamespace(conditional_type=None)) == ""
    assert fl.conditional_disclaimer(SimpleNamespace(conditional_type="parent")) == ""
    assert "CHILD" in fl.conditional_disclaimer(SimpleNamespace(conditional_type="yes"))
    prompt = fl.build_binary_prompt(make_binary(conditional_type="no"), "", date(2026, 1, 1))
    assert "Forecast ONLY the CHILD" in prompt


# ---------------------------------------------------------------- parsing

@pytest.mark.parametrize(
    "text, expected",
    [
        ("...\nProbability: 37%", 0.37),
        ("Probability: 37 %", 0.37),
        ("probability: **12.5%**", 0.125),
        ("Probability = 4%", 0.04),
        ("Probability: 0.42", 0.42),  # bare decimal
        ("Probability: 65", 0.65),  # bare percent
        ("Starting probability: 20%.\nAdjusted...\nProbability: 31%", 0.31),  # last one wins
        ("Probability: 0%", 0.0),
        ("Probability: 100%", 1.0),
    ],
)
def test_parse_binary_probability(text, expected):
    assert fl.parse_binary_probability(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", "I think it's likely.", "Probability: 150%", "Probability: unknown"])
def test_parse_binary_probability_returns_none_when_unparseable(text):
    assert fl.parse_binary_probability(text) is None


# ---------------------------------------------------------------- clipping

@pytest.mark.parametrize(
    "raw, expected",
    [(0.0, 0.01), (0.004, 0.01), (0.01, 0.01), (0.5, 0.5), (0.99, 0.99), (0.999, 0.99), (1.0, 0.99)],
)
def test_clip_binary(raw, expected):
    assert fl.clip_binary(raw) == pytest.approx(expected)


def test_clip_binary_rejects_nan():
    with pytest.raises(ValueError):
        fl.clip_binary(math.nan)


# ---------------------------------------------------------------- aggregation

def test_aggregate_binary_is_median_of_samples():
    assert fl.aggregate_binary([0.10, 0.20, 0.90, 0.30, 0.25]) == pytest.approx(0.25)


def test_aggregate_binary_even_count_and_clipping():
    assert fl.aggregate_binary([0.2, 0.4]) == pytest.approx(0.3)
    assert fl.aggregate_binary([0.999] * 5) == pytest.approx(0.99)
    assert fl.aggregate_binary([0.0, 0.0, 0.5]) == pytest.approx(0.01)


def test_aggregate_binary_empty_raises():
    with pytest.raises(ValueError):
        fl.aggregate_binary([])


def test_aggregate_multiple_choice_median_floor_and_normalisation():
    samples = [
        {"A": 0.6, "B": 0.4, "C": 0.0},
        {"A": 0.5, "B": 0.5, "C": 0.0},
        {"A": 0.9, "B": 0.1, "C": 0.0},
    ]
    out = fl.aggregate_multiple_choice(samples)
    assert list(out) == ["A", "B", "C"]
    assert sum(out.values()) == pytest.approx(1.0, abs=1e-12)
    # medians 0.6 / 0.4 / floor 0.005, renormalised by 1.005
    assert out["A"] == pytest.approx(0.6 / 1.005)
    assert out["B"] == pytest.approx(0.4 / 1.005)
    assert out["C"] == pytest.approx(0.005 / 1.005)


def test_aggregate_multiple_choice_rejects_mismatched_options():
    with pytest.raises(ValueError):
        fl.aggregate_multiple_choice([{"A": 1.0}, {"B": 1.0}])


# ---------------------------------------------------------------- question selection

def _q(i, forecasted=False):
    return SimpleNamespace(id=i, already_forecasted=forecasted)


def test_select_questions_skips_forecasted_and_caps():
    qs = [_q(1), _q(2, True), _q(3), _q(4), _q(5)]
    selected, left = fl.select_questions(qs, skip_previously_forecasted=True, remaining_budget=2)
    assert [q.id for q in selected] == [1, 3]
    assert left == 0
    selected, left = fl.select_questions(qs, skip_previously_forecasted=True, remaining_budget=left)
    assert selected == [] and left == 0


def test_select_questions_unlimited_and_no_skip():
    qs = [_q(1), _q(2, True)]
    selected, left = fl.select_questions(qs, skip_previously_forecasted=False, remaining_budget=None)
    assert [q.id for q in selected] == [1, 2]
    assert left is None
