"""
Pure forecasting logic: prompt text, parsing, clipping and aggregation.

Nothing here calls the network, so everything is unit-tested offline.
Question objects are duck-typed (anything with the attributes used below), so
both forecasting_tools question models and simple test doubles work.
"""
from __future__ import annotations

import logging
import re
import statistics
import textwrap
from datetime import date
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)

BINARY_MIN = 0.01
BINARY_MAX = 0.99
MC_MIN_OPTION_PROBABILITY = 0.005


def conditional_disclaimer(question: Any) -> str:
    if getattr(question, "conditional_type", None) not in ("yes", "no"):
        return ""
    return (
        "This is a conditional question with a parent and a child. Forecast ONLY the CHILD "
        "question, given the stated resolution of the parent question. Never re-forecast the "
        "parent question, but reason carefully about how the parent's resolution changes the child.\n"
    )


def build_binary_prompt(question: Any, research: str, today: date) -> str:
    """Base rate -> evidence -> calibration, ending in 'Probability: ZZ%'."""
    research_block = research.strip() if research and research.strip() else (
        "No news search was available for this run. Rely on the question text and your own knowledge, "
        "and remember that your knowledge may be out of date."
    )
    template = """
        You are an experienced, well-calibrated superforecaster. Your forecasts are scored with a proper
        log scoring rule against other forecasters, so both overconfidence and timidity are costly.

        Question:
        {question_text}

        Background:
        {background}

        Resolution criteria (these have NOT yet been satisfied):
        {criteria}

        Fine print:
        {fine_print}

        News research:
        {research_block}

        Today is {today}.
        {conditional}
        Work through the following steps in order and write each one out.

        Step 1 - Pin down the question.
        State exactly what must happen for YES, the deadline, and how much time is left.

        Step 2 - Base rate (outside view).
        Name one to three reference classes of similar situations. For each, estimate how often the
        YES-type outcome happened within a comparable time window, and say how well the class fits.
        Combine them into a single starting probability. Also state the status-quo outcome (what happens
        if nothing changes), since the world usually changes slowly.

        Step 3 - Evidence (inside view).
        List the most important specific pieces of evidence from the research and background. For each,
        say which direction it pushes (toward YES or toward NO), how strong it is (weak / moderate / strong),
        and how reliable the source is. Then adjust step by step from the base rate, saying how many
        percentage points each adjustment moves you.

        Step 4 - Calibration check.
        - Scope: would your number change sensibly if the deadline were half as long or twice as long?
        - Overreaction: are you moving too far on a single headline or on vivid but weak evidence?
        - Resolution details: could a technicality, an ambiguity, or the exact wording flip the outcome?
        - Extremes: only go below 3% or above 97% if the outcome is close to locked in by facts you are
          confident about. Otherwise stay inside that range.
        State your final probability after this check.

        The last line of your answer must be exactly in this format, with nothing after it:
        Probability: ZZ%
        """
    return textwrap.dedent(template).strip().format(
        question_text=question.question_text,
        background=question.background_info or "(none given)",
        criteria=question.resolution_criteria or "(none given)",
        fine_print=question.fine_print or "(none given)",
        research_block=research_block,
        today=today.isoformat(),
        conditional=conditional_disclaimer(question),
    ) + "\n"


_PROB_PATTERN = re.compile(
    r"probability\s*[:=]\s*\**\s*([0-9]+(?:\.[0-9]+)?)\s*(%?)", re.IGNORECASE
)


def parse_binary_probability(text: str) -> float | None:
    """
    Return the LAST 'Probability: ZZ%' in the text as a decimal (not clipped),
    or None if no such line exists. A bare number without '%' is read as a
    percentage if > 1, otherwise as a decimal.
    """
    if not text:
        return None
    matches = _PROB_PATTERN.findall(text)
    if not matches:
        return None
    number_str, percent_sign = matches[-1]
    value = float(number_str)
    if percent_sign or value > 1:
        value = value / 100.0
    if not 0.0 <= value <= 1.0:
        return None
    return value


def clip_binary(p: float, low: float = BINARY_MIN, high: float = BINARY_MAX) -> float:
    if p != p:  # NaN
        raise ValueError("probability is NaN")
    return max(low, min(high, float(p)))


def aggregate_binary(predictions: Sequence[float]) -> float:
    """Median of the samples, then clipped to [1%, 99%]."""
    if not predictions:
        raise ValueError("Cannot aggregate empty list of predictions")
    return clip_binary(float(statistics.median(predictions)))


def aggregate_multiple_choice(
    samples: Sequence[dict[str, float]],
    floor: float = MC_MIN_OPTION_PROBABILITY,
) -> dict[str, float]:
    """
    Per-option median across samples, each option floored at `floor`, then
    renormalized to sum to exactly 1. Option order follows the first sample.
    """
    if not samples:
        raise ValueError("Cannot aggregate empty list of predictions")
    names = list(samples[0].keys())
    for s in samples:
        if set(s.keys()) != set(names):
            raise ValueError(f"Option names differ between samples: {sorted(s)} vs {sorted(names)}")
    medians = {n: max(floor, float(statistics.median(s[n] for s in samples))) for n in names}
    total = sum(medians.values())
    normalized = {n: v / total for n, v in medians.items()}
    # Fix floating error on the largest option so the sum is exactly 1.
    largest = max(normalized, key=normalized.get)
    normalized[largest] += 1.0 - sum(normalized.values())
    return normalized


def select_questions(
    questions: Iterable[Any],
    skip_previously_forecasted: bool,
    remaining_budget: int | None,
) -> tuple[list[Any], int | None]:
    """
    Drop already-forecasted questions (if asked) and cap how many are sent.
    Returns (selected_questions, new_remaining_budget). remaining_budget=None
    means unlimited.
    """
    selected = [
        q for q in questions
        if not (skip_previously_forecasted and getattr(q, "already_forecasted", False))
    ]
    if remaining_budget is None:
        return selected, None
    if len(selected) > remaining_budget:
        logger.warning(
            f"Question cap reached: forecasting {remaining_budget} of {len(selected)} new questions this run"
        )
        selected = selected[:remaining_budget]
    return selected, remaining_budget - len(selected)
