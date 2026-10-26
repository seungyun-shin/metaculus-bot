"""
End-to-end runs of the bot class with fake LLMs and the network blocked.
publish is always False, so nothing is ever sent to Metaculus.
"""
import asyncio
import json
from datetime import datetime, timezone

import pytest

import bot_config as bc
from forecasting_tools import (
    BinaryQuestion,
    DateQuestion,
    GeneralLlm,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.ai_models.resource_managers.hard_limit_manager import (
    HardLimitExceededError,
)
from budgeted_llm import BudgetedLlm


class ScriptedLlm(GeneralLlm):
    """GeneralLlm whose invoke() returns scripted answers instead of calling an API."""

    def __init__(self, answers, model="openrouter/anthropic/claude-sonnet-5.5"):
        super().__init__(model=model)
        self.answers = list(answers)
        self.prompts: list[str] = []

    async def invoke(self, prompt, system_prompt=None):
        self.prompts.append(prompt if isinstance(prompt, str) else str(prompt))
        answer = self.answers[(len(self.prompts) - 1) % len(self.answers)]
        return answer(prompt) if callable(answer) else answer


def settings(**overrides):
    base = bc.load_config({"METACULUS_TOKEN": "t", "OPENROUTER_API_KEY": "or", "BOT_PUBLISH": "false"})
    return bc.BotConfig(**{**base.__dict__, **overrides})


def make_bot(forecaster, parser=None, **setting_overrides):
    import main

    cfg = settings(**setting_overrides)
    llms = {
        "default": forecaster,
        "summarizer": parser or ScriptedLlm(["unused"]),
        "researcher": bc.NO_RESEARCH,
        "parser": parser or ScriptedLlm(["unused"]),
    }
    bot = main.CalibratedBot(bot_settings=cfg, llms=llms, skip_previously_forecasted_questions=True)
    assert bot.publish_reports_to_metaculus is False
    return bot


def binary_question(i=1, forecasted=False):
    return BinaryQuestion(
        question_text=f"Will event {i} happen before 2027?",
        background_info="Background.",
        resolution_criteria="Resolves YES if event happens.",
        fine_print="",
        page_url=f"https://www.metaculus.com/questions/{i}/",
        id_of_question=i,
        id_of_post=i,
        already_forecasted=forecasted,
    )


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- binary

def test_binary_five_samples_median_and_clip():
    forecaster = ScriptedLlm(
        ["...Probability: 10%", "...Probability: 20%", "...Probability: 99.9%", "...Probability: 30%", "...Probability: 25%"]
    )
    bot = make_bot(forecaster)
    [report] = run(bot.forecast_questions([binary_question()]))
    assert len(forecaster.prompts) == 5  # 5 samples
    assert report.prediction == pytest.approx(0.25)  # median
    assert "Base rate (outside view)" in forecaster.prompts[0]
    assert "No news search was available" in forecaster.prompts[0]


def test_binary_extremes_are_clipped():
    bot = make_bot(ScriptedLlm(["Probability: 0%"]))
    [report] = run(bot.forecast_questions([binary_question()]))
    assert report.prediction == pytest.approx(0.01)
    bot = make_bot(ScriptedLlm(["Probability: 100%"]))
    [report] = run(bot.forecast_questions([binary_question()]))
    assert report.prediction == pytest.approx(0.99)


def test_binary_falls_back_to_parser_llm_when_regex_fails():
    forecaster = ScriptedLlm(["I lean towards yes, roughly two in five."])
    parser = ScriptedLlm([json.dumps({"prediction_in_decimal": 0.4})], model="openrouter/openai/gpt-4o-mini")
    bot = make_bot(forecaster, parser=parser, predictions_per_question=1)
    [report] = run(bot.forecast_questions([binary_question()]))
    assert report.prediction == pytest.approx(0.4)
    assert len(parser.prompts) >= 1


# ---------------------------------------------------------------- other question types still work

def test_multiple_choice_median_aggregation():
    q = MultipleChoiceQuestion(
        question_text="Which party wins?",
        options=["A", "B", "C"],
        page_url="https://www.metaculus.com/questions/2/",
        id_of_question=2,
        id_of_post=2,
    )
    sample_probs = [(0.6, 0.3, 0.1), (0.5, 0.4, 0.1), (0.9, 0.05, 0.05), (0.7, 0.2, 0.1), (0.6, 0.3, 0.1)]
    forecaster = ScriptedLlm([f"A: {a*100}%\nB: {b*100}%\nC: {c*100}%" for a, b, c in sample_probs])

    def parse(prompt):
        text = prompt if isinstance(prompt, str) else str(prompt)
        for a, b, c in sample_probs:
            if f"A: {a*100}%" in text:
                return json.dumps({"predicted_options": [
                    {"option_name": "A", "probability": a},
                    {"option_name": "B", "probability": b},
                    {"option_name": "C", "probability": c},
                ]})
        raise AssertionError("unexpected parser prompt")

    parser = ScriptedLlm([parse], model="openrouter/openai/gpt-4o-mini")
    bot = make_bot(forecaster, parser=parser)
    [report] = run(bot.forecast_questions([q]))
    probs = {o.option_name: o.probability for o in report.prediction.predicted_options}
    assert sum(probs.values()) == pytest.approx(1.0)
    assert probs["A"] == pytest.approx(0.6)  # median of A
    assert probs["B"] == pytest.approx(0.3)
    assert probs["C"] == pytest.approx(0.1)


def _percentiles_json(values):
    return json.dumps([
        {"percentile": p, "value": v} for p, v in zip([0.1, 0.2, 0.4, 0.6, 0.8, 0.9], values)
    ])


def test_numeric_question_runs():
    q = NumericQuestion(
        question_text="What will X be?",
        page_url="https://www.metaculus.com/questions/3/",
        id_of_question=3,
        id_of_post=3,
        lower_bound=0.0,
        upper_bound=100.0,
        open_lower_bound=False,
        open_upper_bound=True,
        unit_of_measure="units",
    )
    forecaster = ScriptedLlm(["Percentile 10: 20\nPercentile 90: 60"])
    parser = ScriptedLlm([_percentiles_json([20, 25, 35, 42, 52, 60])], model="openrouter/openai/gpt-4o-mini")
    bot = make_bot(forecaster, parser=parser)
    [report] = run(bot.forecast_questions([q]))
    cdf = report.prediction.get_cdf()
    assert len(cdf) == q.cdf_size
    assert 0.3 < report.prediction.get_cdf()[len(cdf) // 2].percentile < 0.9


def test_date_question_runs():
    lo = datetime(2026, 1, 1, tzinfo=timezone.utc)
    hi = datetime(2030, 1, 1, tzinfo=timezone.utc)
    q = DateQuestion(
        question_text="When will Y happen?",
        page_url="https://www.metaculus.com/questions/4/",
        id_of_question=4,
        id_of_post=4,
        lower_bound=lo,
        upper_bound=hi,
        open_lower_bound=False,
        open_upper_bound=True,
    )
    dates = ["2026-06-01", "2026-09-01", "2027-01-01", "2027-06-01", "2028-01-01", "2028-09-01"]
    forecaster = ScriptedLlm(["Percentile 10: 2026-06-01 ... Percentile 90: 2028-09-01"])
    parser = ScriptedLlm(
        [json.dumps([{"percentile": p, "value": d + "T00:00:00Z"} for p, d in zip([0.1, 0.2, 0.4, 0.6, 0.8, 0.9], dates)])],
        model="openrouter/openai/gpt-4o-mini",
    )
    bot = make_bot(forecaster, parser=parser)
    [report] = run(bot.forecast_questions([q]))
    assert len(report.prediction.get_cdf()) == q.cdf_size


# ---------------------------------------------------------------- limits

def test_question_cap_is_shared_across_calls():
    bot = make_bot(ScriptedLlm(["Probability: 40%"]), max_questions_per_run=2, predictions_per_question=1)
    qs = [binary_question(1), binary_question(2, forecasted=True), binary_question(3), binary_question(4)]
    first = run(bot.forecast_questions(qs))
    assert [r.question.id_of_question for r in first] == [1, 3]  # 2 skipped (already forecasted), 4 over cap
    second = run(bot.forecast_questions([binary_question(5)]))
    assert second == []


def test_cost_cap_per_question_stops_further_llm_calls():
    # Real BudgetedLlm (the class build_llms uses) + litellm mock_response: no
    # network, but litellm's cost callbacks run. The forecaster answer has no "Probability:" line, so the
    # bot needs a second (parser) call; by then the first call's cost has
    # exhausted the $0.0001 cap and the pre-call check must refuse it.
    bc.register_prices(["openrouter/anthropic/claude-sonnet-5.5", "openrouter/openai/gpt-4o-mini"])
    forecaster = BudgetedLlm(
        model="openrouter/anthropic/claude-sonnet-5.5",
        mock_response="no number here " * 50,
        api_key="fake",
    )
    parser = BudgetedLlm(model="openrouter/openai/gpt-4o-mini", mock_response='{"prediction_in_decimal": 0.5}', api_key="fake")
    bot = make_bot(forecaster, parser=parser, predictions_per_question=1, max_cost_per_question_usd=0.0001)
    [result] = run(bot.forecast_questions([binary_question()], return_exceptions=True))
    assert isinstance(result, BaseException)
    assert "HardLimitExceededError" in repr(result) or isinstance(result, HardLimitExceededError)


def test_cost_cap_allows_cheap_question():
    bc.register_prices(["openrouter/anthropic/claude-sonnet-5.5"])
    forecaster = BudgetedLlm(
        model="openrouter/anthropic/claude-sonnet-5.5", mock_response="Probability: 33%", api_key="fake"
    )
    bot = make_bot(forecaster, predictions_per_question=5, max_cost_per_question_usd=1.0)
    [report] = run(bot.forecast_questions([binary_question()]))
    assert report.prediction == pytest.approx(0.33)
    assert 0 < report.price_estimate < 1.0  # cost was tracked


def test_research_failure_degrades_to_no_research(monkeypatch):
    import main

    class Boom:
        async def call_preconfigured_version(self, *a, **k):
            raise RuntimeError("AskNews quota exceeded")

    monkeypatch.setattr(main, "AskNewsSearcher", lambda: Boom())
    forecaster = ScriptedLlm(["Probability: 40%"])
    bot = make_bot(forecaster, predictions_per_question=1)
    bot._llms["researcher"] = bc.ASKNEWS_RESEARCHER
    [report] = run(bot.forecast_questions([binary_question()]))
    assert report.prediction == pytest.approx(0.4)
    assert "No news search was available" in forecaster.prompts[0]


def test_main_exits_1_without_token(clean_env, capsys):
    import main

    assert main.main(["--mode", "test_questions", "--dry-run"]) == 1
    assert "METACULUS_TOKEN" in capsys.readouterr().out


def test_main_exits_1_on_bad_config(clean_env, capsys):
    import main

    clean_env.setenv("METACULUS_TOKEN", "t")
    clean_env.setenv("BOT_EFFORT", "extreme")
    assert main.main(["--mode", "test_questions", "--dry-run"]) == 1
    assert "Configuration error" in capsys.readouterr().out


def test_run_all_over_two_tournaments_with_run_cost_cap(monkeypatch):
    # Covers the tournament flow (FutureEval + MiniBench) with the Metaculus
    # client stubbed out, and the run-level cost cap across questions.
    import main

    bc.register_prices(["openrouter/anthropic/claude-sonnet-5.5"])
    forecaster = BudgetedLlm(
        model="openrouter/anthropic/claude-sonnet-5.5", mock_response="Probability: 60%", api_key="fake"
    )
    bot = make_bot(forecaster, predictions_per_question=1)
    served = {"fe": [binary_question(1)], "minibench": [binary_question(2), binary_question(3)]}
    monkeypatch.setattr(
        bot.metaculus_client, "get_all_open_questions_from_tournament", lambda tid: served[tid]
    )
    # Unlimited run budget: all three questions forecast.
    reports = run(main.run_all(bot, ["fe", "minibench"], max_run_cost=0))
    assert [r.prediction for r in reports] == [pytest.approx(0.6)] * 3

    # Tiny run budget: the first question spends it, later ones are refused.
    bot = make_bot(forecaster, predictions_per_question=1)
    monkeypatch.setattr(
        bot.metaculus_client, "get_all_open_questions_from_tournament", lambda tid: served[tid]
    )
    reports = run(main.run_all(bot, ["fe", "minibench"], max_run_cost=0.00001))
    ok = [r for r in reports if not isinstance(r, BaseException)]
    failed = [r for r in reports if isinstance(r, BaseException)]
    assert len(ok) == 1 and len(failed) == 2
    assert all("HardLimitExceededError" in repr(e) for e in failed)


def test_tournament_targets():
    import main
    from forecasting_tools import MetaculusClient

    client = MetaculusClient()
    assert main.tournament_targets("tournament", client) == [
        client.CURRENT_AI_COMPETITION_ID,
        client.CURRENT_MINIBENCH_ID,
    ]
    assert main.tournament_targets("test_questions", client) == ["bot-testing-area"]
    assert main.tournament_targets("minibench", client) == ["minibench"]


def test_cost_is_counted_once_per_call():
    # Regression test for the double counting described in budgeted_llm.py:
    # tracked cost must equal tokens x registered price, once per call.
    import litellm
    from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
        MonetaryCostManager,
    )

    model = "openrouter/anthropic/claude-sonnet-5.5"
    bc.register_prices([model])
    answer = "x " * 50
    prompt = "hello " * 100
    llm = BudgetedLlm(model=model, mock_response=answer, api_key="fake")

    async def reference_usage():
        resp = await litellm.acompletion(
            model=model, messages=[{"role": "user", "content": prompt}], mock_response=answer, api_key="fake"
        )
        return resp.usage.prompt_tokens, resp.usage.completion_tokens

    async def tracked(n):
        with MonetaryCostManager(0) as cm:
            for _ in range(n):
                await llm.invoke(prompt)
            await asyncio.sleep(0.3)  # let litellm's background callbacks finish
            return cm.current_usage

    p_tok, c_tok = run(reference_usage())
    expected_one = p_tok * 2e-06 + c_tok * 1e-05
    assert expected_one > 0
    assert run(tracked(1)) == pytest.approx(expected_one, rel=1e-6)
    assert run(tracked(4)) == pytest.approx(4 * expected_one, rel=1e-6)
