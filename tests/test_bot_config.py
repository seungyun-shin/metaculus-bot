import pytest

import bot_config as bc


def env(**kw):
    return {k: v for k, v in kw.items()}


# ---------------------------------------------------------------- provider choice

def test_provider_priority_openrouter_then_anthropic_then_proxy():
    assert bc.resolve_provider(env(OPENROUTER_API_KEY="or", ANTHROPIC_API_KEY="an", METACULUS_TOKEN="t")) == "openrouter"
    assert bc.resolve_provider(env(ANTHROPIC_API_KEY="an", METACULUS_TOKEN="t")) == "anthropic"
    assert bc.resolve_provider(env(METACULUS_TOKEN="t")) == "metaculus_proxy"


def test_placeholders_are_ignored():
    assert bc.resolve_provider(env(OPENROUTER_API_KEY="REPLACE_ME", METACULUS_TOKEN="t")) == "metaculus_proxy"
    with pytest.raises(bc.ConfigError):
        bc.resolve_provider(env(OPENROUTER_API_KEY="REPLACE_ME", METACULUS_TOKEN="  "))


def test_explicit_provider_requires_its_key():
    assert bc.resolve_provider(env(BOT_LLM_PROVIDER="metaculus_proxy", OPENROUTER_API_KEY="or", METACULUS_TOKEN="t")) == "metaculus_proxy"
    with pytest.raises(bc.ConfigError, match="ANTHROPIC_API_KEY"):
        bc.resolve_provider(env(BOT_LLM_PROVIDER="anthropic", METACULUS_TOKEN="t"))
    with pytest.raises(bc.ConfigError):
        bc.resolve_provider(env(BOT_LLM_PROVIDER="gemini", METACULUS_TOKEN="t"))


# ---------------------------------------------------------------- defaults

def test_defaults_with_token_only():
    cfg = bc.load_config(env(METACULUS_TOKEN="t"))
    assert cfg.provider == "metaculus_proxy"
    assert cfg.forecaster_model == "metaculus/gpt-4o"
    assert cfg.parser_model == "metaculus/gpt-4o-mini"
    assert cfg.researcher == bc.NO_RESEARCH
    assert cfg.predictions_per_question == 5
    assert cfg.max_cost_per_question_usd == 1.0
    assert cfg.max_cost_per_run_usd == 10.0
    assert cfg.max_questions_per_run == 20
    assert cfg.effort == "medium"
    assert cfg.publish is True


def test_openrouter_and_anthropic_default_models():
    cfg = bc.load_config(env(METACULUS_TOKEN="t", OPENROUTER_API_KEY="or"))
    assert cfg.forecaster_model == "openrouter/anthropic/claude-sonnet-5.5"
    assert cfg.parser_model == "openrouter/openai/gpt-4o-mini"
    cfg = bc.load_config(env(METACULUS_TOKEN="t", ANTHROPIC_API_KEY="an"))
    assert cfg.forecaster_model == "anthropic/claude-sonnet-5-5"
    assert cfg.parser_model == "anthropic/claude-haiku-5-5"


def test_env_overrides_are_parsed():
    cfg = bc.load_config(
        env(
            METACULUS_TOKEN="t",
            ANTHROPIC_API_KEY="an",
            BOT_MODEL="opus",
            BOT_EFFORT="HIGH",
            BOT_PREDICTIONS_PER_QUESTION="3",
            BOT_MAX_COST_PER_QUESTION_USD="0.4",
            BOT_MAX_COST_PER_RUN_USD="0",
            BOT_MAX_QUESTIONS_PER_RUN="7",
            BOT_PUBLISH="false",
        )
    )
    assert cfg.forecaster_model == "anthropic/claude-opus-5-5"
    assert cfg.effort == "high"
    assert cfg.predictions_per_question == 3
    assert cfg.max_cost_per_question_usd == 0.4
    assert cfg.max_cost_per_run_usd == 0.0
    assert cfg.max_questions_per_run == 7
    assert cfg.publish is False
    assert "unlimited" in cfg.summary()  # run cost 0 = unlimited


@pytest.mark.parametrize(
    "name, value",
    [
        ("BOT_PREDICTIONS_PER_QUESTION", "0"),
        ("BOT_PREDICTIONS_PER_QUESTION", "five"),
        ("BOT_MAX_COST_PER_QUESTION_USD", "-1"),
        ("BOT_MAX_QUESTIONS_PER_RUN", "-2"),
        ("BOT_PUBLISH", "maybe"),
        ("BOT_EFFORT", "extreme"),
        ("BOT_RESEARCH", "google"),
    ],
)
def test_invalid_values_raise(name, value):
    with pytest.raises(bc.ConfigError):
        bc.load_config(env(METACULUS_TOKEN="t", **{name: value}))


def test_model_validation():
    assert bc.resolve_model("openrouter", "opus", "forecaster") == "openrouter/anthropic/claude-opus-5.5"
    assert bc.resolve_model("metaculus_proxy", "metaculus/gpt-4o-mini", "forecaster") == "metaculus/gpt-4o-mini"
    with pytest.raises(bc.ConfigError, match="prefix"):
        bc.resolve_model("openrouter", "anthropic/claude-opus-5-5", "forecaster")
    with pytest.raises(bc.ConfigError, match="not available"):
        bc.resolve_model("metaculus_proxy", "sonnet", "forecaster")


# ---------------------------------------------------------------- research

def test_research_auto_uses_asknews_only_with_keys():
    assert bc.resolve_researcher(env()) == bc.NO_RESEARCH
    assert bc.resolve_researcher(env(ASKNEWS_CLIENT_ID="id")) == bc.NO_RESEARCH  # secret missing
    assert bc.resolve_researcher(env(ASKNEWS_CLIENT_ID="id", ASKNEWS_SECRET="s")) == bc.ASKNEWS_RESEARCHER
    assert bc.resolve_researcher(env(ASKNEWS_API_KEY="k")) == bc.ASKNEWS_RESEARCHER
    assert bc.resolve_researcher(env(ASKNEWS_API_KEY="k", BOT_RESEARCH="none")) == bc.NO_RESEARCH
    with pytest.raises(bc.ConfigError):
        bc.resolve_researcher(env(BOT_RESEARCH="asknews"))


# ---------------------------------------------------------------- llm kwargs

def test_llm_kwargs_never_send_temperature():
    for model in (
        "anthropic/claude-opus-5-5",
        "anthropic/claude-haiku-5-5",
        "openrouter/anthropic/claude-sonnet-5.5",
        "metaculus/gpt-4o",
    ):
        assert "temperature" not in bc.llm_kwargs(model, "medium", 16000, 300)


def test_llm_kwargs_anthropic_thinking_and_effort():
    kw = bc.llm_kwargs("anthropic/claude-sonnet-5-5", "high", 16000, 300)
    assert kw["thinking"] == {"type": "adaptive"}
    assert kw["output_config"] == {"effort": "high"}
    assert kw["allowed_openai_params"] == ["output_config"]
    haiku = bc.llm_kwargs("anthropic/claude-haiku-5-5", "low", 4000, 120)
    assert "thinking" not in haiku and haiku["output_config"] == {"effort": "low"}
    plain = bc.llm_kwargs("openrouter/anthropic/claude-sonnet-5.5", "high", 16000, 300)
    assert "thinking" not in plain and "output_config" not in plain


def test_build_llms_and_price_registration(clean_env):
    import litellm
    from forecasting_tools import GeneralLlm

    cfg = bc.load_config(env(METACULUS_TOKEN="t", OPENROUTER_API_KEY="or", BOT_MODEL="opus"))
    llms = bc.build_llms(cfg)
    assert set(llms) == {"default", "summarizer", "researcher", "parser"}
    assert isinstance(llms["default"], GeneralLlm)
    assert type(llms["default"]).__name__ == "BudgetedLlm"
    assert type(llms["parser"]).__name__ == "BudgetedLlm"
    assert llms["default"].model == "openrouter/anthropic/claude-opus-5.5"
    assert llms["parser"].model == "openrouter/openai/gpt-4o-mini"
    assert llms["researcher"] == bc.NO_RESEARCH
    info = litellm.model_cost["openrouter/anthropic/claude-opus-5.5"]
    assert info["input_cost_per_token"] == 4e-06
    assert info["output_cost_per_token"] == 2e-05
