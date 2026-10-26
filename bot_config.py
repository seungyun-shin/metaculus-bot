"""
Environment-driven configuration for the bot.

Importing this module has no side effects and does not import
forecasting_tools, so it can be unit-tested offline. `build_llms()` is the only
function that touches forecasting_tools / litellm, and it imports them lazily.

Model strings and where they were verified (2026-10-26):
- metaculus/gpt-4o, metaculus/gpt-4o-mini: forecasting-tools 0.3.5,
  ForecastBot._llm_config_defaults() -> the branch used when only METACULUS_TOKEN
  is set. GeneralLlm routes the "metaculus/" prefix to
  https://llm-proxy.metaculus.com/proxy/{openai/v1|anthropic} with the token.
- openrouter/...: ids from https://openrouter.ai/api/v1/models. The template
  README says the free LLM credits from the participation form are delivered as
  an OPENROUTER_API_KEY.
- anthropic/claude-*-5-5: litellm model_cost (provider "anthropic") and the
  Anthropic model table (claude-opus-5-5, claude-sonnet-5-5, claude-haiku-5-5).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

PLACEHOLDER_VALUES = {"", "REPLACE_ME", "1234567890", "your-token-here", "your-api-key-here"}

PROVIDERS = ("openrouter", "anthropic", "metaculus_proxy")
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

PROVIDER_DEFAULT_MODELS: dict[str, dict[str, str]] = {
    "openrouter": {
        "forecaster": "openrouter/anthropic/claude-sonnet-5.5",
        "parser": "openrouter/openai/gpt-4o-mini",
    },
    "anthropic": {
        "forecaster": "anthropic/claude-sonnet-5-5",
        "parser": "anthropic/claude-haiku-5-5",
    },
    "metaculus_proxy": {
        "forecaster": "metaculus/gpt-4o",
        "parser": "metaculus/gpt-4o-mini",
    },
}

# Short aliases for BOT_MODEL. Not offered for the Metaculus proxy because its
# list of allowed Claude models is not documented anywhere we could check.
MODEL_ALIASES: dict[str, dict[str, str]] = {
    "openrouter": {
        "sonnet": "openrouter/anthropic/claude-sonnet-5.5",
        "opus": "openrouter/anthropic/claude-opus-5.5",
        "haiku": "openrouter/anthropic/claude-haiku-5.5",
    },
    "anthropic": {
        "sonnet": "anthropic/claude-sonnet-5-5",
        "opus": "anthropic/claude-opus-5-5",
        "haiku": "anthropic/claude-haiku-5-5",
    },
}

# USD per token (input, output) from https://openrouter.ai/api/v1/models on
# 2026-10-26. litellm ships no OpenRouter prices, so without registering these
# the per-question cost cap could not see OpenRouter spend.
OPENROUTER_PRICES: dict[str, tuple[float, float]] = {
    "openrouter/anthropic/claude-sonnet-5.5": (2e-06, 1e-05),
    "openrouter/anthropic/claude-opus-5.5": (4e-06, 2e-05),
    "openrouter/anthropic/claude-haiku-5.5": (1e-07, 5e-07),
    "openrouter/openai/gpt-4o-mini": (1.5e-07, 6e-07),
    "openrouter/openai/gpt-4o": (2.5e-06, 1e-05),
}

ASKNEWS_RESEARCHER = "asknews/news-summaries"
NO_RESEARCH = "no_research"


class ConfigError(ValueError):
    pass


def is_real(value: str | None) -> bool:
    return bool(value and value.strip() and value.strip() not in PLACEHOLDER_VALUES)


@dataclass(frozen=True)
class BotConfig:
    provider: str
    forecaster_model: str
    parser_model: str
    researcher: str
    predictions_per_question: int
    max_cost_per_question_usd: float
    max_cost_per_run_usd: float
    max_questions_per_run: int
    effort: str
    publish: bool
    max_tokens: int
    llm_timeout_seconds: int

    def summary(self) -> str:
        cost_q = f"${self.max_cost_per_question_usd:.2f}" if self.max_cost_per_question_usd else "unlimited"
        cost_r = f"${self.max_cost_per_run_usd:.2f}" if self.max_cost_per_run_usd else "unlimited"
        max_q = str(self.max_questions_per_run) if self.max_questions_per_run else "unlimited"
        return (
            f"provider={self.provider} forecaster={self.forecaster_model} "
            f"parser={self.parser_model} research={self.researcher} "
            f"samples={self.predictions_per_question} effort={self.effort} "
            f"cost/question<={cost_q} cost/run<={cost_r} questions/run<={max_q} "
            f"publish={'yes' if self.publish else 'no (dry run)'}"
        )


def _get(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    return value.strip() if is_real(value) else None


def _parse_int(env: Mapping[str, str], name: str, default: int, minimum: int) -> int:
    raw = _get(env, name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as e:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from e
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


def _parse_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = _get(env, name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as e:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from e
    if value < 0:
        raise ConfigError(f"{name} must be >= 0 (0 = unlimited), got {value}")
    return value


def _parse_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _get(env, name)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in ("1", "true", "yes", "y", "on"):
        return True
    if lowered in ("0", "false", "no", "n", "off"):
        return False
    raise ConfigError(f"{name} must be true/false, got {raw!r}")


def resolve_provider(env: Mapping[str, str]) -> str:
    """
    Explicit BOT_LLM_PROVIDER wins. Otherwise ("auto"):
      1. OPENROUTER_API_KEY  -> openrouter  (the key Metaculus issues with free credits)
      2. ANTHROPIC_API_KEY   -> anthropic   (your own Claude key)
      3. METACULUS_TOKEN     -> metaculus_proxy (Metaculus LLM proxy, token only)
    """
    explicit = (_get(env, "BOT_LLM_PROVIDER") or "auto").lower()
    if explicit != "auto":
        if explicit not in PROVIDERS:
            raise ConfigError(
                f"BOT_LLM_PROVIDER must be one of auto/{'/'.join(PROVIDERS)}, got {explicit!r}"
            )
        required = {
            "openrouter": "OPENROUTER_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "metaculus_proxy": "METACULUS_TOKEN",
        }[explicit]
        if not _get(env, required):
            raise ConfigError(f"BOT_LLM_PROVIDER={explicit} needs {required} to be set")
        return explicit
    if _get(env, "OPENROUTER_API_KEY"):
        return "openrouter"
    if _get(env, "ANTHROPIC_API_KEY"):
        return "anthropic"
    if _get(env, "METACULUS_TOKEN"):
        return "metaculus_proxy"
    raise ConfigError(
        "No LLM access configured. Set OPENROUTER_API_KEY, ANTHROPIC_API_KEY, or at least METACULUS_TOKEN."
    )


def resolve_model(provider: str, raw: str | None, role: str) -> str:
    if raw is None:
        return PROVIDER_DEFAULT_MODELS[provider][role]
    alias = MODEL_ALIASES.get(provider, {}).get(raw.lower())
    if alias:
        return alias
    if raw.lower() in ("sonnet", "opus", "haiku"):
        raise ConfigError(
            f"Alias {raw!r} is not available for provider {provider}; give a full model string "
            f"(e.g. {PROVIDER_DEFAULT_MODELS[provider][role]!r})"
        )
    expected_prefix = {"openrouter": "openrouter/", "anthropic": "anthropic/", "metaculus_proxy": "metaculus/"}[provider]
    if not raw.startswith(expected_prefix):
        raise ConfigError(
            f"Model {raw!r} does not match provider {provider} (expected prefix {expected_prefix!r})"
        )
    return raw


def resolve_researcher(env: Mapping[str, str]) -> str:
    mode = (_get(env, "BOT_RESEARCH") or "auto").lower()
    has_asknews = bool(
        (_get(env, "ASKNEWS_CLIENT_ID") and _get(env, "ASKNEWS_SECRET"))
        or _get(env, "ASKNEWS_API_KEY")
    )
    if mode == "none":
        return NO_RESEARCH
    if mode == "asknews":
        if not has_asknews:
            raise ConfigError("BOT_RESEARCH=asknews needs ASKNEWS_CLIENT_ID + ASKNEWS_SECRET (or ASKNEWS_API_KEY)")
        return ASKNEWS_RESEARCHER
    if mode != "auto":
        raise ConfigError(f"BOT_RESEARCH must be auto/asknews/none, got {mode!r}")
    return ASKNEWS_RESEARCHER if has_asknews else NO_RESEARCH


def load_config(env: Mapping[str, str] | None = None) -> BotConfig:
    env = os.environ if env is None else env
    provider = resolve_provider(env)
    effort = (_get(env, "BOT_EFFORT") or "medium").lower()
    if effort not in EFFORT_LEVELS:
        raise ConfigError(f"BOT_EFFORT must be one of {EFFORT_LEVELS}, got {effort!r}")
    return BotConfig(
        provider=provider,
        forecaster_model=resolve_model(provider, _get(env, "BOT_MODEL"), "forecaster"),
        parser_model=resolve_model(provider, _get(env, "BOT_PARSER_MODEL"), "parser"),
        researcher=resolve_researcher(env),
        predictions_per_question=_parse_int(env, "BOT_PREDICTIONS_PER_QUESTION", 5, 1),
        max_cost_per_question_usd=_parse_float(env, "BOT_MAX_COST_PER_QUESTION_USD", 1.0),
        max_cost_per_run_usd=_parse_float(env, "BOT_MAX_COST_PER_RUN_USD", 10.0),
        max_questions_per_run=_parse_int(env, "BOT_MAX_QUESTIONS_PER_RUN", 20, 0),
        effort=effort,
        publish=_parse_bool(env, "BOT_PUBLISH", True),
        max_tokens=_parse_int(env, "BOT_MAX_TOKENS", 16000, 1024),
        llm_timeout_seconds=_parse_int(env, "BOT_LLM_TIMEOUT_SECONDS", 300, 10),
    )


def llm_kwargs(model: str, effort: str, max_tokens: int, timeout: int) -> dict[str, Any]:
    """
    Keyword arguments for forecasting_tools.GeneralLlm.

    No temperature is ever sent: Claude Opus 5.5 rejects sampling parameters and
    Sonnet/Haiku 5.5 reject non-default values. GeneralLlm's default is None,
    which litellm drops.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "allowed_tries": 2,
    }
    if model.startswith("anthropic/"):
        # Same pattern Metaculus uses for its own MainBot in forecasting-tools.
        if "haiku" not in model:
            kwargs["thinking"] = {"type": "adaptive"}
        kwargs["output_config"] = {"effort": effort}
        kwargs["allowed_openai_params"] = ["output_config"]
    return kwargs


def register_prices(models: list[str]) -> list[str]:
    """Register OpenRouter prices with litellm so cost caps see the spend."""
    import litellm

    to_register = {
        m: {
            "input_cost_per_token": OPENROUTER_PRICES[m][0],
            "output_cost_per_token": OPENROUTER_PRICES[m][1],
            "litellm_provider": "openrouter",
            "mode": "chat",
        }
        for m in models
        if m in OPENROUTER_PRICES
    }
    if to_register:
        litellm.register_model(to_register)
    return sorted(to_register)


def build_llms(config: BotConfig) -> dict[str, Any]:
    from budgeted_llm import BudgetedLlm

    register_prices([config.forecaster_model, config.parser_model])
    forecaster = BudgetedLlm(
        **llm_kwargs(config.forecaster_model, config.effort, config.max_tokens, config.llm_timeout_seconds)
    )
    parser = BudgetedLlm(
        **llm_kwargs(config.parser_model, "low", 4000, min(config.llm_timeout_seconds, 120))
    )
    return {
        "default": forecaster,
        "summarizer": parser,
        "researcher": config.researcher,
        "parser": parser,
    }
