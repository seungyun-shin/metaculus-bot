"""
Cost-cap fixes for forecasting-tools 0.3.5 + litellm 1.105 (versions in uv.lock).

Two problems were found while testing offline with litellm's mock_response:

1. The cap never blocks a call. forecasting-tools enforces MonetaryCostManager
   limits from a litellm pre-call callback, but litellm catches exceptions
   raised there, logs "litellm.Logging.pre_call(): Exception occured" and makes
   the call anyway. -> BudgetedLlm checks the limit itself before each call.

2. Every call is counted twice. GeneralLlm adds the response's cost directly
   when its callback has not reported anything yet. But litellm runs the async
   success callback as a background task that finishes *after* that check, and
   the callback then adds the same cost again. -> disable_callback_cost_tracking()
   turns the callback into a no-op, so only GeneralLlm's direct accounting
   (response._hidden_params["response_cost"]) counts. All LLM calls in this bot
   go through GeneralLlm, so nothing is lost.
"""
from __future__ import annotations

from forecasting_tools import GeneralLlm
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    LitellmCostTracker,
    MonetaryCostManager,
)


def disable_callback_cost_tracking() -> None:
    def _no_op(self, *args, **kwargs) -> None:
        return None

    LitellmCostTracker.log_success_event = _no_op  # type: ignore[method-assign]
    LitellmCostTracker.async_log_success_event = _no_op  # type: ignore[method-assign]


disable_callback_cost_tracking()


class BudgetedLlm(GeneralLlm):
    async def invoke(self, prompt, system_prompt=None):  # type: ignore[override]
        MonetaryCostManager.raise_error_if_limit_would_be_reached()
        return await super().invoke(prompt, system_prompt=system_prompt)
