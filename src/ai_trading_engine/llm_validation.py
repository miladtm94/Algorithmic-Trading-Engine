from __future__ import annotations

import os
from dataclasses import dataclass

from .config import LLMConfig
from .models import CandidateSignal, ConfluenceBreakdown, RegimeResult

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]


@dataclass(slots=True)
class LLMDecision:
    approved: bool
    note: str


def _rule_based_validation(
    signal: CandidateSignal,
    confluence: ConfluenceBreakdown,
    regime: RegimeResult,
) -> LLMDecision:
    if signal.direction == "LONG" and regime.regime == "TRENDING_BEARISH":
        return LLMDecision(False, "Conflict: LONG signal in bearish trend regime.")
    if signal.direction == "SHORT" and regime.regime == "TRENDING_BULLISH":
        return LLMDecision(False, "Conflict: SHORT signal in bullish trend regime.")
    if len(signal.reasons) < 3:
        return LLMDecision(False, "Insufficient multi-factor confluence.")
    # The engine-level confluence threshold is the single source of truth for
    # the confluence floor (see HybridTradingEngine.evaluate). A second
    # hardcoded floor here diverges from research settings, so we only use the
    # confluence breakdown for context, not as an extra gate.
    _ = confluence
    return LLMDecision(True, "Rule-based validation passed.")


def validate_with_llm(
    signal: CandidateSignal,
    confluence: ConfluenceBreakdown,
    regime: RegimeResult,
    cfg: LLMConfig,
) -> LLMDecision:
    baseline = _rule_based_validation(signal, confluence, regime)
    if not baseline.approved:
        return baseline

    if not cfg.enabled:
        return baseline
    if OpenAI is None:
        return LLMDecision(True, "LLM dependency unavailable; used rule-based validation.")
    if not os.getenv("OPENAI_API_KEY"):
        return LLMDecision(True, "OPENAI_API_KEY missing; used rule-based validation.")

    client = OpenAI()
    prompt = (
        "You are validating a trading signal. Reply strictly as JSON with keys "
        "approved (boolean) and note (short string). Reject if logic conflicts with regime "
        "or if hidden risk appears inconsistent.\n"
        f"Signal direction: {signal.direction}\n"
        f"Regime: {regime.regime} / {regime.strategy}\n"
        f"Confluence score: {confluence.total_score:.2f}\n"
        f"Reasons: {', '.join(signal.reasons)}\n"
        f"Entry: {signal.entry:.4f}, Stop: {signal.stop_loss:.4f}, RR: {signal.risk_reward:.2f}\n"
    )
    try:
        response = client.responses.create(
            model=cfg.model,
            input=prompt,
            temperature=0.0,
        )
        text = getattr(response, "output_text", "") or ""
        if '"approved": false' in text.lower():
            return LLMDecision(False, "LLM rejected signal due to contextual conflict.")
        return LLMDecision(True, "LLM validation passed.")
    except Exception as exc:  # pragma: no cover
        return LLMDecision(True, f"LLM unavailable ({exc}); used rule-based validation.")
