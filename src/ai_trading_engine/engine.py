from __future__ import annotations

from .config import EngineConfig
from .confluence import score_candidate
from .data_validation import DataValidationError, validate_snapshot
from .execution import ExecutionRejection, plan_execution
from .indicators import compute_indicators
from .llm_validation import validate_with_llm
from .models import EngineDecision, FinalSignal, MarketSnapshot, PortfolioState
from .news import apply_event_filter
from .regime import classify_regime
from .risk import (
    RiskRejection,
    build_position_plan,
    enforce_correlation_filter,
    enforce_portfolio_constraints,
)
from .signal_generation import generate_candidate


class HybridTradingEngine:
    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    def evaluate(self, snapshot: MarketSnapshot, portfolio: PortfolioState) -> EngineDecision:
        try:
            validate_snapshot(snapshot, self.config.data_validation)
        except DataValidationError as exc:
            return EngineDecision(signal=None, no_trade_reason=f"Data validation failed: {exc}")

        event_decision = apply_event_filter(snapshot.events, self.config.event)
        if not event_decision.allow_trading:
            return EngineDecision(signal=None, no_trade_reason=event_decision.note)

        indicators = compute_indicators(snapshot.candles, snapshot.order_book)
        regime = classify_regime(indicators, self.config.regime)
        candidate = generate_candidate(snapshot, regime, indicators)
        if candidate is None:
            return EngineDecision(
                signal=None,
                no_trade_reason="Insufficient confluence from quant layer candidate generation.",
            )

        confluence = score_candidate(candidate, snapshot, self.config.confluence_weights)
        if confluence.total_score < self.config.confluence_threshold:
            return EngineDecision(
                signal=None,
                no_trade_reason=(
                    f"Confluence score too low ({confluence.total_score:.1f}% < "
                    f"{self.config.confluence_threshold:.1f}%)."
                ),
            )

        try:
            enforce_portfolio_constraints(candidate, portfolio, self.config.risk)
            enforce_correlation_filter(snapshot.correlation_to_open_positions, self.config.risk)
            position = build_position_plan(
                candidate,
                portfolio,
                self.config.risk,
                risk_multiplier=event_decision.risk_multiplier,
            )
        except RiskRejection as exc:
            return EngineDecision(signal=None, no_trade_reason=f"Risk rejection: {exc}")

        try:
            execution = plan_execution(candidate, position, snapshot, self.config.execution)
        except ExecutionRejection as exc:
            return EngineDecision(signal=None, no_trade_reason=f"Execution rejection: {exc}")

        llm_decision = validate_with_llm(candidate, confluence, regime, self.config.llm)
        if not llm_decision.approved:
            return EngineDecision(signal=None, no_trade_reason=f"LLM validation failed: {llm_decision.note}")

        final = FinalSignal(
            candidate=candidate,
            confluence=confluence,
            execution=execution,
            position=position,
            news_note=event_decision.note,
            llm_note=llm_decision.note,
        )
        return EngineDecision(signal=final)
