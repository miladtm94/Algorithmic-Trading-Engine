from __future__ import annotations

from .config import RiskConfig
from .models import CandidateSignal, PortfolioState, PositionPlan


class RiskRejection(ValueError):
    pass


def enforce_portfolio_constraints(
    signal: CandidateSignal,
    portfolio: PortfolioState,
    cfg: RiskConfig,
) -> None:
    if portfolio.consecutive_losses >= cfg.kill_switch_after_losses:
        raise RiskRejection(
            f"Kill-switch active after {portfolio.consecutive_losses} consecutive losses."
        )
    if len(portfolio.open_positions) >= cfg.max_concurrent_trades:
        raise RiskRejection("Maximum concurrent trades reached.")
    if signal.asset in portfolio.open_positions:
        raise RiskRejection(f"Asset {signal.asset} already has an open position.")


def enforce_correlation_filter(correlations: dict[str, float], cfg: RiskConfig) -> None:
    max_corr = max((abs(v) for v in correlations.values()), default=0.0)
    if max_corr >= cfg.correlation_limit:
        raise RiskRejection(
            f"Correlation filter breach ({max_corr:.2f} >= {cfg.correlation_limit:.2f})."
        )


def build_position_plan(
    signal: CandidateSignal,
    portfolio: PortfolioState,
    cfg: RiskConfig,
    risk_multiplier: float = 1.0,
) -> PositionPlan:
    stop_distance = abs(signal.entry - signal.stop_loss)
    if stop_distance <= 0:
        raise RiskRejection("Invalid stop distance.")

    if signal.risk_reward < cfg.min_rr:
        raise RiskRejection(
            f"Risk/Reward too low ({signal.risk_reward:.2f} < {cfg.min_rr:.2f})."
        )

    base_risk_pct = cfg.risk_per_trade_pct * max(0.1, min(1.0, risk_multiplier))
    base_risk_usd = portfolio.equity_usd * base_risk_pct

    atr_factor = signal.indicators.atr_pct
    volatility_scalar = max(0.35, min(1.0, 1.0 - (atr_factor * 20)))
    adjusted_risk_usd = base_risk_usd * volatility_scalar

    quantity = adjusted_risk_usd / stop_distance
    notional = quantity * signal.entry

    return PositionPlan(
        quantity=quantity,
        notional_usd=notional,
        risk_usd=adjusted_risk_usd,
        risk_pct=(adjusted_risk_usd / portfolio.equity_usd) if portfolio.equity_usd else 0.0,
    )
