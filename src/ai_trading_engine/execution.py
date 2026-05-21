from __future__ import annotations

from typing import Literal

from .config import ExecutionConfig
from .models import CandidateSignal, ExecutionPlan, MarketSnapshot, PositionPlan


class ExecutionRejection(ValueError):
    pass


def plan_execution(
    signal: CandidateSignal,
    position: PositionPlan,
    snapshot: MarketSnapshot,
    cfg: ExecutionConfig,
) -> ExecutionPlan:
    if snapshot.spread_bps > cfg.max_spread_bps:
        raise ExecutionRejection(
            f"Spread too high ({snapshot.spread_bps:.2f}bps > {cfg.max_spread_bps:.2f}bps)."
        )
    if snapshot.depth_usd < cfg.min_depth_usd:
        raise ExecutionRejection(
            f"Liquidity too low (${snapshot.depth_usd:,.0f} < ${cfg.min_depth_usd:,.0f})."
        )

    breakout_context = signal.regime.strategy == "BREAKOUT"
    order_type: Literal["LIMIT", "MARKET"]
    if breakout_context and cfg.breakout_market_order_allowed:
        order_type = "MARKET"
        max_slippage = min(35.0, snapshot.spread_bps * 2.5 + 6.0)
        note = "Breakout context: prioritize immediate fill with slippage cap."
    else:
        order_type = "LIMIT"
        max_slippage = min(20.0, snapshot.spread_bps * 2.0 + 6.0)
        note = "Passive execution preferred to reduce impact."

    # Conservative square-root style market impact proxy scaled in basis points.
    participation = position.notional_usd / max(snapshot.depth_usd, 1.0)
    depth_impact = (participation ** 0.5) * 30.0
    if depth_impact > max_slippage:
        raise ExecutionRejection(
            f"Estimated slippage {depth_impact:.2f}bps exceeds limit {max_slippage:.2f}bps."
        )

    return ExecutionPlan(
        order_type=order_type,
        max_slippage_bps=max_slippage,
        expected_fill_note=note,
    )
