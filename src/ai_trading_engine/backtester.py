"""Walk-forward backtesting engine.

Replays historical candles through HybridTradingEngine with a sliding
window, simulates position management, and reports performance statistics.

Important disclaimer: past backtest results do NOT guarantee future
performance. Always validate a strategy with paper trading before risking
real capital.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Literal

from .config import EngineConfig
from .dataset import build_research_snapshot
from .engine import HybridTradingEngine
from .feature_extractor import extract_features
from .models import (
    Candle,
    MarketSnapshot,
    PortfolioState,
)

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    asset: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_idx: int
    confluence_score: float
    regime: str
    setup_family: str = "GENERIC"
    max_hold_bars: int = 24
    exit_price: float | None = None
    exit_idx: int | None = None
    pnl_pct: float | None = None
    pnl_usd: float | None = None
    outcome: Literal["WIN", "LOSS"] | None = None
    exit_reason: str | None = None
    features: dict = field(default_factory=dict)


@dataclass
class BacktestResult:
    asset: str
    timeframe: str
    total_candles: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    avg_win_pct: float
    avg_loss_pct: float
    trades: list[BacktestTrade] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"{'─'*56}",
            f"  BACKTEST: {self.asset}  [{self.timeframe}]  {self.total_candles} candles",
            f"{'─'*56}",
            f"  Trades:          {self.total_trades}",
            f"  Win rate:        {self.win_rate:.1%}  ({self.winning_trades}W / {self.losing_trades}L)",
            f"  Total return:    {self.total_return_pct:+.2f}%",
            f"  Max drawdown:    {self.max_drawdown_pct:.2f}%",
            f"  Sharpe ratio:    {self.sharpe_ratio:.2f}",
            f"  Profit factor:   {self.profit_factor:.2f}",
            f"  Avg win:         {self.avg_win_pct:+.2f}%",
            f"  Avg loss:        {self.avg_loss_pct:+.2f}%",
            f"{'─'*56}",
        ]
        return "\n".join(lines)


def _build_backtest_snapshot(
    candles: list[Candle],
    asset: str,
    timeframe: str,
) -> MarketSnapshot:
    """Backtest snapshots must match the research pipeline's structure read
    so that what we measure in diagnostics is what the engine trades here."""
    return build_research_snapshot(candles, asset, timeframe)


class Backtester:
    def __init__(self, config: EngineConfig | None = None) -> None:
        self._cfg = config or EngineConfig()
        # Historical candles are intentionally "stale" — disable the freshness
        # check effectively (10 years in minutes is enough for any realistic
        # historical backtest while keeping the check itself in place).
        self._cfg.data_validation.stale_after_minutes = 365 * 24 * 60 * 10
        self._engine = HybridTradingEngine(self._cfg)

    def run(
        self,
        candles: list[Candle],
        asset: str = "ETH/USDT",
        timeframe: str = "1h",
        initial_equity: float = 10_000.0,
        window_size: int = 220,
        risk_per_trade: float = 0.01,
    ) -> BacktestResult:
        if len(candles) < window_size + 2:
            raise ValueError(
                f"Need at least {window_size + 2} candles for backtesting, got {len(candles)}"
            )

        equity = initial_equity
        equity_curve: list[float] = [equity]
        peak_equity = equity
        max_drawdown = 0.0
        trades: list[BacktestTrade] = []
        recent_results: list[Literal["WIN", "LOSS"]] = []
        open_trade: BacktestTrade | None = None

        for i in range(window_size, len(candles) - 1):
            window = candles[i - window_size : i + 1]
            next_candle = candles[i + 1]

            # --- Manage open position against the next candle ---
            if open_trade is not None:
                bar_idx = i + 1
                closed_position = self._check_close(open_trade, next_candle, bar_idx)
                if closed_position:
                    pnl_usd = equity * risk_per_trade * (open_trade.pnl_pct or 0) / abs(
                        (open_trade.stop_loss - open_trade.entry_price) / open_trade.entry_price
                    ) if open_trade.entry_price != open_trade.stop_loss else 0
                    open_trade.pnl_usd = pnl_usd
                    equity += pnl_usd
                    recent_results.append(open_trade.outcome or "LOSS")
                    if len(recent_results) > 30:
                        recent_results = recent_results[-30:]
                    open_trade = None

            # --- Ask engine for a signal (only when flat) ---
            if open_trade is None:
                snapshot = _build_backtest_snapshot(window, asset, timeframe)
                recent_stamps = [
                    candles[t.entry_idx].timestamp.isoformat()
                    for t in trades
                    if t.entry_idx is not None
                ]
                portfolio = PortfolioState(
                    equity_usd=equity,
                    open_positions={},
                    recent_results=recent_results,
                    recent_trade_timestamps=recent_stamps,
                )
                try:
                    decision = self._engine.evaluate(snapshot, portfolio)
                except Exception as exc:
                    logger.debug("Engine error at candle %d: %s", i, exc)
                    decision = None

                if decision and decision.is_trade and decision.signal:
                    s = decision.signal
                    tp1 = s.candidate.take_profits[0] if s.candidate.take_profits else s.candidate.entry
                    open_trade = BacktestTrade(
                        asset=asset,
                        direction=s.candidate.direction,
                        entry_price=s.candidate.entry,
                        stop_loss=s.candidate.stop_loss,
                        take_profit=tp1,
                        entry_idx=i,
                        confluence_score=s.confluence.total_score,
                        regime=s.candidate.regime.regime,
                        setup_family=s.candidate.setup_family,
                        max_hold_bars=s.candidate.max_hold_bars,
                        features=extract_features(s),
                    )
                    trades.append(open_trade)

            # --- Track equity curve ---
            equity_curve.append(equity)
            if equity > peak_equity:
                peak_equity = equity
            dd = (peak_equity - equity) / peak_equity
            if dd > max_drawdown:
                max_drawdown = dd

        # Force-close any open trade at last candle's close
        if open_trade is not None and candles:
            last = candles[-1]
            ep = open_trade.entry_price
            if open_trade.direction == "LONG":
                raw_pnl = (last.close - ep) / ep
            else:
                raw_pnl = (ep - last.close) / ep
            open_trade.exit_price = last.close
            open_trade.exit_idx = len(candles) - 1
            open_trade.pnl_pct = raw_pnl
            open_trade.outcome = "WIN" if raw_pnl > 0 else "LOSS"
            open_trade.pnl_usd = equity * risk_per_trade * raw_pnl

        # --- Aggregate statistics ---
        closed_trades = [t for t in trades if t.outcome is not None]
        wins = [t for t in closed_trades if t.outcome == "WIN"]
        losses = [t for t in closed_trades if t.outcome == "LOSS"]
        win_rate = len(wins) / len(closed_trades) if closed_trades else 0.0
        total_return = (equity - initial_equity) / initial_equity * 100

        win_pnls = [t.pnl_pct or 0.0 for t in wins]
        loss_pnls = [t.pnl_pct or 0.0 for t in losses]
        avg_win = (sum(win_pnls) / len(win_pnls) * 100) if win_pnls else 0.0
        avg_loss = (sum(loss_pnls) / len(loss_pnls) * 100) if loss_pnls else 0.0

        gross_profit = sum(t.pnl_usd or 0 for t in wins)
        gross_loss = abs(sum(t.pnl_usd or 0 for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        sharpe = _sharpe(equity_curve)

        return BacktestResult(
            asset=asset,
            timeframe=timeframe,
            total_candles=len(candles),
            total_trades=len(closed_trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            total_return_pct=total_return,
            max_drawdown_pct=max_drawdown * 100,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            trades=trades,
        )

    def _check_close(self, trade: BacktestTrade, candle: Candle, bar_idx: int) -> bool:
        """Returns True if the trade should be closed on this candle.

        Intrabar ordering pessimistically assumes the stop fills first when
        both levels are touched inside the same bar, matching the research
        labeling convention in dataset.label_trade_path.
        """
        bars_held = bar_idx - trade.entry_idx
        if trade.direction == "LONG":
            hit_stop = candle.low <= trade.stop_loss
            hit_tp = candle.high >= trade.take_profit
            if hit_stop:
                trade.exit_price = trade.stop_loss
                trade.pnl_pct = (trade.stop_loss - trade.entry_price) / trade.entry_price
                trade.outcome = "LOSS"
                trade.exit_reason = "STOP_LOSS"
                trade.exit_idx = bar_idx
                return True
            if hit_tp:
                trade.exit_price = trade.take_profit
                trade.pnl_pct = (trade.take_profit - trade.entry_price) / trade.entry_price
                trade.outcome = "WIN"
                trade.exit_reason = "TAKE_PROFIT"
                trade.exit_idx = bar_idx
                return True
        else:  # SHORT
            hit_stop = candle.high >= trade.stop_loss
            hit_tp = candle.low <= trade.take_profit
            if hit_stop:
                trade.exit_price = trade.stop_loss
                trade.pnl_pct = (trade.entry_price - trade.stop_loss) / trade.entry_price
                trade.outcome = "LOSS"
                trade.exit_reason = "STOP_LOSS"
                trade.exit_idx = bar_idx
                return True
            if hit_tp:
                trade.exit_price = trade.take_profit
                trade.pnl_pct = (trade.entry_price - trade.take_profit) / trade.entry_price
                trade.outcome = "WIN"
                trade.exit_reason = "TAKE_PROFIT"
                trade.exit_idx = bar_idx
                return True

        # Family-specific horizon exit — matches research label horizon and
        # prevents mean-reversion trades from holding forever when TP/SL both miss.
        if bars_held >= max(1, trade.max_hold_bars):
            if trade.direction == "LONG":
                trade.pnl_pct = (candle.close - trade.entry_price) / trade.entry_price
            else:
                trade.pnl_pct = (trade.entry_price - candle.close) / trade.entry_price
            trade.exit_price = candle.close
            trade.outcome = "WIN" if (trade.pnl_pct or 0) > 0 else "LOSS"
            trade.exit_reason = "HORIZON"
            trade.exit_idx = bar_idx
            return True
        return False


def _sharpe(equity_curve: list[float], risk_free: float = 0.0) -> float:
    if len(equity_curve) < 2:
        return 0.0
    returns = [
        (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
        for i in range(1, len(equity_curve))
        if equity_curve[i - 1] > 0
    ]
    if not returns:
        return 0.0
    mean_r = sum(returns) / len(returns)
    variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
    std_r = math.sqrt(variance)
    if std_r == 0:
        return 0.0
    # Annualise to hourly candles (8760 periods/yr)
    return (mean_r - risk_free) / std_r * math.sqrt(8760)
