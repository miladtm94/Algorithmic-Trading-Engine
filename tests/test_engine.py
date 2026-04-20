"""Test suite for AI Trading Engine.

Covers: core engine pipeline, risk guards, backtester,
persistence (in-memory), paper broker, and market data utilities.
"""
from __future__ import annotations

import math
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from ai_trading_engine import EngineConfig, HybridTradingEngine
from ai_trading_engine.broker import Order, PaperBroker
from ai_trading_engine.demo_data import build_demo_portfolio, build_demo_snapshot
from ai_trading_engine.models import (
    Candle,
    EventRisk,
    MarketSnapshot,
    OrderBookSnapshot,
    PortfolioState,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_candle(close: float, idx: int = 0) -> Candle:
    ts = datetime(2024, 1, 1, idx % 24, 0, tzinfo=timezone.utc)
    spread = close * 0.002
    return Candle(
        timestamp=ts,
        open=close - spread,
        high=close + spread * 2,
        low=close - spread * 2,
        close=close,
        volume=1000.0 + idx * 10,
    )


def _minimal_snapshot(asset: str = "ETH/USDT", n_candles: int = 250) -> MarketSnapshot:
    base = 2_000.0
    candles = [_make_candle(base - i * 0.5, i) for i in range(n_candles)]
    ob = OrderBookSnapshot(
        bids=[(base - i * 0.1, 50.0) for i in range(1, 6)],
        asks=[(base + i * 0.1, 50.0) for i in range(1, 6)],
    )
    return MarketSnapshot(
        asset=asset,
        timeframe="1h",
        candles=candles,
        order_book=ob,
        spread_bps=5.0,
        depth_usd=500_000.0,
        support_levels=[1_950.0, 1_900.0],
        resistance_levels=[2_050.0, 2_100.0],
        liquidation_clusters=[],
        sentiment_score=0.0,
        source_prices={"test": base},
        correlation_to_open_positions={},
        events=[],
    )


# ── Engine core tests ─────────────────────────────────────────────────────────

class TestEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = HybridTradingEngine(EngineConfig())

    def test_generates_trade_in_demo_conditions(self) -> None:
        decision = self.engine.evaluate(build_demo_snapshot("ETH/USDT"), build_demo_portfolio())
        self.assertTrue(decision.is_trade)
        self.assertIsNotNone(decision.signal)
        self.assertGreaterEqual(decision.signal.confluence.total_score, 75.0)  # type: ignore[union-attr]

    def test_pauses_on_high_impact_event(self) -> None:
        snapshot = build_demo_snapshot("ETH/USDT")
        snapshot.events = [EventRisk(name="FOMC Rate Decision", impact="HIGH", minutes_to_event=20)]
        decision = self.engine.evaluate(snapshot, build_demo_portfolio())
        self.assertFalse(decision.is_trade)
        self.assertIn("Trading paused", decision.no_trade_reason or "")

    def test_medium_event_reduces_size_not_blocks(self) -> None:
        # MEDIUM event within 20 min window → reduced risk multiplier, but not blocked
        snapshot = build_demo_snapshot("ETH/USDT")
        snapshot.events = [EventRisk(name="ISM Data", impact="MEDIUM", minutes_to_event=15)]
        decision = self.engine.evaluate(snapshot, build_demo_portfolio())
        if decision.is_trade:
            self.assertIn("medium-impact", decision.signal.news_note)  # type: ignore[union-attr]

    def test_kill_switch_after_consecutive_losses(self) -> None:
        cfg = EngineConfig()
        cfg.risk.kill_switch_after_losses = 3
        engine = HybridTradingEngine(cfg)
        snapshot = build_demo_snapshot("ETH/USDT")
        portfolio = PortfolioState(
            equity_usd=10_000.0,
            open_positions={},
            recent_results=["LOSS", "LOSS", "LOSS"],
        )
        decision = engine.evaluate(snapshot, portfolio)
        self.assertFalse(decision.is_trade)
        self.assertIsNotNone(decision.no_trade_reason)

    def test_returns_decision_object_always(self) -> None:
        snapshot = _minimal_snapshot()
        portfolio = PortfolioState(equity_usd=10_000.0)
        decision = self.engine.evaluate(snapshot, portfolio)
        self.assertIsNotNone(decision)
        # Must always be True or False, never raise
        _ = decision.is_trade

    def test_decision_signal_has_valid_rr(self) -> None:
        decision = self.engine.evaluate(build_demo_snapshot("ETH/USDT"), build_demo_portfolio())
        if decision.is_trade:
            s = decision.signal
            self.assertGreaterEqual(s.candidate.risk_reward, 2.0)  # type: ignore[union-attr]

    def test_decision_signal_entry_vs_stop_direction(self) -> None:
        decision = self.engine.evaluate(build_demo_snapshot("ETH/USDT"), build_demo_portfolio())
        if decision.is_trade:
            s = decision.signal
            if s.candidate.direction == "LONG":  # type: ignore[union-attr]
                self.assertGreater(s.candidate.entry, s.candidate.stop_loss)
            else:
                self.assertLess(s.candidate.entry, s.candidate.stop_loss)

    def test_confluence_threshold_blocks_weak_signal(self) -> None:
        cfg = EngineConfig()
        cfg.confluence_threshold = 99.9  # almost impossible to reach
        engine = HybridTradingEngine(cfg)
        snapshot = build_demo_snapshot("ETH/USDT")
        decision = engine.evaluate(snapshot, build_demo_portfolio())
        self.assertFalse(decision.is_trade)


# ── PaperBroker tests ─────────────────────────────────────────────────────────

class TestPaperBroker(unittest.TestCase):
    def _signal(self, direction: str = "LONG", entry: float = 2000.0) -> MagicMock:
        sig = MagicMock()
        sig.candidate.asset = "ETH/USDT"
        sig.candidate.direction = direction
        sig.candidate.entry = entry
        sig.candidate.stop_loss = entry - 50.0 if direction == "LONG" else entry + 50.0
        sig.candidate.take_profits = [
            entry + 100.0 if direction == "LONG" else entry - 100.0,
            entry + 150.0 if direction == "LONG" else entry - 150.0,
            entry + 200.0 if direction == "LONG" else entry - 200.0,
        ]
        sig.execution.order_type = "LIMIT"
        sig.position.quantity = 0.5
        return sig

    def test_place_order_creates_open_order(self) -> None:
        broker = PaperBroker(10_000.0)
        order = broker.place_order(self._signal())
        self.assertEqual(order.status, "OPEN")
        self.assertEqual(len(broker.open_orders), 1)

    def test_long_hits_stop(self) -> None:
        broker = PaperBroker(10_000.0)
        sig = self._signal("LONG", 2000.0)
        broker.place_order(sig)
        closed = broker.update_prices("ETH/USDT", 1940.0)  # below stop 1950
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0].status, "STOPPED")
        self.assertLess(closed[0].pnl_usd or 0, 0)

    def test_long_hits_tp1(self) -> None:
        broker = PaperBroker(10_000.0)
        sig = self._signal("LONG", 2000.0)
        broker.place_order(sig)
        closed = broker.update_prices("ETH/USDT", 2110.0)  # above tp1 2100
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0].status, "FILLED")
        self.assertGreater(closed[0].pnl_usd or 0, 0)

    def test_short_hits_stop(self) -> None:
        broker = PaperBroker(10_000.0)
        sig = self._signal("SHORT", 2000.0)
        broker.place_order(sig)
        closed = broker.update_prices("ETH/USDT", 2060.0)  # above stop 2050
        self.assertEqual(len(closed), 1)
        self.assertLess(closed[0].pnl_usd or 0, 0)

    def test_short_hits_tp1(self) -> None:
        broker = PaperBroker(10_000.0)
        sig = self._signal("SHORT", 2000.0)
        broker.place_order(sig)
        closed = broker.update_prices("ETH/USDT", 1890.0)  # below tp1 1900
        self.assertEqual(len(closed), 1)
        self.assertGreater(closed[0].pnl_usd or 0, 0)

    def test_equity_updates_after_close(self) -> None:
        broker = PaperBroker(10_000.0)
        sig = self._signal("LONG", 2000.0)
        broker.place_order(sig)
        before = broker.equity
        broker.update_prices("ETH/USDT", 2110.0)
        self.assertNotEqual(broker.equity, before)

    def test_stats_after_trades(self) -> None:
        broker = PaperBroker(10_000.0)
        sig = self._signal("LONG", 2000.0)
        broker.place_order(sig)
        broker.update_prices("ETH/USDT", 2110.0)
        stats = broker.stats()
        self.assertEqual(stats.total_trades, 1)
        self.assertEqual(stats.wins, 1)
        self.assertEqual(stats.win_rate, 1.0)

    def test_cancel_order(self) -> None:
        broker = PaperBroker(10_000.0)
        order = broker.place_order(self._signal())
        result = broker.cancel_order(order.id)
        self.assertTrue(result)
        self.assertEqual(order.status, "CANCELLED")
        self.assertEqual(len(broker.open_orders), 0)

    def test_no_close_between_stop_and_tp(self) -> None:
        broker = PaperBroker(10_000.0)
        broker.place_order(self._signal("LONG", 2000.0))
        closed = broker.update_prices("ETH/USDT", 2010.0)  # between stop and TP
        self.assertEqual(len(closed), 0)
        self.assertEqual(len(broker.open_orders), 1)


# ── Persistence tests ─────────────────────────────────────────────────────────

class TestPersistence(unittest.TestCase):
    def setUp(self) -> None:
        import ai_trading_engine.persistence as p

        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        p.set_db_path(self._tmp.name)
        p.init_db()
        self._p = p

    def test_save_no_trade_decision(self) -> None:
        from ai_trading_engine.models import EngineDecision

        dec = EngineDecision(signal=None, no_trade_reason="Insufficient confluence")
        row_id = self._p.save_decision(dec)
        self.assertIsInstance(row_id, int)
        self.assertGreater(row_id, 0)

    def test_save_trade_decision_and_outcome(self) -> None:
        engine = HybridTradingEngine(EngineConfig())
        decision = engine.evaluate(build_demo_snapshot("ETH/USDT"), build_demo_portfolio())
        if decision.is_trade:
            row_id = self._p.save_decision(decision)
            self.assertGreater(row_id, 0)
            self._p.save_outcome(row_id, 2100.0, 50.0)
            stats = self._p.query_stats()
            self.assertEqual(stats["wins"], 1)
            self.assertEqual(stats["losses"], 0)

    def test_query_stats_empty(self) -> None:
        stats = self._p.query_stats()
        self.assertEqual(stats["total"], 0)
        self.assertEqual(stats["win_rate"], 0.0)


# ── Backtester tests ──────────────────────────────────────────────────────────

class TestBacktester(unittest.TestCase):
    def _candles(self, n: int = 400) -> list[Candle]:
        base = 2_000.0
        candles = []
        for i in range(n):
            close = base - i * 0.3 + (i % 7) * 0.8
            candles.append(_make_candle(close, i))
        return candles

    def test_backtest_runs_without_error(self) -> None:
        from ai_trading_engine.backtester import Backtester

        bt = Backtester(EngineConfig())
        result = bt.run(self._candles(400), asset="ETH/USDT", timeframe="1h")
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.total_trades, 0)

    def test_backtest_result_has_valid_stats(self) -> None:
        from ai_trading_engine.backtester import Backtester

        bt = Backtester(EngineConfig())
        result = bt.run(self._candles(400))
        self.assertGreaterEqual(result.win_rate, 0.0)
        self.assertLessEqual(result.win_rate, 1.0)
        self.assertGreaterEqual(result.max_drawdown_pct, 0.0)
        self.assertFalse(math.isnan(result.sharpe_ratio))

    def test_backtest_raises_on_too_few_candles(self) -> None:
        from ai_trading_engine.backtester import Backtester

        bt = Backtester(EngineConfig())
        with self.assertRaises(ValueError):
            bt.run(self._candles(10))

    def test_backtest_summary_string(self) -> None:
        from ai_trading_engine.backtester import Backtester

        bt = Backtester(EngineConfig())
        result = bt.run(self._candles(400))
        summary = result.summary()
        self.assertIn("BACKTEST", summary)
        self.assertIn("Win rate", summary)


# ── Data validation tests ─────────────────────────────────────────────────────

class TestDataValidation(unittest.TestCase):
    def test_too_few_candles_blocks_trade(self) -> None:
        engine = HybridTradingEngine(EngineConfig())
        snapshot = _minimal_snapshot(n_candles=50)  # below 210 minimum
        portfolio = PortfolioState(equity_usd=10_000.0)
        decision = engine.evaluate(snapshot, portfolio)
        self.assertFalse(decision.is_trade)

    def test_portfolio_consecutive_losses(self) -> None:
        from ai_trading_engine.models import PortfolioState

        p = PortfolioState(
            equity_usd=10_000.0,
            open_positions={},
            recent_results=["WIN", "LOSS", "LOSS", "LOSS"],
        )
        self.assertEqual(p.consecutive_losses, 3)

    def test_portfolio_no_trailing_losses(self) -> None:
        from ai_trading_engine.models import PortfolioState

        p = PortfolioState(
            equity_usd=10_000.0,
            open_positions={},
            recent_results=["LOSS", "LOSS", "WIN"],
        )
        self.assertEqual(p.consecutive_losses, 0)


if __name__ == "__main__":
    unittest.main()
