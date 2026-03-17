from __future__ import annotations

import unittest

from ai_trading_engine import EngineConfig, HybridTradingEngine
from ai_trading_engine.demo_data import build_demo_portfolio, build_demo_snapshot
from ai_trading_engine.models import EventRisk


class EngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = HybridTradingEngine(EngineConfig())

    def test_generates_trade_in_demo_conditions(self) -> None:
        snapshot = build_demo_snapshot("ETH/USDT")
        portfolio = build_demo_portfolio()
        decision = self.engine.evaluate(snapshot, portfolio)
        self.assertTrue(decision.is_trade)
        self.assertIsNotNone(decision.signal)
        self.assertGreaterEqual(decision.signal.confluence.total_score, 75.0)  # type: ignore[union-attr]

    def test_pauses_on_high_impact_event(self) -> None:
        snapshot = build_demo_snapshot("ETH/USDT")
        snapshot.events = [
            EventRisk(name="FOMC Rate Decision", impact="HIGH", minutes_to_event=20)
        ]
        portfolio = build_demo_portfolio()
        decision = self.engine.evaluate(snapshot, portfolio)
        self.assertFalse(decision.is_trade)
        self.assertIn("Trading paused", decision.no_trade_reason or "")


if __name__ == "__main__":
    unittest.main()
