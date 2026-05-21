"""Continuous trading runner.

Supports three operating modes:
  demo   — uses generated candle data; no exchange needed, no money
  paper  — live exchange data, simulated order execution
  live   — live exchange data, real order execution (REAL MONEY)

The main loop:
  1. Fetch latest market snapshot
  2. Update open paper positions (if any) against current price
  3. Evaluate the engine
  4. Place order if a trade signal is generated
  5. Save decision to database
  6. Alert via Telegram
  7. Sleep until next candle close
"""
from __future__ import annotations

import logging
import signal as _signal
import time
from datetime import UTC, datetime
from typing import Literal

from .broker import LiveBroker, Order, PaperBroker
from .config import EngineConfig
from .engine import HybridTradingEngine
from .formatters import format_decision
from .models import Direction, EngineDecision, PortfolioState
from .persistence import init_db, save_decision, save_outcome

logger = logging.getLogger(__name__)

_TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1_800,
    "1h": 3_600,
    "4h": 14_400,
    "1d": 86_400,
}


class TradingRunner:
    def __init__(
        self,
        asset: str = "ETH/USDT",
        timeframe: str = "1h",
        mode: str = "demo",
        exchange_id: str = "binance",
        api_key: str = "",
        api_secret: str = "",
        sandbox: bool = False,
        initial_equity: float = 10_000.0,
        engine_config: EngineConfig | None = None,
        telegram_bot=None,
        telegram_chat_id: str = "",
    ) -> None:
        if mode not in ("demo", "paper", "live"):
            raise ValueError(f"Invalid mode {mode!r}. Choose: demo, paper, live")

        self._asset = asset
        self._timeframe = timeframe
        self._mode = mode
        self._telegram = telegram_bot
        self._chat_id = telegram_chat_id
        self._running = False
        self._cfg = engine_config or EngineConfig()
        self._engine = HybridTradingEngine(self._cfg)

        # Map each decision_id → Order so we can record outcomes
        self._pending: dict[int, Order] = {}

        if mode != "demo":
            from .market_data import MarketDataFetcher  # noqa: PLC0415

            self._fetcher = MarketDataFetcher(exchange_id, api_key, api_secret, sandbox)
            if mode == "paper":
                self._broker: PaperBroker | LiveBroker = PaperBroker(initial_equity)
            else:
                self._broker = LiveBroker(exchange_id, api_key, api_secret, sandbox)
            init_db()

        _signal.signal(_signal.SIGTERM, self._on_shutdown)
        _signal.signal(_signal.SIGINT, self._on_shutdown)

        logger.info(
            "TradingRunner ready: asset=%s timeframe=%s mode=%s",
            asset, timeframe, mode,
        )

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def _on_shutdown(self, signum, frame) -> None:  # noqa: ANN001
        logger.info("Shutdown signal received, stopping after current cycle...")
        self._running = False

    # ── Portfolio state ───────────────────────────────────────────────────────

    def _portfolio(self) -> PortfolioState:
        if self._mode == "demo":
            from .demo_data import build_demo_portfolio  # noqa: PLC0415

            return build_demo_portfolio()

        broker = self._broker
        equity = broker.equity
        open_pos: dict[str, Direction] = {o.asset: o.direction for o in broker.open_orders}

        recent_results: list[Literal["WIN", "LOSS"]] = []
        recent_stamps: list[str] = []
        if isinstance(broker, PaperBroker):
            for order in broker.trade_history[-30:]:
                recent_results.append("WIN" if (order.pnl_usd or 0) > 0 else "LOSS")
                recent_stamps.append(order.created_at.isoformat())
            for order in broker.open_orders:
                recent_stamps.append(order.created_at.isoformat())

        return PortfolioState(
            equity_usd=equity,
            open_positions=open_pos,
            recent_results=recent_results,
            recent_trade_timestamps=recent_stamps,
        )

    # ── Single cycle ──────────────────────────────────────────────────────────

    def run_once(self) -> EngineDecision:
        if self._mode == "demo":
            from .demo_data import build_demo_snapshot  # noqa: PLC0415

            snapshot = build_demo_snapshot(self._asset)
            current_price = snapshot.candles[-1].close
        else:
            snapshot = self._fetcher.fetch_snapshot(self._asset, self._timeframe)
            current_price = snapshot.candles[-1].close

        # Update paper positions
        if self._mode == "paper" and isinstance(self._broker, PaperBroker):
            closed_orders = self._broker.update_prices(self._asset, current_price)
            for order in closed_orders:
                # Find matching pending decision and record outcome
                for dec_id, tracked_order in list(self._pending.items()):
                    if tracked_order.id == order.id:
                        save_outcome(dec_id, order.fill_price or current_price, order.pnl_usd or 0)
                        del self._pending[dec_id]
                        break

        portfolio = self._portfolio()
        decision = self._engine.evaluate(snapshot, portfolio)

        # Print & log
        output = format_decision(decision)
        print(output)
        logger.info("Cycle complete: %s", "TRADE" if decision.is_trade else "NO TRADE")

        # Persist
        if self._mode != "demo":
            dec_id = save_decision(decision)
            if decision.is_trade:
                order = self._broker.place_order(decision.signal)  # type: ignore[arg-type]
                self._pending[dec_id] = order

        # Telegram
        if self._telegram and self._chat_id and decision.is_trade:
            try:
                self._telegram.send(self._chat_id, output)
            except Exception as exc:
                logger.warning("Telegram send failed: %s", exc)

        return decision

    # ── Continuous loop ───────────────────────────────────────────────────────

    def run(self) -> None:
        interval = _TIMEFRAME_SECONDS.get(self._timeframe, 3_600)
        self._running = True
        logger.info(
            "Starting live loop: asset=%s timeframe=%s mode=%s interval=%ds",
            self._asset, self._timeframe, self._mode, interval,
        )

        consecutive_errors = 0

        while self._running:
            cycle_start = time.monotonic()
            try:
                self.run_once()
                consecutive_errors = 0
            except KeyboardInterrupt:
                break
            except Exception as exc:
                consecutive_errors += 1
                logger.exception("Error in trading cycle (#%d): %s", consecutive_errors, exc)
                if consecutive_errors >= 5:
                    logger.critical("5 consecutive errors — stopping runner for safety.")
                    break
                # Back off proportionally but cap at 5 minutes
                time.sleep(min(60 * consecutive_errors, 300))
                continue

            elapsed = time.monotonic() - cycle_start
            sleep_time = max(0.0, interval - elapsed)
            logger.info("Sleeping %.1fs until next candle close", sleep_time)
            if self._running:
                time.sleep(sleep_time)

        self._shutdown_report()

    # ── Shutdown summary ──────────────────────────────────────────────────────

    def _shutdown_report(self) -> None:
        logger.info("Runner stopped.")
        if self._mode == "paper" and isinstance(self._broker, PaperBroker):
            stats = self._broker.stats()
            logger.info("Final paper stats: %s", stats)
            print(f"\n{'─'*50}")
            print(f"  PAPER TRADING SESSION SUMMARY  ({datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')})")
            print(f"{'─'*50}")
            print(f"  {stats}")
            print(f"{'─'*50}\n")
