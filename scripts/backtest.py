#!/usr/bin/env python
"""Backtest script.

Demo mode (default): replays generated candle data.
Live mode (--live):  fetches real OHLCV history from the exchange.
History mode:        replays a locally cached CSV (--history-csv PATH), e.g.
                     the `data/historical/ETH_USDT_1h.csv` fetched by
                     scripts/fetch_history.py.

Examples:
  python scripts/backtest.py                                        # demo data
  python scripts/backtest.py --asset BTC/USDT                       # different asset (demo)
  python scripts/backtest.py --live --candles 1000                  # real data from exchange
  python scripts/backtest.py --history-csv data/historical/ETH_USDT_1h.csv --preset strict
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Allow running directly from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)

from build_dataset import load_candles as _load_history_csv  # noqa: E402

from ai_trading_engine.backtester import Backtester  # noqa: E402
from ai_trading_engine.config import EngineConfig  # noqa: E402
from ai_trading_engine.demo_data import build_demo_snapshot  # noqa: E402


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _apply_preset(cfg: EngineConfig, preset: str) -> None:
    if preset == "strict":
        cfg.confluence_threshold = 55.0
        cfg.allowed_setup_families = frozenset({"RANGE_REJECTION_MEAN_REVERSION"})
        cfg.max_trades_per_iso_week = 1
        cfg.risk.risk_per_trade_pct = 0.005
        cfg.risk.min_rr = 1.0
        cfg.risk.kill_switch_after_losses = 5


def _demo_candles(asset: str, count: int):
    snapshot = build_demo_snapshot(asset)
    candles = snapshot.candles
    # Repeat the window to get a larger dataset for meaningful stats
    if len(candles) < count:
        factor = (count // len(candles)) + 1
        base = list(candles)
        extended = []
        for _ in range(factor):
            extended.extend(base)
        candles = extended[:count]
    return candles[:count]


def _live_candles(asset: str, timeframe: str, count: int, exchange_id: str):
    from ai_trading_engine.market_data import MarketDataFetcher  # noqa: E402

    api_key = os.getenv("EXCHANGE_API_KEY", "")
    api_secret = os.getenv("EXCHANGE_API_SECRET", "")
    sandbox = os.getenv("EXCHANGE_SANDBOX", "").lower() in ("1", "true", "yes")

    fetcher = MarketDataFetcher(exchange_id, api_key, api_secret, sandbox)
    print(f"Fetching {count} candles of {asset} [{timeframe}] from {exchange_id}...")
    candles = fetcher.fetch_historical_candles(asset, timeframe, limit=count)
    print(f"  Retrieved {len(candles)} candles.")
    return candles


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Trading Engine — Backtester")
    parser.add_argument("--asset", default=os.getenv("DEFAULT_ASSET", "ETH/USDT"))
    parser.add_argument("--timeframe", default=os.getenv("DEFAULT_TIMEFRAME", "1h"))
    parser.add_argument("--candles", type=int, default=500, help="Number of candles to backtest over")
    parser.add_argument("--equity", type=float, default=10_000.0, help="Starting equity in USD")
    parser.add_argument("--risk", type=float, default=0.01, help="Risk per trade as fraction of equity")
    parser.add_argument("--live", action="store_true", help="Fetch real candle data from exchange")
    parser.add_argument("--exchange", default=os.getenv("EXCHANGE", "binance"))
    parser.add_argument("--confluence", type=float, default=75.0, help="Confluence threshold (0-100)")
    parser.add_argument(
        "--history-csv",
        type=Path,
        default=None,
        help="Backtest a locally cached OHLCV CSV (e.g. data/historical/ETH_USDT_1h.csv).",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Optional ISO date/datetime to filter the history window, e.g. 2024-01-01.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Optional ISO date/datetime to filter the history window, e.g. 2026-04-19.",
    )
    parser.add_argument(
        "--preset",
        choices=["default", "strict"],
        default="default",
        help=(
            "Engine policy preset. 'strict' restricts to RANGE_REJECTION_MEAN_REVERSION, "
            "caps at 1 trade per ISO week, sets confluence floor 55, halves per-trade risk."
        ),
    )
    parser.add_argument(
        "--trades-csv",
        type=Path,
        default=None,
        help="Optional output CSV that lists every closed backtest trade.",
    )
    args = parser.parse_args()

    cfg = EngineConfig()
    cfg.confluence_threshold = args.confluence
    _apply_preset(cfg, args.preset)

    if args.history_csv is not None:
        if not args.history_csv.exists():
            print(f"ERROR: {args.history_csv} not found.")
            sys.exit(1)
        print(f"Loading history from {args.history_csv} ...")
        candles = _load_history_csv(args.history_csv)
        start_dt = _parse_date(args.start)
        end_dt = _parse_date(args.end)
        if start_dt is not None:
            candles = [c for c in candles if c.timestamp >= start_dt]
        if end_dt is not None:
            candles = [c for c in candles if c.timestamp <= end_dt]
        print(f"  {len(candles):,} candles after filtering")
    elif args.live:
        candles = _live_candles(args.asset, args.timeframe, args.candles, args.exchange)
    else:
        print(f"Using generated demo data ({args.candles} candles)...")
        candles = _demo_candles(args.asset, args.candles)

    if len(candles) < 222:
        print(f"ERROR: Need at least 222 candles, got {len(candles)}. Use --candles 500 or more.")
        sys.exit(1)

    bt = Backtester(cfg)
    result = bt.run(
        candles,
        asset=args.asset,
        timeframe=args.timeframe,
        initial_equity=args.equity,
        risk_per_trade=args.risk if args.preset != "strict" else cfg.risk.risk_per_trade_pct,
    )

    print()
    print(result.summary())

    if result.total_trades > 0:
        print("\n  Recent trades (last 10):")
        print(f"  {'#':>3}  {'Dir':6}  {'Entry':>10}  {'Exit':>10}  {'PnL%':>8}  Outcome")
        print(f"  {'─'*54}")
        for idx, t in enumerate(result.trades[-10:], 1):
            pnl_str = f"{(t.pnl_pct or 0)*100:+.2f}%" if t.pnl_pct is not None else "    ---"
            exit_str = f"{t.exit_price:.4f}" if t.exit_price else "      ---"
            print(
                f"  {idx:>3}  {t.direction:6}  {t.entry_price:>10.4f}  {exit_str:>10}  "
                f"{pnl_str:>8}  {t.outcome or '---'}"
            )

    print()
    note = (
        "\n  NOTE: Backtest results reflect behaviour on historical data only.\n"
        "  They do NOT guarantee future performance. Always validate with\n"
        "  paper trading before committing real capital.\n"
    )
    print(note)

    if args.trades_csv is not None and result.trades:
        import csv as _csv  # noqa: PLC0415

        args.trades_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.trades_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = _csv.writer(handle)
            writer.writerow(
                [
                    "entry_idx",
                    "exit_idx",
                    "bars_held",
                    "direction",
                    "setup_family",
                    "regime",
                    "entry_price",
                    "stop_loss",
                    "take_profit",
                    "exit_price",
                    "pnl_pct",
                    "pnl_usd",
                    "outcome",
                    "exit_reason",
                    "confluence_score",
                ]
            )
            for trade in result.trades:
                bars_held = (
                    trade.exit_idx - trade.entry_idx
                    if trade.exit_idx is not None and trade.entry_idx is not None
                    else ""
                )
                writer.writerow(
                    [
                        trade.entry_idx,
                        trade.exit_idx if trade.exit_idx is not None else "",
                        bars_held,
                        trade.direction,
                        trade.setup_family,
                        trade.regime,
                        f"{trade.entry_price:.6f}",
                        f"{trade.stop_loss:.6f}",
                        f"{trade.take_profit:.6f}",
                        f"{trade.exit_price:.6f}" if trade.exit_price is not None else "",
                        f"{trade.pnl_pct:.6f}" if trade.pnl_pct is not None else "",
                        f"{trade.pnl_usd:.2f}" if trade.pnl_usd is not None else "",
                        trade.outcome or "",
                        trade.exit_reason or "",
                        f"{trade.confluence_score:.2f}",
                    ]
                )
        print(f"  Wrote trade log -> {args.trades_csv}")


if __name__ == "__main__":
    main()
