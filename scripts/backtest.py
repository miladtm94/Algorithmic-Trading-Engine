#!/usr/bin/env python
"""Backtest script.

Demo mode (default): replays generated candle data.
Live mode (--live):  fetches real OHLCV history from the exchange.

Examples:
  python scripts/backtest.py                          # demo data
  python scripts/backtest.py --asset BTC/USDT         # different asset (demo)
  python scripts/backtest.py --live --candles 1000    # real data from exchange
  python scripts/backtest.py --live --exchange bybit  # different exchange
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

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

from ai_trading_engine.backtester import Backtester  # noqa: E402
from ai_trading_engine.config import EngineConfig  # noqa: E402
from ai_trading_engine.demo_data import build_demo_snapshot  # noqa: E402


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
    args = parser.parse_args()

    cfg = EngineConfig()
    cfg.confluence_threshold = args.confluence

    if args.live:
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
        risk_per_trade=args.risk,
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


if __name__ == "__main__":
    main()
