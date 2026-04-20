#!/usr/bin/env python
"""Build the ML feature + label dataset from downloaded historical candles.

Runs the backtester over the full candle history with a sliding window.
For each trade signal the engine generates, it captures:
  - The 24 normalised feature values (indicators, regime, confluence)
  - The forward-looking label: WIN if TP1 was hit before SL, else LOSS
  - The candle timestamp when the signal was generated

Output: data/features/<ASSET>_<TIMEFRAME>_dataset.csv

Usage:
  python scripts/build_dataset.py --asset ETH/USD --timeframe 1h
  python scripts/build_dataset.py --asset ETH/USD --timeframe 1h --confluence 70
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

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

HISTORICAL_DIR = Path(__file__).parent.parent / "data" / "historical"
FEATURES_DIR = Path(__file__).parent.parent / "data" / "features"


def load_candles(csv_path: Path):
    from ai_trading_engine.models import Candle  # noqa: PLC0415

    candles: list[Candle] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            candles.append(
                Candle(
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
    return candles


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ML dataset from historical candles via backtester"
    )
    parser.add_argument("--asset", default=os.getenv("DEFAULT_ASSET", "ETH/USD"))
    parser.add_argument("--timeframe", default=os.getenv("DEFAULT_TIMEFRAME", "1h"))
    parser.add_argument(
        "--confluence", type=float, default=75.0,
        help="Confluence threshold used by the engine (lower = more signals, larger dataset)"
    )
    args = parser.parse_args()

    safe_asset = args.asset.replace("/", "_")
    candle_path = HISTORICAL_DIR / f"{safe_asset}_{args.timeframe}.csv"
    if not candle_path.exists():
        print(f"ERROR: {candle_path} not found.")
        print(f"  Run first: python scripts/fetch_history.py --asset {args.asset} --timeframe {args.timeframe}")
        sys.exit(1)

    print(f"Loading candles from {candle_path} ...")
    candles = load_candles(candle_path)
    print(f"  {len(candles):,} candles  ({candles[0].timestamp.date()} → {candles[-1].timestamp.date()})")

    from ai_trading_engine.backtester import Backtester  # noqa: PLC0415
    from ai_trading_engine.config import EngineConfig  # noqa: PLC0415
    from ai_trading_engine.feature_extractor import FEATURE_NAMES  # noqa: PLC0415

    cfg = EngineConfig()
    cfg.confluence_threshold = args.confluence
    bt = Backtester(cfg)

    print(f"Running backtester (confluence ≥ {args.confluence}%) ...")
    result = bt.run(candles, asset=args.asset, timeframe=args.timeframe)

    labeled = [t for t in result.trades if t.outcome is not None and t.features]
    wins = sum(1 for t in labeled if t.outcome == "WIN")
    losses = len(labeled) - wins

    print(f"  {result.total_candles:,} candles processed  →  {len(labeled)} labeled signals")
    print(f"  WIN: {wins} ({wins/len(labeled):.1%})   LOSS: {losses} ({losses/len(labeled):.1%})")

    if not labeled:
        print("\nERROR: No labeled signals generated.")
        print("  Try: --confluence 65  (lower threshold = more signals)")
        sys.exit(1)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEATURES_DIR / f"{safe_asset}_{args.timeframe}_dataset.csv"

    fieldnames = ["timestamp", "asset", "timeframe", "outcome"] + FEATURE_NAMES
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in labeled:
            ts = candles[t.entry_idx].timestamp.isoformat() if t.entry_idx < len(candles) else ""
            row: dict = {
                "timestamp": ts,
                "asset": t.asset,
                "timeframe": args.timeframe,
                "outcome": t.outcome,
            }
            for k in FEATURE_NAMES:
                row[k] = round(t.features.get(k, 0.0), 6)
            writer.writerow(row)

    print(f"\n  Dataset saved → {out_path}")
    print(f"  Columns: timestamp + outcome + {len(FEATURE_NAMES)} features\n")


if __name__ == "__main__":
    main()
