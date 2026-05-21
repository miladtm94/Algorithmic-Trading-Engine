#!/usr/bin/env python
"""Build the ML feature + label dataset from downloaded historical candles.

Default mode builds a dense research dataset:
  - Every candle after indicator warmup becomes one LONG and one SHORT setup
  - Each setup is labelled WIN/LOSS by walking future candles
  - WIN means TP was reached before SL, or horizon exit was profitable after costs
  - Features are shared with live ML scoring

Output: data/features/<ASSET>_<TIMEFRAME>_dataset.csv

Usage:
  python scripts/build_dataset.py --asset ETH/USDT --timeframe 1h
  python scripts/build_dataset.py --asset ETH/USDT --timeframe 1h --lookahead 36 --stop-atr 1.2
  python scripts/build_dataset.py --asset ETH/USDT --timeframe 1h --mode engine --confluence 70
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from datetime import datetime
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
    parser.add_argument("--asset", default=os.getenv("DEFAULT_ASSET", "ETH/USDT"))
    parser.add_argument("--timeframe", default=os.getenv("DEFAULT_TIMEFRAME", "1h"))
    parser.add_argument(
        "--mode",
        choices=["opportunity", "engine"],
        default="opportunity",
        help="opportunity = dense LONG/SHORT examples; engine = old sparse backtester signals",
    )
    parser.add_argument(
        "--confluence", type=float, default=75.0,
        help="Only used with --mode engine. Lower = more engine signals."
    )
    parser.add_argument("--window-size", type=int, default=220)
    parser.add_argument(
        "--lookahead",
        type=int,
        default=24,
        help="Future candles used for labels. 24 on 1h means one day.",
    )
    parser.add_argument(
        "--stop-atr",
        type=float,
        default=1.5,
        help="Stop distance in ATR multiples for opportunity labels.",
    )
    parser.add_argument(
        "--reward-risk",
        type=float,
        default=2.0,
        help="Take-profit distance as a multiple of stop risk.",
    )
    parser.add_argument(
        "--fee-bps",
        type=float,
        default=10.0,
        help="Round-trip cost removed from label PnL, in basis points.",
    )
    parser.add_argument(
        "--min-profit-pct",
        type=float,
        default=0.0,
        help="Minimum net PnL required for a WIN label, e.g. 0.002 = 0.2%%.",
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

    from ai_trading_engine.config import EngineConfig  # noqa: PLC0415
    from ai_trading_engine.feature_extractor import FEATURE_NAMES  # noqa: PLC0415

    cfg = EngineConfig()

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEATURES_DIR / f"{safe_asset}_{args.timeframe}_dataset.csv"

    if args.mode == "engine":
        from ai_trading_engine.backtester import Backtester  # noqa: PLC0415

        cfg.confluence_threshold = args.confluence
        bt = Backtester(cfg)

        print(f"Running sparse engine backtester (confluence ≥ {args.confluence}%) ...")
        result = bt.run(
            candles,
            asset=args.asset,
            timeframe=args.timeframe,
            window_size=args.window_size,
        )

        labeled = [t for t in result.trades if t.outcome is not None and t.features]
        if not labeled:
            print("\nERROR: No labeled signals generated.")
            print("  Try: --confluence 65  (lower threshold = more signals)")
            sys.exit(1)

        wins = sum(1 for t in labeled if t.outcome == "WIN")
        losses = len(labeled) - wins

        print(f"  {result.total_candles:,} candles processed  →  {len(labeled)} labeled signals")
        print(f"  WIN: {wins} ({wins/len(labeled):.1%})   LOSS: {losses} ({losses/len(labeled):.1%})")

        fieldnames = ["timestamp", "asset", "timeframe", "outcome"] + FEATURE_NAMES
        rows: list[dict[str, object]] = []
        for trade in labeled:
            ts = candles[trade.entry_idx].timestamp.isoformat() if trade.entry_idx < len(candles) else ""
            row: dict[str, object] = {
                "timestamp": ts,
                "asset": trade.asset,
                "timeframe": args.timeframe,
                "outcome": trade.outcome,
            }
            for name in FEATURE_NAMES:
                row[name] = round(trade.features.get(name, 0.0), 8)
            rows.append(row)
    else:
        from ai_trading_engine.dataset import build_opportunity_rows  # noqa: PLC0415

        print(
            "Building dense opportunity dataset "
            f"(lookahead={args.lookahead}, stop={args.stop_atr} ATR, RR={args.reward_risk}) ..."
        )
        rows, summary = build_opportunity_rows(
            candles,
            asset=args.asset,
            timeframe=args.timeframe,
            config=cfg,
            window_size=args.window_size,
            lookahead=args.lookahead,
            stop_atr=args.stop_atr,
            reward_risk=args.reward_risk,
            fee_bps=args.fee_bps,
            min_profit_pct=args.min_profit_pct,
        )
        print(f"  {len(candles):,} candles processed  →  {summary.rows:,} labeled opportunities")
        print(
            f"  WIN: {summary.wins:,} ({summary.win_rate:.1%})   "
            f"LOSS: {summary.losses:,} ({1 - summary.win_rate:.1%})"
        )
        print(
            f"  LONG win rate: {summary.long_win_rate:.1%}   "
            f"SHORT win rate: {summary.short_win_rate:.1%}"
        )

        fieldnames = [
            "timestamp",
            "asset",
            "timeframe",
            "side",
            "outcome",
            "exit_reason",
            "entry",
            "stop_loss",
            "take_profit",
            "exit_price",
            "bars_held",
            "pnl_pct",
            "risk_pct",
            "net_return_pct",
            "net_r",
            "max_favorable_pct",
            "max_adverse_pct",
            "max_favorable_r",
            "max_adverse_r",
            "bars_to_target",
            "bars_to_stop",
            "r_bucket",
            "meta_label",
        ] + FEATURE_NAMES

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Dataset saved → {out_path}")
    print(f"  Columns: timestamp + outcome + {len(FEATURE_NAMES)} features\n")


if __name__ == "__main__":
    main()
