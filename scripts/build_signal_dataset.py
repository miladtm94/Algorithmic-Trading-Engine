#!/usr/bin/env python
"""Build a sparse ML dataset from real engine candidate setups.

Each row represents a candidate emitted by the rule engine on a historical
bar. The label asks whether that exact setup would have resolved as a WIN
or LOSS over the configured lookahead path.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from build_dataset import load_candles

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)

HISTORICAL_DIR = Path(__file__).parent.parent / "data" / "historical"
FEATURES_DIR = Path(__file__).parent.parent / "data" / "features"


def _float_value(value: str | None) -> float:
    try:
        return float(value) if value not in (None, "") else 0.0
    except ValueError:
        return 0.0


def _family_stats(rows: list[dict[str, object]]) -> list[dict[str, float | int | str]]:
    grouped: dict[str, dict[str, float | int | str]] = defaultdict(
        lambda: {
            "family": "",
            "count": 0,
            "wins": 0,
            "losses": 0,
            "pnl_sum": 0.0,
            "net_r_sum": 0.0,
            "quality_sum": 0.0,
        }
    )
    for row in rows:
        family = str(row.get("setup_family", "UNKNOWN") or "UNKNOWN")
        stats = grouped[family]
        stats["family"] = family
        stats["count"] = int(stats["count"]) + 1
        if row.get("outcome") == "WIN":
            stats["wins"] = int(stats["wins"]) + 1
        else:
            stats["losses"] = int(stats["losses"]) + 1
        stats["pnl_sum"] = float(stats["pnl_sum"]) + _float_value(str(row.get("pnl_pct", 0.0)))
        stats["net_r_sum"] = float(stats["net_r_sum"]) + _float_value(str(row.get("net_r", 0.0)))
        stats["quality_sum"] = float(stats["quality_sum"]) + _float_value(str(row.get("setup_quality", 0.0)))

    summary: list[dict[str, float | int | str]] = []
    for family, stats in grouped.items():
        count = int(stats["count"])
        wins = int(stats["wins"])
        summary.append(
            {
                "family": family,
                "count": count,
                "wins": wins,
                "losses": int(stats["losses"]),
                "win_rate": wins / count if count else 0.0,
                "avg_pnl_pct": float(stats["pnl_sum"]) / count if count else 0.0,
                "avg_net_r": float(stats["net_r_sum"]) / count if count else 0.0,
                "avg_setup_quality": float(stats["quality_sum"]) / count if count else 0.0,
            }
        )
    return sorted(summary, key=lambda item: int(item["count"]), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sparse signal-selection dataset")
    parser.add_argument("--asset", default=os.getenv("DEFAULT_ASSET", "ETH/USDT"))
    parser.add_argument("--timeframe", default=os.getenv("DEFAULT_TIMEFRAME", "1h"))
    parser.add_argument("--window-size", type=int, default=220)
    parser.add_argument("--lookahead", type=int, default=24)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--min-profit-pct", type=float, default=0.0)
    parser.add_argument(
        "--min-confluence",
        type=float,
        default=55.0,
        help="Historical candidate gate before ML selection. Lower = more candidates.",
    )
    args = parser.parse_args()

    from ai_trading_engine.config import EngineConfig  # noqa: PLC0415
    from ai_trading_engine.signal_learning import (  # noqa: PLC0415
        SIGNAL_FEATURE_NAMES,
        build_signal_rows,
    )

    safe_asset = args.asset.replace("/", "_")
    candle_path = HISTORICAL_DIR / f"{safe_asset}_{args.timeframe}.csv"
    if not candle_path.exists():
        print(f"ERROR: {candle_path} not found.")
        print(
            f"  Run first: python scripts/fetch_history.py --asset {args.asset} --timeframe {args.timeframe}"
        )
        sys.exit(1)

    print(f"Loading candles from {candle_path} ...")
    candles = load_candles(candle_path)
    print(
        f"  {len(candles):,} candles  ({candles[0].timestamp.date()} → {candles[-1].timestamp.date()})"
    )

    print(
        "Building sparse candidate dataset "
        f"(lookahead={args.lookahead}, min_confluence={args.min_confluence}) ..."
    )
    rows, summary = build_signal_rows(
        candles,
        asset=args.asset,
        timeframe=args.timeframe,
        config=EngineConfig(),
        window_size=args.window_size,
        lookahead=args.lookahead,
        fee_bps=args.fee_bps,
        min_profit_pct=args.min_profit_pct,
        min_confluence=args.min_confluence,
    )
    if not rows:
        print("\nERROR: No candidate rows generated.")
        print("  Try lowering --min-confluence or fetching more history.")
        sys.exit(1)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEATURES_DIR / f"{safe_asset}_{args.timeframe}_signal_dataset.csv"
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
        "setup_family",
        "setup_quality",
        "max_hold_bars",
        "reference_level",
        "signal_score",
        "regime",
        "strategy",
        "regime_confidence_raw",
        "reason_count_raw",
        "engine_passed_threshold",
        *SIGNAL_FEATURE_NAMES,
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Candidate rows:   {summary.rows:,}")
    print(
        f"  WIN / LOSS:       {summary.wins:,} / {summary.losses:,} ({summary.win_rate:.1%} win rate)"
    )
    print(f"  LONG / SHORT:     {summary.long_rows:,} / {summary.short_rows:,}")
    print(f"  Engine-grade rows: {summary.engine_threshold_rows:,} at default threshold 75")
    print("  Family breakdown:")
    for stats in _family_stats(rows):
        print(
            f"    {stats['family']}: "
            f"{int(stats['count']):4d} rows, "
            f"{float(stats['win_rate']):.1%} win rate, "
            f"{float(stats['avg_pnl_pct']):+.2%} avg PnL, "
            f"{float(stats['avg_net_r']):+.2f} avg net R, "
            f"{float(stats['avg_setup_quality']):.1f} avg quality"
        )
    print(f"\n  Wrote {out_path}\n")


if __name__ == "__main__":
    main()
