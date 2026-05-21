#!/usr/bin/env python
"""Inspect which setup-family rules are blocking sparse candidate generation."""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from build_dataset import load_candles

HISTORICAL_DIR = Path(__file__).parent.parent / "data" / "historical"


def _trend_pullback_long_rejection(snapshot, indicators, structure) -> str | None:
    import ai_trading_engine.signal_generation as sg  # noqa: PLC0415

    candles = snapshot.candles
    last = candles[-1]
    entry = last.close
    atr = max(indicators.atr, entry * 0.003)
    pullback_window = candles[-8:] if len(candles) >= 8 else candles
    pullback_low = min(candle.low for candle in pullback_window)
    pullback_depth = entry - pullback_low
    touched_support = pullback_low <= indicators.ema20 + 1.0 * atr
    reclaimed_support = entry >= indicators.ema20 - 0.35 * atr or entry >= indicators.vwap - 0.2 * atr

    if not (
        indicators.ema20 > indicators.ema50 > indicators.ema200
        and indicators.macd_hist > 0
        and touched_support
        and reclaimed_support
    ):
        return "trend_stack_or_reclaim"
    if pullback_depth < 0.15 * atr or pullback_depth > 3.25 * atr:
        return "pullback_depth_band"
    if pullback_low < indicators.ema50 - 1.25 * atr:
        return "pullback_below_ema50_buffer"
    if not (
        last.close >= last.open * 0.998
        and (
            sg._lower_wick(last) > sg._body(last) * 0.15
            or last.close >= indicators.vwap
            or last.close >= indicators.ema20 - 0.15 * atr
        )
    ):
        return "confirmation_candle"

    stop = min(pullback_low - 0.2 * atr, structure.nearest_support - 0.1 * atr)
    risk = entry - stop
    if risk <= 0:
        return "non_positive_risk"
    if (
        not structure.broke_resistance
        and structure.nearest_resistance > entry
        and (structure.nearest_resistance - entry) < 0.8 * risk
    ):
        return "resistance_too_close"
    return None


def _trend_pullback_short_rejection(snapshot, indicators, structure) -> str | None:
    import ai_trading_engine.signal_generation as sg  # noqa: PLC0415

    candles = snapshot.candles
    last = candles[-1]
    entry = last.close
    atr = max(indicators.atr, entry * 0.003)
    pullback_window = candles[-8:] if len(candles) >= 8 else candles
    pullback_high = max(candle.high for candle in pullback_window)
    pullback_depth = pullback_high - entry
    touched_resistance = pullback_high >= indicators.ema20 - 1.0 * atr
    reclaimed_resistance = entry <= indicators.ema20 + 0.35 * atr or entry <= indicators.vwap + 0.2 * atr

    if not (
        indicators.ema20 < indicators.ema50 < indicators.ema200
        and indicators.macd_hist < 0
        and touched_resistance
        and reclaimed_resistance
    ):
        return "trend_stack_or_reclaim"
    if pullback_depth < 0.15 * atr or pullback_depth > 3.25 * atr:
        return "pullback_depth_band"
    if pullback_high > indicators.ema50 + 1.25 * atr:
        return "pullback_above_ema50_buffer"
    if not (
        last.close <= last.open * 1.002
        and (
            sg._upper_wick(last) > sg._body(last) * 0.15
            or last.close <= indicators.vwap
            or last.close <= indicators.ema20 + 0.15 * atr
        )
    ):
        return "confirmation_candle"

    stop = max(pullback_high + 0.2 * atr, structure.nearest_resistance + 0.1 * atr)
    risk = stop - entry
    if risk <= 0:
        return "non_positive_risk"
    if (
        not structure.broke_support
        and structure.nearest_support < entry
        and (entry - structure.nearest_support) < 0.8 * risk
    ):
        return "support_too_close"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose sparse setup-family generation")
    parser.add_argument("--asset", default=os.getenv("DEFAULT_ASSET", "ETH/USDT"))
    parser.add_argument("--timeframe", default=os.getenv("DEFAULT_TIMEFRAME", "1h"))
    parser.add_argument("--window-size", type=int, default=220)
    parser.add_argument("--lookahead", type=int, default=24)
    parser.add_argument("--min-confluence", type=float, default=55.0)
    parser.add_argument(
        "--family",
        choices=["trend_pullback"],
        default="trend_pullback",
        help="Diagnostic family focus. Currently only trend_pullback is supported.",
    )
    args = parser.parse_args()

    import ai_trading_engine.signal_generation as sg  # noqa: PLC0415
    from ai_trading_engine.config import EngineConfig  # noqa: PLC0415
    from ai_trading_engine.confluence import score_candidate  # noqa: PLC0415
    from ai_trading_engine.dataset import build_research_snapshot  # noqa: PLC0415
    from ai_trading_engine.indicators import compute_indicators  # noqa: PLC0415
    from ai_trading_engine.regime import classify_regime  # noqa: PLC0415

    safe_asset = args.asset.replace("/", "_")
    candle_path = HISTORICAL_DIR / f"{safe_asset}_{args.timeframe}.csv"
    if not candle_path.exists():
        print(f"ERROR: {candle_path} not found.")
        print(
            f"  Run first: python scripts/fetch_history.py --asset {args.asset} --timeframe {args.timeframe}"
        )
        sys.exit(1)

    candles = load_candles(candle_path)
    cfg = EngineConfig()
    max_entry_index = len(candles) - args.lookahead - 1
    if len(candles) < args.window_size + args.lookahead + 1:
        print(
            f"ERROR: Need at least {args.window_size + args.lookahead + 1} candles, got {len(candles)}."
        )
        sys.exit(1)

    stats = Counter()
    rejections_long = Counter()
    rejections_short = Counter()

    for idx in range(args.window_size, max_entry_index + 1):
        window = candles[idx - args.window_size : idx + 1]
        snapshot = build_research_snapshot(window, args.asset, args.timeframe)
        indicators = compute_indicators(window, snapshot.order_book)
        regime = classify_regime(indicators, cfg.regime)
        structure = sg.read_structure(snapshot)

        if regime.regime == "TRENDING_BULLISH":
            stats["trend_bullish_bars"] += 1
            trend_pullback = sg._build_trend_pullback_long(snapshot, regime, indicators, structure)
            breakout_retest = sg._build_breakout_retest_long(snapshot, regime, indicators, structure)
            rejection = _trend_pullback_long_rejection(snapshot, indicators, structure)

            if trend_pullback is not None:
                stats["trend_pullback_raw_bull"] += 1
                if score_candidate(
                    trend_pullback,
                    snapshot,
                    cfg.confluence_weights,
                ).total_score >= args.min_confluence:
                    stats["trend_pullback_confluence_bull"] += 1
            elif rejection is not None:
                rejections_long[rejection] += 1

            if breakout_retest is not None:
                stats["breakout_retest_raw_bull"] += 1

            if trend_pullback is not None and breakout_retest is not None:
                stats["both_candidates_bull"] += 1
                chosen = sg._choose_candidate([breakout_retest, trend_pullback])
                if chosen is not None and chosen.setup_family == sg.FAMILY_TREND_PULLBACK:
                    stats["trend_pullback_selected_bull"] += 1
                else:
                    stats["breakout_retest_selected_over_trend_pullback_bull"] += 1
            elif trend_pullback is not None:
                stats["trend_pullback_only_bull"] += 1
            elif breakout_retest is not None:
                stats["breakout_retest_only_bull"] += 1

        elif regime.regime == "TRENDING_BEARISH":
            stats["trend_bearish_bars"] += 1
            trend_pullback = sg._build_trend_pullback_short(snapshot, regime, indicators, structure)
            breakout_retest = sg._build_breakout_retest_short(snapshot, regime, indicators, structure)
            rejection = _trend_pullback_short_rejection(snapshot, indicators, structure)

            if trend_pullback is not None:
                stats["trend_pullback_raw_bear"] += 1
                if score_candidate(
                    trend_pullback,
                    snapshot,
                    cfg.confluence_weights,
                ).total_score >= args.min_confluence:
                    stats["trend_pullback_confluence_bear"] += 1
            elif rejection is not None:
                rejections_short[rejection] += 1

            if breakout_retest is not None:
                stats["breakout_retest_raw_bear"] += 1

            if trend_pullback is not None and breakout_retest is not None:
                stats["both_candidates_bear"] += 1
                chosen = sg._choose_candidate([breakout_retest, trend_pullback])
                if chosen is not None and chosen.setup_family == sg.FAMILY_TREND_PULLBACK:
                    stats["trend_pullback_selected_bear"] += 1
                else:
                    stats["breakout_retest_selected_over_trend_pullback_bear"] += 1
            elif trend_pullback is not None:
                stats["trend_pullback_only_bear"] += 1
            elif breakout_retest is not None:
                stats["breakout_retest_only_bear"] += 1

    total_trending = stats["trend_bullish_bars"] + stats["trend_bearish_bars"]
    total_trend_pullback_raw = stats["trend_pullback_raw_bull"] + stats["trend_pullback_raw_bear"]
    total_trend_pullback_confluence = (
        stats["trend_pullback_confluence_bull"] + stats["trend_pullback_confluence_bear"]
    )
    total_breakout_retest_raw = (
        stats["breakout_retest_raw_bull"] + stats["breakout_retest_raw_bear"]
    )
    total_selected = stats["trend_pullback_selected_bull"] + stats["trend_pullback_selected_bear"]

    print(f"\nSignal-family diagnosis for {args.asset} [{args.timeframe}]")
    print(f"History: {candles[0].timestamp.date()} -> {candles[-1].timestamp.date()}")
    print(f"Window size: {args.window_size}")
    print(f"Lookahead: {args.lookahead}")
    print(f"Confluence gate: {args.min_confluence:.1f}")
    print()
    print("Trend regime coverage:")
    print(f"  Trending bullish bars: {stats['trend_bullish_bars']}")
    print(f"  Trending bearish bars: {stats['trend_bearish_bars']}")
    print(f"  Total trending bars:   {total_trending}")
    print()
    print("Raw candidate generation:")
    print(f"  Trend pullback raw:    {total_trend_pullback_raw}")
    print(f"  Breakout retest raw:   {total_breakout_retest_raw}")
    print(f"  Trend pullback @ gate: {total_trend_pullback_confluence}")
    print(f"  Trend pullback chosen: {total_selected}")
    print()
    print("Head-to-head selection:")
    print(f"  Both candidates bullish: {stats['both_candidates_bull']}")
    print(f"  Both candidates bearish: {stats['both_candidates_bear']}")
    print(
        "  Breakout retest selected over trend pullback: "
        f"{stats['breakout_retest_selected_over_trend_pullback_bull'] + stats['breakout_retest_selected_over_trend_pullback_bear']}"
    )
    print()
    print("Top trend-pullback rejection reasons (bullish):")
    for reason, count in rejections_long.most_common(8):
        print(f"  {reason}: {count}")
    print()
    print("Top trend-pullback rejection reasons (bearish):")
    for reason, count in rejections_short.most_common(8):
        print(f"  {reason}: {count}")
    print()
    if total_trend_pullback_raw == 0:
        print("Diagnosis:")
        print("  Trend pullback is not being suppressed by candidate priority.")
        print("  It is failing before raw candidate creation.")
        print("  The dominant blockers are the trend-stack/reclaim gate, pullback-depth band,")
        print("  confirmation candle, and nearby support/resistance spacing checks.")
        print()


if __name__ == "__main__":
    main()
