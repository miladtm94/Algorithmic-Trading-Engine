from __future__ import annotations

import unittest
from datetime import UTC, datetime

from ai_trading_engine.dataset import label_trade_path
from ai_trading_engine.feature_extractor import extract_candidate_features
from ai_trading_engine.models import (
    Candle,
    ConfluenceBreakdown,
    IndicatorSet,
    MarketSnapshot,
    OrderBookSnapshot,
    RegimeResult,
)
from ai_trading_engine.signal_diagnostics import (
    group_summaries,
    oracle_summary,
    oracle_top_k_rows,
    summarize_rows,
)
from ai_trading_engine.signal_generation import generate_candidate
from ai_trading_engine.signal_learning import effective_label_lookahead
from ai_trading_engine.validation import (
    selected_flags,
    temporal_train_validation_test_split,
    threshold_metrics,
)


def _candle(
    *,
    hour: int,
    open_: float,
    high: float,
    low: float,
    close: float,
) -> Candle:
    return Candle(
        timestamp=datetime(2024, 1, 1, hour, 0, tzinfo=UTC),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=1_000.0,
    )


def _indicator_set(
    *,
    ema20: float,
    ema50: float,
    ema200: float,
    vwap: float,
    rsi: float,
    macd_hist: float,
    atr: float,
    bb_upper: float,
    bb_middle: float,
    bb_lower: float,
    bb_width_pct: float,
    volume_ratio: float,
) -> IndicatorSet:
    last_close = bb_middle if bb_middle else 1.0
    return IndicatorSet(
        ema20=ema20,
        ema50=ema50,
        ema200=ema200,
        vwap=vwap,
        rsi=rsi,
        macd=macd_hist * last_close,
        macd_signal=macd_hist * last_close * 0.5,
        macd_hist=macd_hist,
        atr=atr,
        atr_pct=atr / last_close,
        bb_upper=bb_upper,
        bb_middle=bb_middle,
        bb_lower=bb_lower,
        bb_width_pct=bb_width_pct,
        avg_volume=1_000.0,
        volume_ratio=volume_ratio,
        order_book_imbalance=0.0,
    )


def _snapshot(
    candles: list[Candle],
    *,
    supports: list[float],
    resistances: list[float],
) -> MarketSnapshot:
    close = candles[-1].close
    return MarketSnapshot(
        asset="ETH/USDT",
        timeframe="1h",
        candles=candles,
        order_book=OrderBookSnapshot(
            bids=[(close - 0.1 * idx, 50_000.0) for idx in range(1, 6)],
            asks=[(close + 0.1 * idx, 50_000.0) for idx in range(1, 6)],
        ),
        spread_bps=4.0,
        depth_usd=1_000_000.0,
        support_levels=supports,
        resistance_levels=resistances,
        liquidation_clusters=[],
        sentiment_score=0.0,
        source_prices={"test": close},
        correlation_to_open_positions={},
        events=[],
    )


class TestResearchLabels(unittest.TestCase):
    def test_label_trade_path_emits_r_based_stats(self) -> None:
        result = label_trade_path(
            side="LONG",
            entry=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            future_candles=[_candle(hour=1, open_=100.0, high=111.0, low=99.0, close=108.0)],
            fee_bps=10.0,
        )

        self.assertEqual(result.outcome, "WIN")
        self.assertEqual(result.exit_reason, "TAKE_PROFIT")
        self.assertAlmostEqual(result.risk_pct, 0.05, places=8)
        self.assertAlmostEqual(result.net_return_pct, 0.099, places=8)
        self.assertAlmostEqual(result.net_r, 1.98, places=8)
        self.assertEqual(result.bars_to_target, 1)
        self.assertIsNone(result.bars_to_stop)
        self.assertEqual(result.r_bucket, "1R_TO_2R")
        self.assertEqual(result.meta_label, 1)

    def test_label_trade_path_keeps_stop_first_on_same_candle_touch(self) -> None:
        result = label_trade_path(
            side="LONG",
            entry=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            future_candles=[_candle(hour=1, open_=100.0, high=111.0, low=94.0, close=100.0)],
            fee_bps=10.0,
        )

        self.assertEqual(result.outcome, "LOSS")
        self.assertEqual(result.exit_reason, "STOP_LOSS")
        self.assertEqual(result.bars_to_target, 1)
        self.assertEqual(result.bars_to_stop, 1)
        self.assertLess(result.net_r, 0.0)
        self.assertEqual(result.meta_label, 0)


class TestValidationUtilities(unittest.TestCase):
    def test_weekly_selection_flags_match_threshold_summary(self) -> None:
        rows = [
            {"timestamp": "2024-01-01T00:00:00+00:00", "outcome": "WIN", "pnl_pct": "0.01", "signal_score": "70"},
            {"timestamp": "2024-01-02T00:00:00+00:00", "outcome": "LOSS", "pnl_pct": "-0.01", "signal_score": "65"},
            {"timestamp": "2024-01-03T00:00:00+00:00", "outcome": "WIN", "pnl_pct": "0.02", "signal_score": "60"},
            {"timestamp": "2024-01-08T00:00:00+00:00", "outcome": "LOSS", "pnl_pct": "-0.02", "signal_score": "75"},
            {"timestamp": "2024-01-09T00:00:00+00:00", "outcome": "WIN", "pnl_pct": "0.03", "signal_score": "68"},
            {"timestamp": "2024-01-10T00:00:00+00:00", "outcome": "LOSS", "pnl_pct": "-0.03", "signal_score": "50"},
        ]
        scores = [0.90, 0.80, 0.70, 0.95, 0.60, 0.50]

        stats = threshold_metrics(rows, scores, thresholds=[0.55], weekly_cap=2)
        flags = selected_flags(rows, scores, threshold=0.55, weekly_cap=2)

        self.assertEqual(sum(flags), int(stats[0]["count"]))
        self.assertEqual(sum(flags), 4)

    def test_temporal_split_applies_purge_gaps(self) -> None:
        def timestamp_for_index(index: int) -> str:
            month = (index - 1) // 28 + 1
            day = (index - 1) % 28 + 1
            return f"2024-{month:02d}-{day:02d}T00:00:00+00:00"

        rows = [
            {"timestamp": timestamp_for_index(index), "outcome": "WIN"}
            for index in range(1, 101)
        ]

        split = temporal_train_validation_test_split(
            rows,
            validation_pct=0.20,
            test_pct=0.20,
            purge_rows=5,
        )

        self.assertEqual(len(split.train_rows), 55)
        self.assertEqual(len(split.validation_rows), 15)
        self.assertEqual(len(split.test_rows), 20)
        self.assertEqual(split.train_rows[-1]["timestamp"], timestamp_for_index(55))
        self.assertEqual(split.validation_rows[0]["timestamp"], timestamp_for_index(61))
        self.assertEqual(split.validation_rows[-1]["timestamp"], timestamp_for_index(75))
        self.assertEqual(split.test_rows[0]["timestamp"], timestamp_for_index(81))


class TestSignalDiagnostics(unittest.TestCase):
    def test_summarize_rows_tracks_expectancy_and_exit_mix(self) -> None:
        rows = [
            {
                "timestamp": "2024-01-01T00:00:00+00:00",
                "side": "LONG",
                "outcome": "WIN",
                "exit_reason": "TAKE_PROFIT",
                "pnl_pct": "0.02",
                "net_r": "1.2",
                "max_favorable_r": "1.3",
                "max_adverse_r": "-0.2",
                "bars_held": "4",
                "signal_score": "80",
                "setup_quality": "75",
            },
            {
                "timestamp": "2024-01-02T00:00:00+00:00",
                "side": "SHORT",
                "outcome": "LOSS",
                "exit_reason": "STOP_LOSS",
                "pnl_pct": "-0.01",
                "net_r": "-1.0",
                "max_favorable_r": "0.2",
                "max_adverse_r": "-1.0",
                "bars_held": "2",
                "signal_score": "70",
                "setup_quality": "72",
            },
        ]

        summary = summarize_rows(rows)

        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["wins"], 1)
        self.assertAlmostEqual(float(summary["win_rate"]), 0.5)
        self.assertAlmostEqual(float(summary["avg_pnl_pct"]), 0.005)
        self.assertAlmostEqual(float(summary["avg_net_r"]), 0.1)
        self.assertAlmostEqual(float(summary["take_profit_rate"]), 0.5)
        self.assertAlmostEqual(float(summary["stop_loss_rate"]), 0.5)

    def test_group_summaries_splits_family_and_side(self) -> None:
        rows = [
            {
                "timestamp": "2024-01-01T00:00:00+00:00",
                "setup_family": "A",
                "side": "LONG",
                "outcome": "WIN",
                "pnl_pct": "0.01",
                "net_r": "1.0",
            },
            {
                "timestamp": "2024-01-02T00:00:00+00:00",
                "setup_family": "A",
                "side": "SHORT",
                "outcome": "LOSS",
                "pnl_pct": "-0.01",
                "net_r": "-1.0",
            },
            {
                "timestamp": "2024-01-03T00:00:00+00:00",
                "setup_family": "B",
                "side": "LONG",
                "outcome": "WIN",
                "pnl_pct": "0.02",
                "net_r": "2.0",
            },
        ]

        summaries = group_summaries(rows, ["setup_family", "side"])
        labels = {str(summary["label"]) for summary in summaries}

        self.assertEqual(labels, {"A / LONG", "A / SHORT", "B / LONG"})

    def test_oracle_top_k_rows_selects_best_realized_rows_per_week(self) -> None:
        rows = [
            {
                "timestamp": "2024-01-01T00:00:00+00:00",
                "outcome": "LOSS",
                "pnl_pct": "-0.02",
                "net_r": "-1.0",
                "signal_score": "90",
            },
            {
                "timestamp": "2024-01-02T00:00:00+00:00",
                "outcome": "WIN",
                "pnl_pct": "0.03",
                "net_r": "1.5",
                "signal_score": "70",
            },
            {
                "timestamp": "2024-01-08T00:00:00+00:00",
                "outcome": "WIN",
                "pnl_pct": "0.01",
                "net_r": "0.4",
                "signal_score": "60",
            },
            {
                "timestamp": "2024-01-09T00:00:00+00:00",
                "outcome": "LOSS",
                "pnl_pct": "-0.01",
                "net_r": "-0.5",
                "signal_score": "95",
            },
        ]

        selected = oracle_top_k_rows(rows, weekly_cap=1)
        summary = oracle_summary(rows, weekly_cap=1)

        self.assertEqual([row["timestamp"] for row in selected], [
            "2024-01-02T00:00:00+00:00",
            "2024-01-08T00:00:00+00:00",
        ])
        self.assertEqual(summary["count"], 2)
        self.assertAlmostEqual(float(summary["avg_net_r"]), 0.95)


class TestSignalLearning(unittest.TestCase):
    def test_effective_label_lookahead_honors_family_max_hold(self) -> None:
        candles = [
            _candle(hour=0, open_=100.0, high=100.8, low=99.7, close=100.5),
            _candle(hour=1, open_=100.5, high=101.4, low=100.2, close=101.1),
            _candle(hour=2, open_=101.1, high=102.1, low=100.9, close=101.9),
            _candle(hour=3, open_=101.9, high=103.0, low=101.7, close=102.8),
            _candle(hour=4, open_=102.8, high=104.0, low=102.5, close=103.7),
            _candle(hour=5, open_=103.7, high=105.0, low=103.4, close=104.8),
            _candle(hour=6, open_=104.8, high=106.2, low=104.5, close=105.9),
            _candle(hour=7, open_=105.9, high=107.3, low=105.6, close=106.8),
            _candle(hour=8, open_=106.8, high=108.2, low=106.6, close=107.9),
            _candle(hour=9, open_=107.9, high=109.4, low=107.7, close=108.8),
            _candle(hour=10, open_=108.8, high=110.4, low=108.5, close=109.9),
            _candle(hour=11, open_=109.9, high=111.6, low=109.6, close=111.0),
            _candle(hour=12, open_=114.0, high=114.8, low=113.8, close=114.5),
            _candle(hour=13, open_=114.5, high=115.0, low=114.2, close=114.8),
            _candle(hour=14, open_=114.8, high=115.3, low=114.4, close=115.0),
            _candle(hour=15, open_=115.0, high=116.0, low=114.8, close=115.5),
            _candle(hour=16, open_=115.5, high=115.8, low=114.9, close=115.1),
            _candle(hour=17, open_=115.1, high=115.4, low=114.7, close=114.9),
            _candle(hour=18, open_=114.9, high=115.3, low=114.6, close=114.8),
            _candle(hour=19, open_=114.9, high=116.0, low=114.7, close=115.8),
        ]
        snapshot = _snapshot(candles, supports=[114.6, 113.8], resistances=[119.5, 121.0])
        indicators = _indicator_set(
            ema20=115.0,
            ema50=112.5,
            ema200=108.0,
            vwap=115.1,
            rsi=58.0,
            macd_hist=0.4,
            atr=1.0,
            bb_upper=117.2,
            bb_middle=115.1,
            bb_lower=113.0,
            bb_width_pct=0.043,
            volume_ratio=1.1,
        )
        regime = RegimeResult(
            regime="TRENDING_BULLISH",
            strategy="TREND_FOLLOWING",
            confidence=0.8,
            reason="test",
        )

        candidate = generate_candidate(snapshot, regime, indicators)
        self.assertIsNotNone(candidate)

        self.assertEqual(effective_label_lookahead(candidate, 48), candidate.max_hold_bars)
        self.assertEqual(effective_label_lookahead(candidate, 12), 12)


class TestSetupFamilies(unittest.TestCase):
    def test_generate_candidate_builds_trend_pullback_family(self) -> None:
        candles = [
            _candle(hour=0, open_=100.0, high=100.8, low=99.7, close=100.5),
            _candle(hour=1, open_=100.5, high=101.4, low=100.2, close=101.1),
            _candle(hour=2, open_=101.1, high=102.1, low=100.9, close=101.9),
            _candle(hour=3, open_=101.9, high=103.0, low=101.7, close=102.8),
            _candle(hour=4, open_=102.8, high=104.0, low=102.5, close=103.7),
            _candle(hour=5, open_=103.7, high=105.0, low=103.4, close=104.8),
            _candle(hour=6, open_=104.8, high=106.2, low=104.5, close=105.9),
            _candle(hour=7, open_=105.9, high=107.3, low=105.6, close=106.8),
            _candle(hour=8, open_=106.8, high=108.2, low=106.6, close=107.9),
            _candle(hour=9, open_=107.9, high=109.4, low=107.7, close=108.8),
            _candle(hour=10, open_=108.8, high=110.4, low=108.5, close=109.9),
            _candle(hour=11, open_=109.9, high=111.6, low=109.6, close=111.0),
            _candle(hour=12, open_=114.0, high=114.8, low=113.8, close=114.5),
            _candle(hour=13, open_=114.5, high=115.0, low=114.2, close=114.8),
            _candle(hour=14, open_=114.8, high=115.3, low=114.4, close=115.0),
            _candle(hour=15, open_=115.0, high=116.0, low=114.8, close=115.5),
            _candle(hour=16, open_=115.5, high=115.8, low=114.9, close=115.1),
            _candle(hour=17, open_=115.1, high=115.4, low=114.7, close=114.9),
            _candle(hour=18, open_=114.9, high=115.3, low=114.6, close=114.8),
            _candle(hour=19, open_=114.9, high=116.0, low=114.7, close=115.8),
        ]
        snapshot = _snapshot(candles, supports=[114.6, 113.8], resistances=[119.5, 121.0])
        indicators = _indicator_set(
            ema20=115.0,
            ema50=112.5,
            ema200=108.0,
            vwap=115.1,
            rsi=58.0,
            macd_hist=0.4,
            atr=1.0,
            bb_upper=117.2,
            bb_middle=115.1,
            bb_lower=113.0,
            bb_width_pct=0.043,
            volume_ratio=1.1,
        )
        regime = RegimeResult(
            regime="TRENDING_BULLISH",
            strategy="TREND_FOLLOWING",
            confidence=0.8,
            reason="test",
        )

        candidate = generate_candidate(snapshot, regime, indicators)

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.setup_family, "TREND_PULLBACK_CONTINUATION")
        self.assertEqual(candidate.direction, "LONG")
        self.assertEqual(candidate.max_hold_bars, 36)
        self.assertGreater(candidate.setup_quality, 70.0)

    def test_generate_candidate_builds_breakout_retest_family(self) -> None:
        candles = [
            _candle(hour=0, open_=100.0, high=100.6, low=99.9, close=100.3),
            _candle(hour=1, open_=100.3, high=101.1, low=100.2, close=100.8),
            _candle(hour=2, open_=100.8, high=101.8, low=100.7, close=101.4),
            _candle(hour=3, open_=101.4, high=102.4, low=101.2, close=102.0),
            _candle(hour=4, open_=102.0, high=103.1, low=101.9, close=102.8),
            _candle(hour=5, open_=102.8, high=104.0, low=102.7, close=103.6),
            _candle(hour=6, open_=103.6, high=105.0, low=103.5, close=104.6),
            _candle(hour=7, open_=104.6, high=106.0, low=104.4, close=105.4),
            _candle(hour=8, open_=105.4, high=107.0, low=105.2, close=106.3),
            _candle(hour=9, open_=106.3, high=108.0, low=106.1, close=107.2),
            _candle(hour=10, open_=107.2, high=108.9, low=107.0, close=108.1),
            _candle(hour=11, open_=108.1, high=109.6, low=107.9, close=109.0),
            _candle(hour=12, open_=109.0, high=109.5, low=108.7, close=109.1),
            _candle(hour=13, open_=109.1, high=109.4, low=108.8, close=109.0),
            _candle(hour=14, open_=109.0, high=109.3, low=108.7, close=109.0),
            _candle(hour=15, open_=109.0, high=109.3, low=108.8, close=109.0),
            _candle(hour=16, open_=109.0, high=109.4, low=108.7, close=109.1),
            _candle(hour=17, open_=109.1, high=109.5, low=108.9, close=109.2),
            _candle(hour=18, open_=109.2, high=109.6, low=109.0, close=109.3),
            _candle(hour=19, open_=109.8, high=111.3, low=109.5, close=110.8),
        ]
        snapshot = _snapshot(candles, supports=[108.8, 107.5], resistances=[114.5, 116.0])
        indicators = _indicator_set(
            ema20=109.8,
            ema50=108.5,
            ema200=105.0,
            vwap=109.9,
            rsi=61.0,
            macd_hist=0.45,
            atr=1.3,
            bb_upper=111.5,
            bb_middle=109.8,
            bb_lower=108.1,
            bb_width_pct=0.03,
            volume_ratio=1.25,
        )
        regime = RegimeResult(
            regime="TRENDING_BULLISH",
            strategy="TREND_FOLLOWING",
            confidence=0.85,
            reason="test",
        )

        candidate = generate_candidate(snapshot, regime, indicators)

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.setup_family, "BREAKOUT_RETEST_CONTINUATION")
        self.assertEqual(candidate.direction, "LONG")
        self.assertEqual(candidate.max_hold_bars, 24)
        self.assertGreater(candidate.reference_level or 0.0, 0.0)

    def test_generate_candidate_builds_range_rejection_family(self) -> None:
        candles = [
            _candle(hour=0, open_=101.8, high=102.2, low=101.4, close=102.0),
            _candle(hour=1, open_=102.0, high=103.2, low=101.8, close=102.9),
            _candle(hour=2, open_=102.9, high=103.8, low=102.5, close=103.4),
            _candle(hour=3, open_=103.4, high=104.0, low=102.8, close=103.2),
            _candle(hour=4, open_=103.2, high=103.7, low=102.4, close=102.8),
            _candle(hour=5, open_=102.8, high=103.4, low=101.9, close=102.3),
            _candle(hour=6, open_=102.3, high=103.0, low=101.4, close=101.8),
            _candle(hour=7, open_=101.8, high=102.6, low=100.9, close=101.2),
            _candle(hour=8, open_=101.2, high=102.2, low=100.6, close=101.5),
            _candle(hour=9, open_=101.5, high=102.8, low=101.0, close=102.1),
            _candle(hour=10, open_=102.1, high=103.2, low=101.6, close=102.7),
            _candle(hour=11, open_=102.7, high=103.6, low=102.1, close=103.0),
            _candle(hour=12, open_=103.0, high=103.9, low=102.5, close=103.4),
            _candle(hour=13, open_=103.4, high=103.8, low=102.6, close=103.1),
            _candle(hour=14, open_=103.1, high=103.4, low=102.3, close=102.7),
            _candle(hour=15, open_=102.7, high=103.0, low=101.7, close=102.0),
            _candle(hour=16, open_=102.0, high=102.5, low=101.0, close=101.4),
            _candle(hour=17, open_=101.4, high=101.9, low=100.5, close=101.0),
            _candle(hour=18, open_=101.0, high=101.6, low=100.3, close=100.8),
            _candle(hour=19, open_=101.1, high=101.7, low=100.5, close=101.3),
        ]
        snapshot = _snapshot(candles, supports=[100.0, 99.6], resistances=[103.9, 104.2])
        indicators = _indicator_set(
            ema20=101.9,
            ema50=102.0,
            ema200=101.8,
            vwap=102.0,
            rsi=44.0,
            macd_hist=-0.05,
            atr=0.8,
            bb_upper=103.8,
            bb_middle=102.0,
            bb_lower=100.4,
            bb_width_pct=0.028,
            volume_ratio=1.0,
        )
        regime = RegimeResult(
            regime="RANGE_BOUND",
            strategy="MEAN_REVERSION",
            confidence=0.85,
            reason="test",
        )

        candidate = generate_candidate(snapshot, regime, indicators)

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.setup_family, "RANGE_REJECTION_MEAN_REVERSION")
        self.assertEqual(candidate.direction, "LONG")
        self.assertEqual(candidate.max_hold_bars, 12)

    def test_generate_candidate_builds_failed_breakout_family(self) -> None:
        candles = [
            _candle(hour=0, open_=101.0, high=101.7, low=100.7, close=101.3),
            _candle(hour=1, open_=101.3, high=102.4, low=101.0, close=101.9),
            _candle(hour=2, open_=101.9, high=103.0, low=101.6, close=102.4),
            _candle(hour=3, open_=102.4, high=103.6, low=102.1, close=102.9),
            _candle(hour=4, open_=102.9, high=104.0, low=102.6, close=103.3),
            _candle(hour=5, open_=103.3, high=103.8, low=102.7, close=103.1),
            _candle(hour=6, open_=103.1, high=103.6, low=102.4, close=102.8),
            _candle(hour=7, open_=102.8, high=103.4, low=102.1, close=102.6),
            _candle(hour=8, open_=102.6, high=103.2, low=102.0, close=102.3),
            _candle(hour=9, open_=102.3, high=103.0, low=101.8, close=102.0),
            _candle(hour=10, open_=102.0, high=102.8, low=101.5, close=101.9),
            _candle(hour=11, open_=101.9, high=102.5, low=101.3, close=101.7),
            _candle(hour=12, open_=101.7, high=102.4, low=101.1, close=101.6),
            _candle(hour=13, open_=101.6, high=102.3, low=101.0, close=101.5),
            _candle(hour=14, open_=101.5, high=102.2, low=100.9, close=101.4),
            _candle(hour=15, open_=101.4, high=102.4, low=101.0, close=101.9),
            _candle(hour=16, open_=101.9, high=103.1, low=101.7, close=102.4),
            _candle(hour=17, open_=102.4, high=103.7, low=102.2, close=103.0),
            _candle(hour=18, open_=103.0, high=104.0, low=102.7, close=103.4),
            _candle(hour=19, open_=104.1, high=104.9, low=103.4, close=103.6),
        ]
        snapshot = _snapshot(candles, supports=[100.8, 100.2], resistances=[104.0, 104.4])
        indicators = _indicator_set(
            ema20=102.8,
            ema50=102.3,
            ema200=101.5,
            vwap=103.0,
            rsi=63.0,
            macd_hist=0.15,
            atr=0.8,
            bb_upper=104.2,
            bb_middle=102.8,
            bb_lower=101.4,
            bb_width_pct=0.031,
            volume_ratio=1.15,
        )
        regime = RegimeResult(
            regime="HIGH_VOLATILITY",
            strategy="BREAKOUT",
            confidence=0.9,
            reason="test",
        )

        candidate = generate_candidate(snapshot, regime, indicators)

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.setup_family, "FAILED_BREAKOUT_REVERSAL")
        self.assertEqual(candidate.direction, "SHORT")
        self.assertEqual(candidate.max_hold_bars, 18)

    def test_setup_family_metadata_flows_into_features(self) -> None:
        candles = [
            _candle(hour=0, open_=100.0, high=100.8, low=99.7, close=100.5),
            _candle(hour=1, open_=100.5, high=101.4, low=100.2, close=101.1),
            _candle(hour=2, open_=101.1, high=102.1, low=100.9, close=101.9),
            _candle(hour=3, open_=101.9, high=103.0, low=101.7, close=102.8),
            _candle(hour=4, open_=102.8, high=104.0, low=102.5, close=103.7),
            _candle(hour=5, open_=103.7, high=105.0, low=103.4, close=104.8),
            _candle(hour=6, open_=104.8, high=106.2, low=104.5, close=105.9),
            _candle(hour=7, open_=105.9, high=107.3, low=105.6, close=106.8),
            _candle(hour=8, open_=106.8, high=108.2, low=106.6, close=107.9),
            _candle(hour=9, open_=107.9, high=109.4, low=107.7, close=108.8),
            _candle(hour=10, open_=108.8, high=110.4, low=108.5, close=109.9),
            _candle(hour=11, open_=109.9, high=111.6, low=109.6, close=111.0),
            _candle(hour=12, open_=114.0, high=114.8, low=113.8, close=114.5),
            _candle(hour=13, open_=114.5, high=115.0, low=114.2, close=114.8),
            _candle(hour=14, open_=114.8, high=115.3, low=114.4, close=115.0),
            _candle(hour=15, open_=115.0, high=116.0, low=114.8, close=115.5),
            _candle(hour=16, open_=115.5, high=115.8, low=114.9, close=115.1),
            _candle(hour=17, open_=115.1, high=115.4, low=114.7, close=114.9),
            _candle(hour=18, open_=114.9, high=115.3, low=114.6, close=114.8),
            _candle(hour=19, open_=114.9, high=116.0, low=114.7, close=115.8),
        ]
        snapshot = _snapshot(candles, supports=[114.6, 113.8], resistances=[119.5, 121.0])
        indicators = _indicator_set(
            ema20=115.0,
            ema50=112.5,
            ema200=108.0,
            vwap=115.1,
            rsi=58.0,
            macd_hist=0.4,
            atr=1.0,
            bb_upper=117.2,
            bb_middle=115.1,
            bb_lower=113.0,
            bb_width_pct=0.043,
            volume_ratio=1.1,
        )
        regime = RegimeResult(
            regime="TRENDING_BULLISH",
            strategy="TREND_FOLLOWING",
            confidence=0.8,
            reason="test",
        )
        confluence = ConfluenceBreakdown(
            trend_alignment=80.0,
            momentum=75.0,
            volume_liquidity=70.0,
            structure=74.0,
            sentiment=50.0,
            total_score=73.8,
        )

        candidate = generate_candidate(snapshot, regime, indicators)
        self.assertIsNotNone(candidate)

        features = extract_candidate_features(candidate, confluence)

        self.assertEqual(features["setup_trend_pullback"], 1.0)
        self.assertEqual(features["setup_breakout_retest"], 0.0)
        self.assertGreater(features["setup_quality"], 70.0)
        self.assertGreater(features["max_hold_bars_norm"], 0.0)


if __name__ == "__main__":
    unittest.main()
