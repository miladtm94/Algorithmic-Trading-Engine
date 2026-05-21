"""Helpers for sparse signal-selection datasets and models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean

from .config import EngineConfig
from .confluence import score_candidate
from .dataset import build_research_snapshot, label_trade_path
from .feature_extractor import extract_candidate_features
from .indicators import compute_indicators
from .models import CandidateSignal, Candle, ConfluenceBreakdown, MarketSnapshot
from .regime import classify_regime
from .signal_generation import generate_candidate, read_structure

SIGNAL_FEATURE_NAMES: list[str] = [
    "direction",
    "ema20_ratio",
    "ema50_ratio",
    "ema200_ratio",
    "vwap_ratio",
    "rsi",
    "macd_hist_norm",
    "atr_pct",
    "bb_width_pct",
    "bb_position",
    "volume_ratio",
    "order_book_imbalance",
    "is_trending_bullish",
    "is_trending_bearish",
    "is_range_bound",
    "is_high_volatility",
    "regime_confidence",
    "conf_trend",
    "conf_momentum",
    "conf_volume",
    "conf_structure",
    "conf_sentiment",
    "conf_total",
    "risk_reward",
    "setup_trend_pullback",
    "setup_breakout_retest",
    "setup_range_rejection",
    "setup_failed_breakout",
    "setup_quality",
    "max_hold_bars_norm",
    "reference_distance_pct",
    "reference_distance_directional",
    "price_vs_ema20",
    "price_vs_ema50",
    "price_vs_ema200",
    "price_vs_vwap",
    "ema20_50_spread",
    "ema50_200_spread",
    "ema20_200_spread",
    "ema_stack_directional",
    "ema20_directional",
    "ema50_directional",
    "ema200_directional",
    "vwap_directional",
    "rsi_centered",
    "rsi_directional",
    "rsi_overbought",
    "rsi_oversold",
    "macd_norm",
    "macd_signal_norm",
    "macd_directional",
    "macd_hist_directional",
    "bb_position_centered",
    "bb_position_directional",
    "volume_expansion",
    "high_volume",
    "stop_distance_pct",
    "tp1_distance_pct",
    "conf_trend_norm",
    "conf_momentum_norm",
    "conf_volume_norm",
    "conf_structure_norm",
    "conf_sentiment_norm",
    "conf_total_norm",
    "ret_1h",
    "ret_3h",
    "ret_6h",
    "ret_12h",
    "ret_24h",
    "volatility_6h",
    "volatility_24h",
    "candle_range_pct",
    "candle_body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "directional_body_pct",
    "directional_ret_3h",
    "directional_ret_12h",
    "support_distance_pct",
    "resistance_distance_pct",
    "directional_support_distance_pct",
    "directional_resistance_distance_pct",
    "near_support",
    "near_resistance",
    "broke_support",
    "broke_resistance",
    "reason_count",
    "engine_passed_threshold",
    "htf_ema20_ratio",
    "htf_ema50_ratio",
    "htf_ema200_ratio",
    "htf_rsi",
    "htf_macd_hist_norm",
    "htf_atr_pct",
    "htf_trend_aligned",
    "range_width_state_pct",
    "range_position_state",
    "range_position_directional",
    "pullback_depth_atr",
    "breakout_distance_atr",
    "rejection_wick_bias",
    "compression_flag",
]


@dataclass(slots=True)
class SignalDatasetSummary:
    rows: int
    wins: int
    losses: int
    long_rows: int
    short_rows: int
    engine_threshold_rows: int

    @property
    def win_rate(self) -> float:
        return self.wins / self.rows if self.rows else 0.0


def aggregate_candles(candles: list[Candle], factor: int) -> list[Candle]:
    """Aggregate lower-timeframe candles into higher-timeframe bars."""
    if factor <= 1 or len(candles) < factor:
        return candles

    trimmed = candles[len(candles) % factor :]
    aggregated: list[Candle] = []
    for idx in range(0, len(trimmed), factor):
        chunk = trimmed[idx : idx + factor]
        if len(chunk) < factor:
            continue
        aggregated.append(
            Candle(
                timestamp=chunk[-1].timestamp,
                open=chunk[0].open,
                high=max(candle.high for candle in chunk),
                low=min(candle.low for candle in chunk),
                close=chunk[-1].close,
                volume=sum(candle.volume for candle in chunk),
            )
        )
    return aggregated or candles


def build_learning_features(
    candidate: CandidateSignal,
    confluence: ConfluenceBreakdown,
    snapshot: MarketSnapshot,
    *,
    engine_threshold: float,
) -> dict[str, float]:
    """Build richer, direction-aware features for sparse signal selection."""
    features = extract_candidate_features(candidate, confluence)
    candles = snapshot.candles
    closes = [candle.close for candle in candles]
    direction = 1.0 if candidate.direction == "LONG" else -1.0
    last = candles[-1]
    structure = read_structure(snapshot)

    def past_return(bars: int) -> float:
        if len(closes) <= bars:
            return 0.0
        prev = closes[-bars - 1]
        return (closes[-1] - prev) / prev if prev else 0.0

    def realized_volatility(bars: int) -> float:
        if len(closes) <= bars:
            return 0.0
        returns = [
            (closes[idx] - closes[idx - 1]) / closes[idx - 1]
            for idx in range(len(closes) - bars, len(closes))
            if idx > 0 and closes[idx - 1] > 0
        ]
        if len(returns) < 2:
            return 0.0
        avg = mean(returns)
        variance = sum((value - avg) ** 2 for value in returns) / len(returns)
        return math.sqrt(variance)

    candle_range = (last.high - last.low) / last.close if last.close else 0.0
    candle_body = (last.close - last.open) / last.open if last.open else 0.0
    upper_wick = (last.high - max(last.open, last.close)) / last.close if last.close else 0.0
    lower_wick = (min(last.open, last.close) - last.low) / last.close if last.close else 0.0
    atr = max(candidate.indicators.atr, candidate.entry * 0.003)

    support_distance = (
        abs(candidate.entry - structure.nearest_support) / candidate.entry
        if candidate.entry
        else 0.0
    )
    resistance_distance = (
        abs(structure.nearest_resistance - candidate.entry) / candidate.entry
        if candidate.entry
        else 0.0
    )

    htf_source = candles[-240:] if len(candles) >= 240 else candles
    htf_candles = aggregate_candles(htf_source, 4)
    htf_indicators = compute_indicators(htf_candles, snapshot.order_book)
    htf_entry = htf_candles[-1].close
    recent_pullback_window = candles[-6:] if len(candles) >= 6 else candles
    recent_low = min(candle.low for candle in recent_pullback_window)
    recent_high = max(candle.high for candle in recent_pullback_window)
    pullback_depth_atr = (
        max(0.0, candidate.entry - recent_low) / atr
        if candidate.direction == "LONG"
        else max(0.0, recent_high - candidate.entry) / atr
    )
    breakout_distance_atr = (
        (candidate.entry - structure.prior_high) / atr
        if candidate.direction == "LONG"
        else (structure.prior_low - candidate.entry) / atr
    )
    rejection_wick_bias = lower_wick - upper_wick
    htf_trend_aligned = float(
        (
            candidate.direction == "LONG"
            and htf_indicators.ema20 > htf_indicators.ema50 > htf_indicators.ema200
        )
        or (
            candidate.direction == "SHORT"
            and htf_indicators.ema20 < htf_indicators.ema50 < htf_indicators.ema200
        )
    )

    features.update(
        {
            "ret_1h": past_return(1),
            "ret_3h": past_return(3),
            "ret_6h": past_return(6),
            "ret_12h": past_return(12),
            "ret_24h": past_return(24),
            "volatility_6h": realized_volatility(6),
            "volatility_24h": realized_volatility(24),
            "candle_range_pct": candle_range,
            "candle_body_pct": candle_body,
            "upper_wick_pct": upper_wick,
            "lower_wick_pct": lower_wick,
            "directional_body_pct": direction * candle_body,
            "directional_ret_3h": direction * past_return(3),
            "directional_ret_12h": direction * past_return(12),
            "support_distance_pct": support_distance,
            "resistance_distance_pct": resistance_distance,
            "directional_support_distance_pct": direction * (1.0 - support_distance),
            "directional_resistance_distance_pct": direction * (1.0 - resistance_distance),
            "near_support": float(structure.near_support),
            "near_resistance": float(structure.near_resistance),
            "broke_support": float(structure.broke_support),
            "broke_resistance": float(structure.broke_resistance),
            "reason_count": float(len(candidate.reasons)),
            "engine_passed_threshold": float(confluence.total_score >= engine_threshold),
            "htf_ema20_ratio": (htf_indicators.ema20 - htf_entry) / htf_entry if htf_entry else 0.0,
            "htf_ema50_ratio": (htf_indicators.ema50 - htf_entry) / htf_entry if htf_entry else 0.0,
            "htf_ema200_ratio": (htf_indicators.ema200 - htf_entry) / htf_entry
            if htf_entry
            else 0.0,
            "htf_rsi": htf_indicators.rsi,
            "htf_macd_hist_norm": htf_indicators.macd_hist / htf_entry if htf_entry else 0.0,
            "htf_atr_pct": htf_indicators.atr_pct,
            "htf_trend_aligned": htf_trend_aligned,
            "range_width_state_pct": structure.range_width_pct,
            "range_position_state": structure.range_position,
            "range_position_directional": direction * ((structure.range_position - 0.5) * 2.0),
            "pullback_depth_atr": pullback_depth_atr,
            "breakout_distance_atr": breakout_distance_atr,
            "rejection_wick_bias": rejection_wick_bias,
            "compression_flag": float(structure.is_compressed),
        }
    )
    return features


def effective_label_lookahead(candidate: CandidateSignal, configured_lookahead: int) -> int:
    """Return the family-aware label horizon, capped by the requested research lookahead."""
    configured = max(1, configured_lookahead)
    family_hold = max(1, candidate.max_hold_bars)
    return min(configured, family_hold)


def build_signal_rows(
    candles: list[Candle],
    asset: str,
    timeframe: str,
    *,
    config: EngineConfig | None = None,
    window_size: int = 220,
    lookahead: int = 24,
    fee_bps: float = 10.0,
    min_profit_pct: float = 0.0,
    min_confluence: float = 55.0,
) -> tuple[list[dict[str, object]], SignalDatasetSummary]:
    """Create a sparse labeled dataset from real engine candidate setups."""
    if len(candles) < window_size + lookahead + 1:
        raise ValueError(f"Need at least {window_size + lookahead + 1} candles, got {len(candles)}")

    cfg = config or EngineConfig()
    rows: list[dict[str, object]] = []
    max_entry_index = len(candles) - lookahead - 1

    for idx in range(window_size, max_entry_index + 1):
        window = candles[idx - window_size : idx + 1]
        snapshot = build_research_snapshot(window, asset, timeframe)
        indicators = compute_indicators(window, snapshot.order_book)
        regime = classify_regime(indicators, cfg.regime)
        candidate = generate_candidate(snapshot, regime, indicators)
        if candidate is None:
            continue

        confluence = score_candidate(candidate, snapshot, cfg.confluence_weights)
        if confluence.total_score < min_confluence:
            continue

        label_lookahead = effective_label_lookahead(candidate, lookahead)
        label = label_trade_path(
            side=candidate.direction,
            entry=candidate.entry,
            stop_loss=candidate.stop_loss,
            take_profit=candidate.take_profits[0],
            future_candles=candles[idx + 1 : idx + label_lookahead + 1],
            fee_bps=fee_bps,
            min_profit_pct=min_profit_pct,
        )
        features = build_learning_features(
            candidate,
            confluence,
            snapshot,
            engine_threshold=cfg.confluence_threshold,
        )
        row: dict[str, object] = {
            "timestamp": candles[idx].timestamp.isoformat(),
            "asset": asset,
            "timeframe": timeframe,
            "side": candidate.direction,
            "outcome": label.outcome,
            "exit_reason": label.exit_reason,
            "entry": round(candidate.entry, 8),
            "stop_loss": round(candidate.stop_loss, 8),
            "take_profit": round(candidate.take_profits[0], 8),
            "exit_price": round(label.exit_price, 8),
            "bars_held": label.bars_held,
            "pnl_pct": round(label.pnl_pct, 8),
            "risk_pct": round(label.risk_pct, 8),
            "net_return_pct": round(label.net_return_pct, 8),
            "net_r": round(label.net_r, 8),
            "max_favorable_pct": round(label.max_favorable_pct, 8),
            "max_adverse_pct": round(label.max_adverse_pct, 8),
            "max_favorable_r": round(label.max_favorable_r, 8),
            "max_adverse_r": round(label.max_adverse_r, 8),
            "bars_to_target": label.bars_to_target,
            "bars_to_stop": label.bars_to_stop,
            "r_bucket": label.r_bucket,
            "meta_label": label.meta_label,
            "setup_family": candidate.setup_family,
            "setup_quality": round(candidate.setup_quality, 8),
            "max_hold_bars": candidate.max_hold_bars,
            "reference_level": round(candidate.reference_level, 8)
            if candidate.reference_level is not None
            else "",
            "signal_score": round(confluence.total_score, 8),
            "regime": regime.regime,
            "strategy": regime.strategy,
            "regime_confidence_raw": round(regime.confidence, 8),
            "reason_count_raw": len(candidate.reasons),
            "engine_passed_threshold": int(confluence.total_score >= cfg.confluence_threshold),
        }
        for name in SIGNAL_FEATURE_NAMES:
            row[name] = round(float(features.get(name, 0.0)), 8)
        rows.append(row)

    long_rows = sum(1 for row in rows if row["side"] == "LONG")
    wins = sum(1 for row in rows if row["outcome"] == "WIN")
    engine_threshold_rows = sum(1 for row in rows if row.get("engine_passed_threshold") == 1)
    summary = SignalDatasetSummary(
        rows=len(rows),
        wins=wins,
        losses=len(rows) - wins,
        long_rows=long_rows,
        short_rows=len(rows) - long_rows,
        engine_threshold_rows=engine_threshold_rows,
    )
    return rows, summary


def features_to_row(features: dict[str, float]) -> list[float]:
    """Convert a feature dict to model row order."""
    return [float(features.get(name, 0.0)) for name in SIGNAL_FEATURE_NAMES]
