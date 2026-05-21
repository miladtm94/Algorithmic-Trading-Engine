"""Extract flat feature vectors for ML training/inference."""
from __future__ import annotations

from .models import CandidateSignal, ConfluenceBreakdown, FinalSignal

# Canonical feature order — must stay stable across build/train/evaluate
FEATURE_NAMES: list[str] = [
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
]


def extract_features(signal: FinalSignal) -> dict[str, float]:
    """Return a price-normalised, direction-aware feature dict from a FinalSignal."""
    return extract_candidate_features(signal.candidate, signal.confluence)


def extract_candidate_features(
    candidate: CandidateSignal,
    confluence: ConfluenceBreakdown,
) -> dict[str, float]:
    """Return a price-normalised, direction-aware feature dict from a candidate setup."""
    ind = candidate.indicators
    conf = confluence
    regime = candidate.regime
    entry = candidate.entry
    direction = 1.0 if candidate.direction == "LONG" else -1.0
    reference_level = candidate.reference_level if candidate.reference_level is not None else entry
    reference_distance = abs(entry - reference_level) / entry if entry > 0 else 0.0
    reference_directional = (
        direction * ((entry - reference_level) / entry) if entry > 0 and reference_level else 0.0
    )

    bb_range = ind.bb_upper - ind.bb_lower
    bb_pos = (entry - ind.bb_lower) / bb_range if bb_range > 0 else 0.5
    tp1 = candidate.take_profits[0] if candidate.take_profits else entry
    stop_distance = abs(entry - candidate.stop_loss) / entry if entry > 0 else 0.0
    tp1_distance = abs(tp1 - entry) / entry if entry > 0 else 0.0
    price_vs_ema20 = (entry - ind.ema20) / entry if entry > 0 else 0.0
    price_vs_ema50 = (entry - ind.ema50) / entry if entry > 0 else 0.0
    price_vs_ema200 = (entry - ind.ema200) / entry if entry > 0 else 0.0
    price_vs_vwap = (entry - ind.vwap) / entry if entry > 0 else 0.0
    ema20_50 = (ind.ema20 - ind.ema50) / entry if entry > 0 else 0.0
    ema50_200 = (ind.ema50 - ind.ema200) / entry if entry > 0 else 0.0
    ema20_200 = (ind.ema20 - ind.ema200) / entry if entry > 0 else 0.0
    macd_norm = ind.macd / entry if entry > 0 else 0.0
    macd_signal_norm = ind.macd_signal / entry if entry > 0 else 0.0
    macd_hist_norm = ind.macd_hist / entry if entry > 0 else 0.0
    rsi_centered = (ind.rsi - 50.0) / 50.0
    bb_centered = (bb_pos - 0.5) * 2.0

    return {
        "direction": direction,
        # EMAs expressed as fractional distance from entry (e.g. 0.02 = 2% above)
        "ema20_ratio": (ind.ema20 - entry) / entry,
        "ema50_ratio": (ind.ema50 - entry) / entry,
        "ema200_ratio": (ind.ema200 - entry) / entry,
        "vwap_ratio": (ind.vwap - entry) / entry,
        # Momentum — already dimensionless or normalised
        "rsi": ind.rsi,
        "macd_hist_norm": macd_hist_norm,
        # Volatility
        "atr_pct": ind.atr_pct,
        "bb_width_pct": ind.bb_width_pct,
        "bb_position": bb_pos,  # 0 = at lower band, 1 = at upper band
        # Volume / liquidity
        "volume_ratio": ind.volume_ratio,
        "order_book_imbalance": ind.order_book_imbalance,
        # Regime one-hot
        "is_trending_bullish": float(regime.regime == "TRENDING_BULLISH"),
        "is_trending_bearish": float(regime.regime == "TRENDING_BEARISH"),
        "is_range_bound": float(regime.regime == "RANGE_BOUND"),
        "is_high_volatility": float(regime.regime == "HIGH_VOLATILITY"),
        "regime_confidence": regime.confidence,
        # Confluence sub-scores (0-100 each)
        "conf_trend": conf.trend_alignment,
        "conf_momentum": conf.momentum,
        "conf_volume": conf.volume_liquidity,
        "conf_structure": conf.structure,
        "conf_sentiment": conf.sentiment,
        "conf_total": conf.total_score,
        # Setup quality
        "risk_reward": candidate.risk_reward,
        "setup_trend_pullback": float(
            candidate.setup_family == "TREND_PULLBACK_CONTINUATION"
        ),
        "setup_breakout_retest": float(
            candidate.setup_family == "BREAKOUT_RETEST_CONTINUATION"
        ),
        "setup_range_rejection": float(
            candidate.setup_family == "RANGE_REJECTION_MEAN_REVERSION"
        ),
        "setup_failed_breakout": float(
            candidate.setup_family == "FAILED_BREAKOUT_REVERSAL"
        ),
        "setup_quality": candidate.setup_quality,
        "max_hold_bars_norm": candidate.max_hold_bars / 48.0,
        "reference_distance_pct": reference_distance,
        "reference_distance_directional": reference_directional,
        # Richer price-action / direction-aware features
        "price_vs_ema20": price_vs_ema20,
        "price_vs_ema50": price_vs_ema50,
        "price_vs_ema200": price_vs_ema200,
        "price_vs_vwap": price_vs_vwap,
        "ema20_50_spread": ema20_50,
        "ema50_200_spread": ema50_200,
        "ema20_200_spread": ema20_200,
        "ema_stack_directional": direction * (ema20_50 + ema50_200),
        "ema20_directional": direction * price_vs_ema20,
        "ema50_directional": direction * price_vs_ema50,
        "ema200_directional": direction * price_vs_ema200,
        "vwap_directional": direction * price_vs_vwap,
        "rsi_centered": rsi_centered,
        "rsi_directional": direction * rsi_centered,
        "rsi_overbought": float(ind.rsi >= 70.0),
        "rsi_oversold": float(ind.rsi <= 30.0),
        "macd_norm": macd_norm,
        "macd_signal_norm": macd_signal_norm,
        "macd_directional": direction * macd_norm,
        "macd_hist_directional": direction * macd_hist_norm,
        "bb_position_centered": bb_centered,
        "bb_position_directional": direction * bb_centered,
        "volume_expansion": ind.volume_ratio - 1.0,
        "high_volume": float(ind.volume_ratio >= 1.5),
        "stop_distance_pct": stop_distance,
        "tp1_distance_pct": tp1_distance,
        "conf_trend_norm": conf.trend_alignment / 100.0,
        "conf_momentum_norm": conf.momentum / 100.0,
        "conf_volume_norm": conf.volume_liquidity / 100.0,
        "conf_structure_norm": conf.structure / 100.0,
        "conf_sentiment_norm": conf.sentiment / 100.0,
        "conf_total_norm": conf.total_score / 100.0,
    }


def features_to_row(features: dict[str, float]) -> list[float]:
    """Convert feature dict to a list in canonical FEATURE_NAMES order."""
    return [features.get(k, 0.0) for k in FEATURE_NAMES]
