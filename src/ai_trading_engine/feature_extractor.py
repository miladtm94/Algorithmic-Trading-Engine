"""Extract a flat feature vector from a FinalSignal for ML training/inference."""
from __future__ import annotations

from .models import FinalSignal

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
]


def extract_features(signal: FinalSignal) -> dict[str, float]:
    """Return a price-normalised, direction-aware feature dict from a FinalSignal."""
    ind = signal.candidate.indicators
    conf = signal.confluence
    regime = signal.candidate.regime
    entry = signal.candidate.entry
    direction = 1.0 if signal.candidate.direction == "LONG" else -1.0

    bb_range = ind.bb_upper - ind.bb_lower
    bb_pos = (entry - ind.bb_lower) / bb_range if bb_range > 0 else 0.5

    return {
        "direction": direction,
        # EMAs expressed as fractional distance from entry (e.g. 0.02 = 2% above)
        "ema20_ratio": (ind.ema20 - entry) / entry,
        "ema50_ratio": (ind.ema50 - entry) / entry,
        "ema200_ratio": (ind.ema200 - entry) / entry,
        "vwap_ratio": (ind.vwap - entry) / entry,
        # Momentum — already dimensionless or normalised
        "rsi": ind.rsi,
        "macd_hist_norm": ind.macd_hist / entry if entry > 0 else 0.0,
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
        "risk_reward": signal.candidate.risk_reward,
    }


def features_to_row(features: dict[str, float]) -> list[float]:
    """Convert feature dict to a list in canonical FEATURE_NAMES order."""
    return [features.get(k, 0.0) for k in FEATURE_NAMES]
