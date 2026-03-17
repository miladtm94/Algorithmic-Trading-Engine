from __future__ import annotations

from .config import RegimeConfig
from .models import IndicatorSet, RegimeResult


def classify_regime(ind: IndicatorSet, cfg: RegimeConfig) -> RegimeResult:
    trend_strength = abs(ind.ema20 - ind.ema200) / ind.ema200 if ind.ema200 else 0.0
    bullish_trend = ind.ema20 > ind.ema50 > ind.ema200 and ind.macd_hist > 0
    bearish_trend = ind.ema20 < ind.ema50 < ind.ema200 and ind.macd_hist < 0

    if ind.atr_pct >= cfg.high_volatility_atr_pct:
        return RegimeResult(
            regime="HIGH_VOLATILITY",
            strategy="BREAKOUT",
            confidence=min(1.0, ind.atr_pct / (cfg.high_volatility_atr_pct * 2)),
            reason=f"ATR% elevated at {ind.atr_pct:.2%}.",
        )

    if bullish_trend and trend_strength >= cfg.trend_strength_threshold:
        return RegimeResult(
            regime="TRENDING_BULLISH",
            strategy="TREND_FOLLOWING",
            confidence=min(1.0, trend_strength * 20),
            reason="EMA20 > EMA50 > EMA200 with positive MACD histogram.",
        )

    if bearish_trend and trend_strength >= cfg.trend_strength_threshold:
        return RegimeResult(
            regime="TRENDING_BEARISH",
            strategy="TREND_FOLLOWING",
            confidence=min(1.0, trend_strength * 20),
            reason="EMA20 < EMA50 < EMA200 with negative MACD histogram.",
        )

    if ind.bb_width_pct <= cfg.range_bb_width_pct:
        return RegimeResult(
            regime="RANGE_BOUND",
            strategy="MEAN_REVERSION",
            confidence=min(1.0, (cfg.range_bb_width_pct - ind.bb_width_pct) * 30),
            reason=f"Bollinger width compressed at {ind.bb_width_pct:.2%}.",
        )

    if ind.macd_hist >= 0:
        return RegimeResult(
            regime="TRENDING_BULLISH",
            strategy="TREND_FOLLOWING",
            confidence=0.52,
            reason="Mild upward momentum bias.",
        )

    return RegimeResult(
        regime="TRENDING_BEARISH",
        strategy="TREND_FOLLOWING",
        confidence=0.52,
        reason="Mild downward momentum bias.",
    )
