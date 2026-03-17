from __future__ import annotations

from dataclasses import dataclass

from .models import CandidateSignal, IndicatorSet, MarketSnapshot, RegimeResult


@dataclass(slots=True)
class StructureRead:
    nearest_support: float
    nearest_resistance: float
    broke_support: bool
    broke_resistance: bool
    near_support: bool
    near_resistance: bool


def read_structure(snapshot: MarketSnapshot) -> StructureRead:
    last = snapshot.candles[-1].close
    supports = sorted(snapshot.support_levels)
    resistances = sorted(snapshot.resistance_levels)

    nearest_support = max((s for s in supports if s <= last), default=last * 0.99)
    nearest_resistance = min((r for r in resistances if r >= last), default=last * 1.01)
    broke_support = any(last < s for s in supports[-2:]) if supports else False
    broke_resistance = any(last > r for r in resistances[:2]) if resistances else False
    near_support = abs(last - nearest_support) / last < 0.004
    near_resistance = abs(nearest_resistance - last) / last < 0.004

    return StructureRead(
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        broke_support=broke_support,
        broke_resistance=broke_resistance,
        near_support=near_support,
        near_resistance=near_resistance,
    )


def _build_long_candidate(
    snapshot: MarketSnapshot,
    regime: RegimeResult,
    ind: IndicatorSet,
    structure: StructureRead,
) -> CandidateSignal | None:
    entry = snapshot.candles[-1].close
    atr = max(ind.atr, entry * 0.003)
    stop = min(structure.nearest_support - 0.2 * atr, entry - 1.2 * atr)
    risk = entry - stop
    if risk <= 0:
        return None
    tp1 = entry + 2 * risk
    tp2 = entry + 3 * risk
    tp3 = entry + 4 * risk

    reasons: list[str] = []
    if ind.ema20 > ind.ema50 > ind.ema200:
        reasons.append("EMA 20/50/200 bullish alignment")
    if ind.macd_hist > 0:
        reasons.append("MACD histogram positive")
    if ind.rsi > 52:
        reasons.append("RSI bullish momentum")
    if structure.broke_resistance or structure.near_support:
        reasons.append("Structure supports upside continuation")
    if ind.volume_ratio > 1.2:
        reasons.append("Volume expansion confirms move")
    if ind.order_book_imbalance > 0.1:
        reasons.append("Order book bid-side imbalance")

    if len(reasons) < 3:
        return None

    return CandidateSignal(
        asset=snapshot.asset,
        direction="LONG",
        entry=entry,
        stop_loss=stop,
        take_profits=[tp1, tp2, tp3],
        regime=regime,
        indicators=ind,
        reasons=reasons,
        invalidation=f"Close below {stop:.2f} on {snapshot.timeframe} timeframe",
        risk_reward=(tp1 - entry) / risk,
    )


def _build_short_candidate(
    snapshot: MarketSnapshot,
    regime: RegimeResult,
    ind: IndicatorSet,
    structure: StructureRead,
) -> CandidateSignal | None:
    entry = snapshot.candles[-1].close
    atr = max(ind.atr, entry * 0.003)
    stop = max(structure.nearest_resistance + 0.2 * atr, entry + 1.2 * atr)
    risk = stop - entry
    if risk <= 0:
        return None
    tp1 = entry - 2 * risk
    tp2 = entry - 3 * risk
    tp3 = entry - 4 * risk

    reasons: list[str] = []
    if ind.ema20 < ind.ema50 < ind.ema200:
        reasons.append("EMA 20/50/200 bearish alignment")
    if ind.macd_hist < 0:
        reasons.append("MACD histogram negative")
    if ind.rsi < 48:
        reasons.append("RSI bearish momentum")
    if structure.broke_support or structure.near_resistance:
        reasons.append("Structure supports downside continuation")
    if ind.volume_ratio > 1.2:
        reasons.append("Volume expansion confirms sell-off")
    if ind.order_book_imbalance < -0.1:
        reasons.append("Order book ask-side imbalance")

    if len(reasons) < 3:
        return None

    return CandidateSignal(
        asset=snapshot.asset,
        direction="SHORT",
        entry=entry,
        stop_loss=stop,
        take_profits=[tp1, tp2, tp3],
        regime=regime,
        indicators=ind,
        reasons=reasons,
        invalidation=f"Close above {stop:.2f} on {snapshot.timeframe} timeframe",
        risk_reward=(entry - tp1) / risk,
    )


def generate_candidate(
    snapshot: MarketSnapshot,
    regime: RegimeResult,
    ind: IndicatorSet,
) -> CandidateSignal | None:
    structure = read_structure(snapshot)

    if regime.strategy == "MEAN_REVERSION":
        if snapshot.candles[-1].close <= ind.bb_lower and ind.rsi < 35:
            return _build_long_candidate(snapshot, regime, ind, structure)
        if snapshot.candles[-1].close >= ind.bb_upper and ind.rsi > 65:
            return _build_short_candidate(snapshot, regime, ind, structure)
        return None

    if regime.regime == "TRENDING_BULLISH":
        return _build_long_candidate(snapshot, regime, ind, structure)
    if regime.regime == "TRENDING_BEARISH":
        return _build_short_candidate(snapshot, regime, ind, structure)

    if regime.regime == "HIGH_VOLATILITY":
        # Breakout context: trade the side of current order-flow imbalance.
        if ind.order_book_imbalance >= 0:
            return _build_long_candidate(snapshot, regime, ind, structure)
        return _build_short_candidate(snapshot, regime, ind, structure)

    return None
