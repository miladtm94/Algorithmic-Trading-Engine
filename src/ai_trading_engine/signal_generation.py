from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from .models import (
    CandidateSignal,
    Candle,
    Direction,
    IndicatorSet,
    MarketSnapshot,
    RegimeResult,
    SetupFamily,
)

FAMILY_TREND_PULLBACK: SetupFamily = "TREND_PULLBACK_CONTINUATION"
FAMILY_BREAKOUT_RETEST: SetupFamily = "BREAKOUT_RETEST_CONTINUATION"
FAMILY_RANGE_REJECTION: SetupFamily = "RANGE_REJECTION_MEAN_REVERSION"
FAMILY_FAILED_BREAKOUT: SetupFamily = "FAILED_BREAKOUT_REVERSAL"


@dataclass(slots=True)
class StructureRead:
    nearest_support: float
    nearest_resistance: float
    broke_support: bool
    broke_resistance: bool
    near_support: bool
    near_resistance: bool
    range_high: float
    range_low: float
    range_mid: float
    range_width_pct: float
    range_position: float
    prior_high: float
    prior_low: float
    is_compressed: bool


def _range_pct(candle: Candle) -> float:
    return (candle.high - candle.low) / candle.close if candle.close else 0.0


def _body(candle: Candle) -> float:
    return abs(candle.close - candle.open)


def _upper_wick(candle: Candle) -> float:
    return max(0.0, candle.high - max(candle.open, candle.close))


def _lower_wick(candle: Candle) -> float:
    return max(0.0, min(candle.open, candle.close) - candle.low)


def _bullish_rejection(candle: Candle) -> bool:
    return candle.close > candle.open and _lower_wick(candle) > max(_body(candle) * 1.2, candle.close * 0.001)


def _bearish_rejection(candle: Candle) -> bool:
    return candle.close < candle.open and _upper_wick(candle) > max(_body(candle) * 1.2, candle.close * 0.001)


def _mean_range_pct(candles: list[Candle]) -> float:
    if not candles:
        return 0.0
    return mean(_range_pct(candle) for candle in candles)


def read_structure(snapshot: MarketSnapshot) -> StructureRead:
    candles = snapshot.candles
    last = candles[-1].close
    recent = candles[-20:] if len(candles) >= 20 else candles
    prior = recent[:-1] or recent
    supports = sorted(snapshot.support_levels)
    resistances = sorted(snapshot.resistance_levels)

    range_high = max(candle.high for candle in recent)
    range_low = min(candle.low for candle in recent)
    range_mid = (range_high + range_low) / 2
    range_width = range_high - range_low
    range_width_pct = range_width / last if last else 0.0
    range_position = (last - range_low) / range_width if range_width > 0 else 0.5

    nearest_support = max(
        (support for support in supports if support <= last),
        default=min(range_low, last * 0.99),
    )
    nearest_resistance = min(
        (resistance for resistance in resistances if resistance >= last),
        default=max(range_high, last * 1.01),
    )
    broke_support = any(last < support for support in supports[-2:]) if supports else last < range_low
    broke_resistance = (
        any(last > resistance for resistance in resistances[:2]) if resistances else last > range_high
    )
    near_support = abs(last - nearest_support) / last < 0.004 if last else False
    near_resistance = abs(nearest_resistance - last) / last < 0.004 if last else False

    prior_high = max(candle.high for candle in prior)
    prior_low = min(candle.low for candle in prior)
    recent_avg_range = _mean_range_pct(recent[-5:]) if len(recent) >= 5 else _mean_range_pct(recent)
    prior_avg_range = _mean_range_pct(recent[-15:-5]) if len(recent) >= 15 else _mean_range_pct(prior)
    is_compressed = prior_avg_range > 0 and recent_avg_range < prior_avg_range * 0.8

    return StructureRead(
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        broke_support=broke_support,
        broke_resistance=broke_resistance,
        near_support=near_support,
        near_resistance=near_resistance,
        range_high=range_high,
        range_low=range_low,
        range_mid=range_mid,
        range_width_pct=range_width_pct,
        range_position=range_position,
        prior_high=prior_high,
        prior_low=prior_low,
        is_compressed=is_compressed,
    )


def _candidate(
    *,
    snapshot: MarketSnapshot,
    regime: RegimeResult,
    ind: IndicatorSet,
    direction: Direction,
    stop_loss: float,
    take_profits: list[float],
    reasons: list[str],
    invalidation: str,
    setup_family: SetupFamily,
    setup_quality: float,
    max_hold_bars: int,
    reference_level: float | None,
) -> CandidateSignal | None:
    entry = snapshot.candles[-1].close
    if direction == "LONG":
        risk = entry - stop_loss
        risk_reward = (take_profits[0] - entry) / risk if risk > 0 and take_profits else 0.0
    else:
        risk = stop_loss - entry
        risk_reward = (entry - take_profits[0]) / risk if risk > 0 and take_profits else 0.0

    if risk <= 0 or risk_reward <= 0:
        return None

    return CandidateSignal(
        asset=snapshot.asset,
        direction=direction,
        entry=entry,
        stop_loss=stop_loss,
        take_profits=take_profits,
        regime=regime,
        indicators=ind,
        reasons=reasons,
        invalidation=invalidation,
        risk_reward=risk_reward,
        setup_family=setup_family,
        setup_quality=round(setup_quality, 4),
        max_hold_bars=max_hold_bars,
        reference_level=reference_level,
    )


def _build_trend_pullback_long(
    snapshot: MarketSnapshot,
    regime: RegimeResult,
    ind: IndicatorSet,
    structure: StructureRead,
) -> CandidateSignal | None:
    candles = snapshot.candles
    last = candles[-1]
    entry = last.close
    atr = max(ind.atr, entry * 0.003)
    pullback_window = candles[-8:] if len(candles) >= 8 else candles
    pullback_low = min(candle.low for candle in pullback_window)
    pullback_depth = entry - pullback_low
    touched_support = pullback_low <= ind.ema20 + 1.0 * atr
    reclaimed_support = entry >= ind.ema20 - 0.35 * atr or entry >= ind.vwap - 0.2 * atr

    if not (ind.ema20 > ind.ema50 > ind.ema200 and ind.macd_hist > 0 and touched_support and reclaimed_support):
        return None
    if pullback_depth < 0.15 * atr or pullback_depth > 3.25 * atr:
        return None
    if pullback_low < ind.ema50 - 1.25 * atr:
        return None
    if not (
        last.close >= last.open * 0.998
        and (
            _lower_wick(last) > _body(last) * 0.15
            or last.close >= ind.vwap
            or last.close >= ind.ema20 - 0.15 * atr
        )
    ):
        return None

    stop = min(pullback_low - 0.2 * atr, structure.nearest_support - 0.1 * atr)
    risk = entry - stop
    if risk <= 0:
        return None
    if (
        not structure.broke_resistance
        and structure.nearest_resistance > entry
        and (structure.nearest_resistance - entry) < 0.8 * risk
    ):
        return None

    reasons = [
        "Trend pullback continuation in bullish EMA stack",
        "Price reclaimed pullback support near EMA20/VWAP",
        "Bullish candle confirms continuation after controlled retrace",
    ]
    quality = 70.0
    if pullback_low <= ind.ema20 + 0.2 * atr:
        quality += 6.0
        reasons.append("Pullback tagged EMA20 support cleanly")
    if ind.volume_ratio >= 1.05:
        quality += 4.0
        reasons.append("Volume held up on reclaim")
    if structure.near_support:
        quality += 3.0
        reasons.append("Nearest support is still close below entry")

    return _candidate(
        snapshot=snapshot,
        regime=regime,
        ind=ind,
        direction="LONG",
        stop_loss=stop,
        take_profits=[entry + 2.0 * risk, entry + 3.0 * risk, entry + 4.0 * risk],
        reasons=reasons,
        invalidation=f"Continuation fails if price closes below pullback low {stop:.2f}",
        setup_family=FAMILY_TREND_PULLBACK,
        setup_quality=quality,
        max_hold_bars=36,
        reference_level=ind.ema20,
    )


def _build_trend_pullback_short(
    snapshot: MarketSnapshot,
    regime: RegimeResult,
    ind: IndicatorSet,
    structure: StructureRead,
) -> CandidateSignal | None:
    candles = snapshot.candles
    last = candles[-1]
    entry = last.close
    atr = max(ind.atr, entry * 0.003)
    pullback_window = candles[-8:] if len(candles) >= 8 else candles
    pullback_high = max(candle.high for candle in pullback_window)
    pullback_depth = pullback_high - entry
    touched_resistance = pullback_high >= ind.ema20 - 1.0 * atr
    reclaimed_resistance = entry <= ind.ema20 + 0.35 * atr or entry <= ind.vwap + 0.2 * atr

    if not (
        ind.ema20 < ind.ema50 < ind.ema200
        and ind.macd_hist < 0
        and touched_resistance
        and reclaimed_resistance
    ):
        return None
    if pullback_depth < 0.15 * atr or pullback_depth > 3.25 * atr:
        return None
    if pullback_high > ind.ema50 + 1.25 * atr:
        return None
    if not (
        last.close <= last.open * 1.002
        and (
            _upper_wick(last) > _body(last) * 0.15
            or last.close <= ind.vwap
            or last.close <= ind.ema20 + 0.15 * atr
        )
    ):
        return None

    stop = max(pullback_high + 0.2 * atr, structure.nearest_resistance + 0.1 * atr)
    risk = stop - entry
    if risk <= 0:
        return None
    if (
        not structure.broke_support
        and structure.nearest_support < entry
        and (entry - structure.nearest_support) < 0.8 * risk
    ):
        return None

    reasons = [
        "Trend pullback continuation in bearish EMA stack",
        "Price rejected pullback resistance near EMA20/VWAP",
        "Bearish candle confirms continuation after controlled bounce",
    ]
    quality = 70.0
    if pullback_high >= ind.ema20 - 0.2 * atr:
        quality += 6.0
        reasons.append("Pullback tagged EMA20 resistance cleanly")
    if ind.volume_ratio >= 1.05:
        quality += 4.0
        reasons.append("Volume held up on rejection")
    if structure.near_resistance:
        quality += 3.0
        reasons.append("Nearest resistance is still close above entry")

    return _candidate(
        snapshot=snapshot,
        regime=regime,
        ind=ind,
        direction="SHORT",
        stop_loss=stop,
        take_profits=[entry - 2.0 * risk, entry - 3.0 * risk, entry - 4.0 * risk],
        reasons=reasons,
        invalidation=f"Continuation fails if price closes above pullback high {stop:.2f}",
        setup_family=FAMILY_TREND_PULLBACK,
        setup_quality=quality,
        max_hold_bars=36,
        reference_level=ind.ema20,
    )


def _build_breakout_retest_long(
    snapshot: MarketSnapshot,
    regime: RegimeResult,
    ind: IndicatorSet,
    structure: StructureRead,
) -> CandidateSignal | None:
    candles = snapshot.candles
    last = candles[-1]
    entry = last.close
    atr = max(ind.atr, entry * 0.003)
    breakout_level = structure.prior_high
    extension = entry - breakout_level

    if breakout_level <= 0 or extension <= 0 or extension > 1.25 * atr:
        return None
    if last.low > breakout_level + 0.25 * atr:
        return None
    if not (
        last.close > breakout_level
        and last.close > last.open
        and ind.volume_ratio >= 1.05
    ):
        return None
    if not (structure.is_compressed or ind.bb_width_pct <= 0.04):
        return None
    # Require trend-stack alignment inside the rule. The regime classifier
    # alone allowed long breakout-retests to fire in chop, producing the
    # -1.20 net R long-side damage on the 2026-04-25 ETH/USDT 1h test slice.
    if not (ind.ema20 > ind.ema50 and entry > ind.ema50):
        return None

    stop = min(last.low - 0.2 * atr, breakout_level - 0.35 * atr)
    risk = entry - stop
    if risk <= 0:
        return None
    if (
        not structure.broke_resistance
        and structure.nearest_resistance > entry
        and (structure.nearest_resistance - entry) < 0.8 * risk
    ):
        return None

    measured_move = breakout_level + (structure.range_high - structure.range_low)
    tp1 = max(entry + 1.8 * risk, measured_move)

    reasons = [
        "Breakout retest continuation through prior range high",
        "Price accepted back above the breakout level after retest",
        "Volume expanded enough to support continuation",
    ]
    quality = 74.0
    if structure.is_compressed:
        quality += 6.0
        reasons.append("Setup emerged from prior range compression")
    if last.close >= last.high - 0.3 * atr:
        quality += 4.0
        reasons.append("Retest candle closed strong near the highs")

    return _candidate(
        snapshot=snapshot,
        regime=regime,
        ind=ind,
        direction="LONG",
        stop_loss=stop,
        take_profits=[tp1, entry + 3.0 * risk, entry + 4.0 * risk],
        reasons=reasons,
        invalidation=f"Breakout retest fails below {stop:.2f}",
        setup_family=FAMILY_BREAKOUT_RETEST,
        setup_quality=quality,
        max_hold_bars=24,
        reference_level=breakout_level,
    )


def _build_breakout_retest_short(
    snapshot: MarketSnapshot,
    regime: RegimeResult,
    ind: IndicatorSet,
    structure: StructureRead,
) -> CandidateSignal | None:
    candles = snapshot.candles
    last = candles[-1]
    entry = last.close
    atr = max(ind.atr, entry * 0.003)
    breakout_level = structure.prior_low
    extension = breakout_level - entry

    if breakout_level <= 0 or extension <= 0 or extension > 1.25 * atr:
        return None
    if last.high < breakout_level - 0.25 * atr:
        return None
    if not (
        last.close < breakout_level
        and last.close < last.open
        and ind.volume_ratio >= 1.05
    ):
        return None
    if not (structure.is_compressed or ind.bb_width_pct <= 0.04):
        return None
    # Symmetric trend-stack alignment gate for shorts.
    if not (ind.ema20 < ind.ema50 and entry < ind.ema50):
        return None

    stop = max(last.high + 0.2 * atr, breakout_level + 0.35 * atr)
    risk = stop - entry
    if risk <= 0:
        return None
    if (
        not structure.broke_support
        and structure.nearest_support < entry
        and (entry - structure.nearest_support) < 0.8 * risk
    ):
        return None

    measured_move = breakout_level - (structure.range_high - structure.range_low)
    tp1 = min(entry - 1.8 * risk, measured_move)

    reasons = [
        "Breakdown retest continuation through prior range low",
        "Price accepted back below the breakdown level after retest",
        "Volume expanded enough to support continuation",
    ]
    quality = 74.0
    if structure.is_compressed:
        quality += 6.0
        reasons.append("Setup emerged from prior range compression")
    if last.close <= last.low + 0.3 * atr:
        quality += 4.0
        reasons.append("Retest candle closed strong near the lows")

    return _candidate(
        snapshot=snapshot,
        regime=regime,
        ind=ind,
        direction="SHORT",
        stop_loss=stop,
        take_profits=[tp1, entry - 3.0 * risk, entry - 4.0 * risk],
        reasons=reasons,
        invalidation=f"Breakdown retest fails above {stop:.2f}",
        setup_family=FAMILY_BREAKOUT_RETEST,
        setup_quality=quality,
        max_hold_bars=24,
        reference_level=breakout_level,
    )


def _build_range_rejection_long(
    snapshot: MarketSnapshot,
    regime: RegimeResult,
    ind: IndicatorSet,
    structure: StructureRead,
) -> CandidateSignal | None:
    last = snapshot.candles[-1]
    entry = last.close
    atr = max(ind.atr, entry * 0.003)

    # Require a real touch of the structural range extreme. Mid-range mean
    # reversion off the BB envelope alone has materially weaker edge, so it is
    # no longer enough by itself to qualify the setup.
    if last.low > structure.range_low + 0.25 * atr:
        return None
    if not _bullish_rejection(last):
        return None

    stop = min(last.low - 0.15 * atr, structure.range_low - 0.2 * atr)
    risk = entry - stop
    if risk <= 0:
        return None

    target_mid = structure.range_mid
    target_far = max(target_mid + 0.5 * risk, structure.range_high - 0.1 * atr)
    target_primary = max(target_mid, entry + 1.1 * risk)
    if target_far <= entry + 1.0 * risk:
        return None
    if target_far <= target_primary:
        return None

    reasons = [
        "Range rejection mean reversion from range support",
        "Bullish rejection wick shows failure to accept lower prices",
        "Targeting mean reversion back toward the range midpoint",
    ]
    quality = 72.0
    if structure.range_width_pct >= ind.atr_pct * 1.2:
        quality += 4.0
        reasons.append("Range is wide enough to pay for mean-reversion risk")
    if last.close > structure.range_low:
        quality += 4.0
        reasons.append("Close finished back inside the range")
    if entry <= ind.bb_lower * 1.01:
        quality += 3.0
        reasons.append("Close also tagged the lower Bollinger band")

    return _candidate(
        snapshot=snapshot,
        regime=regime,
        ind=ind,
        direction="LONG",
        stop_loss=stop,
        take_profits=[target_primary, target_far],
        reasons=reasons,
        invalidation=f"Mean reversion fails below support rejection low {stop:.2f}",
        setup_family=FAMILY_RANGE_REJECTION,
        setup_quality=quality,
        max_hold_bars=12,
        reference_level=structure.range_low,
    )


def _build_range_rejection_short(
    snapshot: MarketSnapshot,
    regime: RegimeResult,
    ind: IndicatorSet,
    structure: StructureRead,
) -> CandidateSignal | None:
    last = snapshot.candles[-1]
    entry = last.close
    atr = max(ind.atr, entry * 0.003)

    # Require a real touch of the structural range extreme. Mid-range mean
    # reversion off the BB envelope alone has materially weaker edge, so it is
    # no longer enough by itself to qualify the setup.
    if last.high < structure.range_high - 0.25 * atr:
        return None
    if not _bearish_rejection(last):
        return None

    stop = max(last.high + 0.15 * atr, structure.range_high + 0.2 * atr)
    risk = stop - entry
    if risk <= 0:
        return None

    target_mid = structure.range_mid
    target_far = min(target_mid - 0.5 * risk, structure.range_low + 0.1 * atr)
    target_primary = min(target_mid, entry - 1.1 * risk)
    if target_far >= entry - 1.0 * risk:
        return None
    if target_far >= target_primary:
        return None

    reasons = [
        "Range rejection mean reversion from range resistance",
        "Bearish rejection wick shows failure to accept higher prices",
        "Targeting mean reversion back toward the range midpoint",
    ]
    quality = 72.0
    if structure.range_width_pct >= ind.atr_pct * 1.2:
        quality += 4.0
        reasons.append("Range is wide enough to pay for mean-reversion risk")
    if last.close < structure.range_high:
        quality += 4.0
        reasons.append("Close finished back inside the range")
    if entry >= ind.bb_upper * 0.99:
        quality += 3.0
        reasons.append("Close also tagged the upper Bollinger band")

    return _candidate(
        snapshot=snapshot,
        regime=regime,
        ind=ind,
        direction="SHORT",
        stop_loss=stop,
        take_profits=[target_primary, target_far],
        reasons=reasons,
        invalidation=f"Mean reversion fails above resistance rejection high {stop:.2f}",
        setup_family=FAMILY_RANGE_REJECTION,
        setup_quality=quality,
        max_hold_bars=12,
        reference_level=structure.range_high,
    )


def _build_failed_breakout_short(
    snapshot: MarketSnapshot,
    regime: RegimeResult,
    ind: IndicatorSet,
    structure: StructureRead,
) -> CandidateSignal | None:
    last = snapshot.candles[-1]
    entry = last.close
    atr = max(ind.atr, entry * 0.003)
    trap_level = structure.prior_high

    if last.high <= trap_level + 0.1 * atr:
        return None
    if last.close >= trap_level or not _bearish_rejection(last):
        return None

    stop = last.high + 0.15 * atr
    risk = stop - entry
    if risk <= 0:
        return None

    target_mid = structure.range_mid
    target_far = min(structure.range_low + 0.1 * atr, entry - 2.0 * risk)
    target_primary = min(target_mid, entry - 1.1 * risk)
    if target_far >= entry - 1.0 * risk:
        return None
    if target_far >= target_primary:
        return None

    reasons = [
        "Failed breakout reversal after bull-trap above prior range high",
        "Price probed above resistance but closed back inside the range",
        "Targeting reversal back toward range value after trap failure",
    ]
    quality = 78.0
    if _upper_wick(last) > _body(last) * 1.6:
        quality += 5.0
        reasons.append("Upper wick shows aggressive rejection of the breakout")
    if ind.rsi >= 58:
        quality += 3.0
        reasons.append("Trap formed after an extended momentum push")

    return _candidate(
        snapshot=snapshot,
        regime=regime,
        ind=ind,
        direction="SHORT",
        stop_loss=stop,
        take_profits=[target_primary, target_far],
        reasons=reasons,
        invalidation=f"Failure reversal is invalid above trap wick {stop:.2f}",
        setup_family=FAMILY_FAILED_BREAKOUT,
        setup_quality=quality,
        max_hold_bars=18,
        reference_level=trap_level,
    )


def _build_failed_breakout_long(
    snapshot: MarketSnapshot,
    regime: RegimeResult,
    ind: IndicatorSet,
    structure: StructureRead,
) -> CandidateSignal | None:
    last = snapshot.candles[-1]
    entry = last.close
    atr = max(ind.atr, entry * 0.003)
    trap_level = structure.prior_low

    if last.low >= trap_level - 0.1 * atr:
        return None
    if last.close <= trap_level or not _bullish_rejection(last):
        return None

    stop = last.low - 0.15 * atr
    risk = entry - stop
    if risk <= 0:
        return None

    target_mid = structure.range_mid
    target_far = max(structure.range_high - 0.1 * atr, entry + 2.0 * risk)
    target_primary = max(target_mid, entry + 1.1 * risk)
    if target_far <= entry + 1.0 * risk:
        return None
    if target_far <= target_primary:
        return None

    reasons = [
        "Failed breakdown reversal after bear-trap below prior range low",
        "Price probed below support but closed back inside the range",
        "Targeting reversal back toward range value after trap failure",
    ]
    quality = 78.0
    if _lower_wick(last) > _body(last) * 1.6:
        quality += 5.0
        reasons.append("Lower wick shows aggressive rejection of the breakdown")
    if ind.rsi <= 42:
        quality += 3.0
        reasons.append("Trap formed after an extended downside push")

    return _candidate(
        snapshot=snapshot,
        regime=regime,
        ind=ind,
        direction="LONG",
        stop_loss=stop,
        take_profits=[target_primary, target_far],
        reasons=reasons,
        invalidation=f"Failure reversal is invalid below trap wick {stop:.2f}",
        setup_family=FAMILY_FAILED_BREAKOUT,
        setup_quality=quality,
        max_hold_bars=18,
        reference_level=trap_level,
    )


def _choose_candidate(candidates: list[CandidateSignal]) -> CandidateSignal | None:
    if not candidates:
        return None
    family_priority = {
        FAMILY_BREAKOUT_RETEST: 4,
        FAMILY_FAILED_BREAKOUT: 3,
        FAMILY_TREND_PULLBACK: 2,
        FAMILY_RANGE_REJECTION: 1,
    }
    return max(
        candidates,
        key=lambda candidate: (
            candidate.setup_quality,
            family_priority.get(candidate.setup_family, 0),
            candidate.risk_reward,
        ),
    )


def generate_candidate(
    snapshot: MarketSnapshot,
    regime: RegimeResult,
    ind: IndicatorSet,
) -> CandidateSignal | None:
    structure = read_structure(snapshot)
    candidates: list[CandidateSignal] = []

    if regime.regime == "TRENDING_BULLISH":
        for candidate in (
            _build_breakout_retest_long(snapshot, regime, ind, structure),
            _build_trend_pullback_long(snapshot, regime, ind, structure),
        ):
            if candidate is not None:
                candidates.append(candidate)
        return _choose_candidate(candidates)

    if regime.regime == "TRENDING_BEARISH":
        for candidate in (
            _build_breakout_retest_short(snapshot, regime, ind, structure),
            _build_trend_pullback_short(snapshot, regime, ind, structure),
        ):
            if candidate is not None:
                candidates.append(candidate)
        return _choose_candidate(candidates)

    if regime.strategy == "MEAN_REVERSION":
        for candidate in (
            _build_range_rejection_long(snapshot, regime, ind, structure),
            _build_range_rejection_short(snapshot, regime, ind, structure),
            _build_failed_breakout_long(snapshot, regime, ind, structure),
            _build_failed_breakout_short(snapshot, regime, ind, structure),
        ):
            if candidate is not None:
                candidates.append(candidate)
        return _choose_candidate(candidates)

    if regime.regime == "HIGH_VOLATILITY":
        for candidate in (
            _build_breakout_retest_long(snapshot, regime, ind, structure),
            _build_breakout_retest_short(snapshot, regime, ind, structure),
            _build_failed_breakout_long(snapshot, regime, ind, structure),
            _build_failed_breakout_short(snapshot, regime, ind, structure),
        ):
            if candidate is not None:
                candidates.append(candidate)
        return _choose_candidate(candidates)

    return None
