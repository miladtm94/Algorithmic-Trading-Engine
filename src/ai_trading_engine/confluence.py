from __future__ import annotations

from .config import ConfluenceWeights
from .models import CandidateSignal, ConfluenceBreakdown, MarketSnapshot


def _clip_0_100(v: float) -> float:
    return max(0.0, min(100.0, v))


def _trend_score(signal: CandidateSignal) -> float:
    ind = signal.indicators
    if signal.direction == "LONG":
        ema_score = 100 if ind.ema20 > ind.ema50 > ind.ema200 else 45
        vwap_score = _clip_0_100(100 - abs(ind.vwap - signal.entry) / signal.entry * 9000)
        return _clip_0_100(0.7 * ema_score + 0.3 * vwap_score)
    ema_score = 100 if ind.ema20 < ind.ema50 < ind.ema200 else 45
    vwap_score = _clip_0_100(100 - abs(signal.entry - ind.vwap) / signal.entry * 9000)
    return _clip_0_100(0.7 * ema_score + 0.3 * vwap_score)


def _momentum_score(signal: CandidateSignal) -> float:
    ind = signal.indicators
    if signal.direction == "LONG":
        rsi = _clip_0_100((ind.rsi - 45) * 2.2)
        macd = _clip_0_100(50 + ind.macd_hist * 7000)
        return _clip_0_100(0.55 * rsi + 0.45 * macd)
    rsi = _clip_0_100((55 - ind.rsi) * 2.2)
    macd = _clip_0_100(50 - ind.macd_hist * 7000)
    return _clip_0_100(0.55 * rsi + 0.45 * macd)


def _volume_liquidity_score(signal: CandidateSignal, snapshot: MarketSnapshot) -> float:
    ind = signal.indicators
    volume_score = _clip_0_100(ind.volume_ratio * 50)
    depth_score = _clip_0_100(snapshot.depth_usd / 4_000)
    spread_score = _clip_0_100(100 - snapshot.spread_bps * 4)
    imbalance_bias = ind.order_book_imbalance * 100
    if signal.direction == "SHORT":
        imbalance_bias *= -1
    imbalance_score = _clip_0_100(50 + imbalance_bias)
    return _clip_0_100(
        0.30 * volume_score + 0.30 * depth_score + 0.25 * spread_score + 0.15 * imbalance_score
    )


def _structure_score(signal: CandidateSignal, snapshot: MarketSnapshot) -> float:
    entry = signal.entry
    supports = snapshot.support_levels
    resistances = snapshot.resistance_levels

    if signal.direction == "LONG":
        broken = any(entry > r for r in resistances[:2])
        level_distance = (
            min(abs(entry - s) / entry for s in supports) * 100 if supports else 2.0
        )
        score = 65 + (20 if broken else 0) + max(0.0, 15 - level_distance * 10)
        return _clip_0_100(score)

    broken = any(entry < s for s in supports[-2:])
    level_distance = min(abs(r - entry) / entry for r in resistances) * 100 if resistances else 2.0
    score = 65 + (20 if broken else 0) + max(0.0, 15 - level_distance * 10)
    return _clip_0_100(score)


def _sentiment_score(signal: CandidateSignal, snapshot: MarketSnapshot) -> float:
    sentiment = snapshot.sentiment_score
    if signal.direction == "LONG":
        return _clip_0_100((sentiment + 1) * 50)
    return _clip_0_100((1 - sentiment) * 50)


def score_candidate(
    signal: CandidateSignal,
    snapshot: MarketSnapshot,
    weights: ConfluenceWeights,
) -> ConfluenceBreakdown:
    w = weights.normalized()
    trend_alignment = _trend_score(signal)
    momentum = _momentum_score(signal)
    volume_liquidity = _volume_liquidity_score(signal, snapshot)
    structure = _structure_score(signal, snapshot)
    sentiment = _sentiment_score(signal, snapshot)

    total = (
        trend_alignment * w.trend_alignment
        + momentum * w.momentum
        + volume_liquidity * w.volume_liquidity
        + structure * w.structure
        + sentiment * w.sentiment
    )

    return ConfluenceBreakdown(
        trend_alignment=trend_alignment,
        momentum=momentum,
        volume_liquidity=volume_liquidity,
        structure=structure,
        sentiment=sentiment,
        total_score=total,
    )
