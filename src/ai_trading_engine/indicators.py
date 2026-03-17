from __future__ import annotations

import math
from statistics import mean

from .models import Candle, IndicatorSet, OrderBookSnapshot


def _ema(values: list[float], period: int) -> float:
    if len(values) < period:
        return values[-1]
    multiplier = 2 / (period + 1)
    ema_value = mean(values[:period])
    for price in values[period:]:
        ema_value = (price - ema_value) * multiplier + ema_value
    return ema_value


def _rsi(values: list[float], period: int = 14) -> float:
    if len(values) <= period:
        return 50.0
    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        gains.append(max(change, 0.0))
        losses.append(abs(min(change, 0.0)))

    avg_gain = mean(gains[:period])
    avg_loss = mean(losses[:period])
    if avg_loss == 0:
        return 100.0

    for i in range(period, len(gains)):
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _macd(values: list[float]) -> tuple[float, float, float]:
    macd_line_series: list[float] = []
    for i in range(26, len(values) + 1):
        window = values[:i]
        macd_line_series.append(_ema(window, 12) - _ema(window, 26))
    if not macd_line_series:
        return 0.0, 0.0, 0.0
    signal = _ema(macd_line_series, 9)
    macd_line = macd_line_series[-1]
    hist = macd_line - signal
    return macd_line, signal, hist


def _atr(candles: list[Candle], period: int = 14) -> float:
    if len(candles) <= 1:
        return 0.0
    true_ranges: list[float] = []
    for idx in range(1, len(candles)):
        candle = candles[idx]
        prev_close = candles[idx - 1].close
        tr = max(
            candle.high - candle.low,
            abs(candle.high - prev_close),
            abs(candle.low - prev_close),
        )
        true_ranges.append(tr)

    if len(true_ranges) < period:
        return mean(true_ranges) if true_ranges else 0.0

    atr_value = mean(true_ranges[:period])
    for tr in true_ranges[period:]:
        atr_value = ((atr_value * (period - 1)) + tr) / period
    return atr_value


def _bollinger(values: list[float], period: int = 20, stdevs: float = 2.0) -> tuple[float, float, float]:
    if len(values) < period:
        middle = values[-1]
        return middle, middle, middle
    window = values[-period:]
    middle = mean(window)
    variance = sum((v - middle) ** 2 for v in window) / len(window)
    sigma = math.sqrt(variance)
    upper = middle + stdevs * sigma
    lower = middle - stdevs * sigma
    return upper, middle, lower


def _vwap(candles: list[Candle], period: int = 50) -> float:
    window = candles[-period:] if len(candles) >= period else candles
    total_volume = sum(c.volume for c in window)
    if total_volume <= 0:
        return window[-1].close
    return sum(((c.high + c.low + c.close) / 3) * c.volume for c in window) / total_volume


def _order_book_imbalance(order_book: OrderBookSnapshot, depth: int = 10) -> float:
    bid_volume = sum(size for _, size in order_book.bids[:depth])
    ask_volume = sum(size for _, size in order_book.asks[:depth])
    denom = bid_volume + ask_volume
    if denom == 0:
        return 0.0
    return (bid_volume - ask_volume) / denom


def compute_indicators(candles: list[Candle], order_book: OrderBookSnapshot) -> IndicatorSet:
    closes = [c.close for c in candles]
    volumes = [c.volume for c in candles]
    last_close = closes[-1]

    ema20 = _ema(closes, 20)
    ema50 = _ema(closes, 50)
    ema200 = _ema(closes, 200)
    vwap = _vwap(candles, 50)
    rsi = _rsi(closes, 14)
    macd, macd_signal, macd_hist = _macd(closes)
    atr = _atr(candles, 14)
    atr_pct = (atr / last_close) if last_close else 0.0
    bb_upper, bb_middle, bb_lower = _bollinger(closes, 20, 2.0)
    bb_width_pct = ((bb_upper - bb_lower) / bb_middle) if bb_middle else 0.0
    avg_volume = mean(volumes[-20:]) if len(volumes) >= 20 else mean(volumes)
    volume_ratio = (volumes[-1] / avg_volume) if avg_volume else 1.0
    imbalance = _order_book_imbalance(order_book, 10)

    return IndicatorSet(
        ema20=ema20,
        ema50=ema50,
        ema200=ema200,
        vwap=vwap,
        rsi=rsi,
        macd=macd,
        macd_signal=macd_signal,
        macd_hist=macd_hist,
        atr=atr,
        atr_pct=atr_pct,
        bb_upper=bb_upper,
        bb_middle=bb_middle,
        bb_lower=bb_lower,
        bb_width_pct=bb_width_pct,
        avg_volume=avg_volume,
        volume_ratio=volume_ratio,
        order_book_imbalance=imbalance,
    )
