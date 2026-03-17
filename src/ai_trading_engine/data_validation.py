from __future__ import annotations

from datetime import datetime, timezone
from math import isnan

from .config import DataValidationConfig
from .models import MarketSnapshot


class DataValidationError(ValueError):
    pass


def validate_snapshot(snapshot: MarketSnapshot, cfg: DataValidationConfig) -> None:
    if len(snapshot.candles) < cfg.min_candles:
        raise DataValidationError(
            f"Insufficient candles ({len(snapshot.candles)} < {cfg.min_candles})."
        )

    timestamps = [c.timestamp for c in snapshot.candles]
    if timestamps != sorted(timestamps):
        raise DataValidationError("Candle timestamps are not sorted ascending.")

    last = snapshot.candles[-1]
    now = datetime.now(timezone.utc)
    last_ts = last.timestamp if last.timestamp.tzinfo else last.timestamp.replace(tzinfo=timezone.utc)
    stale_minutes = (now - last_ts).total_seconds() / 60
    if stale_minutes > cfg.stale_after_minutes:
        raise DataValidationError(
            f"Stale market data: last candle is {stale_minutes:.1f} minutes old."
        )

    for idx, c in enumerate(snapshot.candles):
        values = (c.open, c.high, c.low, c.close, c.volume)
        if any(v is None for v in values):
            raise DataValidationError(f"Missing candle value at index {idx}.")
        if any(isnan(v) for v in values):
            raise DataValidationError(f"NaN candle value at index {idx}.")
        if c.low > c.high:
            raise DataValidationError(f"Invalid OHLC bounds at index {idx}.")

    if snapshot.source_prices:
        prices = list(snapshot.source_prices.values())
        ref = sum(prices) / len(prices)
        if ref <= 0:
            raise DataValidationError("Invalid source price reference.")
        max_dev = max(abs(p - ref) / ref * 100 for p in prices)
        if max_dev > cfg.max_price_deviation_pct:
            raise DataValidationError(
                f"Cross-source deviation too high ({max_dev:.2f}% > {cfg.max_price_deviation_pct:.2f}%)."
            )
