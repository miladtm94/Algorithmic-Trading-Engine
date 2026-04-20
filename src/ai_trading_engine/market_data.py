"""Live market data fetcher using ccxt.

Builds MarketSnapshot objects from real exchange data.
ccxt is an optional dependency — import errors are surfaced clearly.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .models import Candle, MarketSnapshot, OrderBookSnapshot

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_TIMEFRAME_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def _retry(fn, *, retries: int = 3, backoff: float = 2.0):
    last_exc: Exception = RuntimeError("unreachable")
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                wait = backoff * (2**attempt)
                logger.warning("Attempt %d/%d failed (%s), retrying in %.1fs", attempt + 1, retries, exc, wait)
                time.sleep(wait)
    raise last_exc


def _candles_from_ohlcv(ohlcv: list[list]) -> list[Candle]:
    return [
        Candle(
            timestamp=datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
        )
        for row in ohlcv
        if row[4] is not None  # skip incomplete candles
    ]


def _swing_levels(
    candles: list[Candle],
    lookback: int = 60,
    n_levels: int = 5,
) -> tuple[list[float], list[float]]:
    """Detect pivot-based support and resistance from recent candles."""
    window = candles[-lookback:] if len(candles) >= lookback else candles
    highs = [c.high for c in window]
    lows = [c.low for c in window]
    current_price = candles[-1].close

    resistances: list[float] = []
    supports: list[float] = []

    for i in range(2, len(highs) - 2):
        if highs[i] > max(highs[i - 2 : i]) and highs[i] > max(highs[i + 1 : i + 3]):
            resistances.append(highs[i])
        if lows[i] < min(lows[i - 2 : i]) and lows[i] < min(lows[i + 1 : i + 3]):
            supports.append(lows[i])

    # Deduplicate within 0.3% tolerance and keep closest to current price
    def dedup(levels: list[float]) -> list[float]:
        out: list[float] = []
        for lvl in sorted(set(levels)):
            if not out or abs(lvl - out[-1]) / out[-1] > 0.003:
                out.append(lvl)
        return out

    res_clean = sorted(dedup(resistances), key=lambda x: abs(x - current_price))[:n_levels]
    sup_clean = sorted(dedup(supports), key=lambda x: abs(x - current_price))[:n_levels]
    return sup_clean, res_clean


class MarketDataFetcher:
    """Fetches OHLCV candles and order book from any ccxt-supported exchange."""

    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: str = "",
        api_secret: str = "",
        sandbox: bool = False,
    ) -> None:
        try:
            import ccxt  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "ccxt is required for live data. Install with: pip install ccxt"
            ) from exc

        exchange_class = getattr(ccxt, exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"Unknown exchange: {exchange_id!r}. See ccxt docs for supported exchanges.")

        self._exchange = exchange_class(
            {
                "apiKey": api_key or None,
                "secret": api_secret or None,
                "enableRateLimit": True,
            }
        )
        if sandbox:
            self._exchange.set_sandbox_mode(True)

        logger.info("MarketDataFetcher initialised: exchange=%s sandbox=%s", exchange_id, sandbox)

    def fetch_snapshot(
        self,
        asset: str,
        timeframe: str = "1h",
        candle_limit: int = 250,
    ) -> MarketSnapshot:
        logger.debug("Fetching snapshot: %s %s", asset, timeframe)

        ohlcv = _retry(lambda: self._exchange.fetch_ohlcv(asset, timeframe, limit=candle_limit))
        candles = _candles_from_ohlcv(ohlcv)

        if not candles:
            raise ValueError(f"No candle data returned for {asset} ({timeframe})")

        raw_ob = _retry(lambda: self._exchange.fetch_order_book(asset, limit=20))
        bids: list[tuple[float, float]] = [(float(b[0]), float(b[1])) for b in raw_ob["bids"][:10]]
        asks: list[tuple[float, float]] = [(float(a[0]), float(a[1])) for a in raw_ob["asks"][:10]]
        order_book = OrderBookSnapshot(bids=bids, asks=asks)

        best_bid = bids[0][0] if bids else candles[-1].close
        best_ask = asks[0][0] if asks else candles[-1].close
        mid = (best_bid + best_ask) / 2
        spread_bps = ((best_ask - best_bid) / mid) * 10_000 if mid > 0 else 0.0

        bid_depth = sum(p * s for p, s in bids)
        ask_depth = sum(p * s for p, s in asks)
        depth_usd = (bid_depth + ask_depth) / 2

        supports, resistances = _swing_levels(candles)

        total_bid_size = sum(s for _, s in bids)
        total_ask_size = sum(s for _, s in asks)
        total_size = total_bid_size + total_ask_size
        sentiment_score = (total_bid_size - total_ask_size) / total_size if total_size > 0 else 0.0

        return MarketSnapshot(
            asset=asset,
            timeframe=timeframe,
            candles=candles,
            order_book=order_book,
            spread_bps=spread_bps,
            depth_usd=depth_usd,
            support_levels=supports,
            resistance_levels=resistances,
            liquidation_clusters=[],
            sentiment_score=float(sentiment_score),
            source_prices={"exchange": candles[-1].close},
            correlation_to_open_positions={},
            events=[],
        )

    def fetch_historical_candles(
        self,
        asset: str,
        timeframe: str = "1h",
        since_ms: int | None = None,
        limit: int = 1000,
    ) -> list[Candle]:
        """Fetch up to `limit` historical candles starting from `since_ms`."""
        ohlcv = _retry(
            lambda: self._exchange.fetch_ohlcv(asset, timeframe, since=since_ms, limit=limit)
        )
        return _candles_from_ohlcv(ohlcv)
