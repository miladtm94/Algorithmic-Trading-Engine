from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .models import Candle, EventRisk, MarketSnapshot, OrderBookSnapshot, PortfolioState


def _build_downtrend_candles(count: int = 240, start_price: float = 3600.0) -> list[Candle]:
    candles: list[Candle] = []
    now = datetime.now(timezone.utc)
    price = start_price
    for i in range(count):
        ts = now - timedelta(hours=(count - i - 1))
        drift = -1.6
        oscillation = ((i % 9) - 4) * 0.9
        close = max(200.0, price + drift + oscillation)
        high = max(price, close) + 7.0
        low = min(price, close) - 8.0
        volume = 2500 + (i % 11) * 180
        if i == count - 1:
            volume *= 1.8
        candles.append(
            Candle(
                timestamp=ts,
                open=price,
                high=high,
                low=low,
                close=close,
                volume=volume,
            )
        )
        price = close
    return candles


def build_demo_snapshot(asset: str = "ETH/USDT") -> MarketSnapshot:
    candles = _build_downtrend_candles()
    last = candles[-1].close
    order_book = OrderBookSnapshot(
        bids=[(last - 0.2, 150), (last - 0.4, 220), (last - 0.6, 300)],
        asks=[(last + 0.2, 350), (last + 0.4, 500), (last + 0.6, 600)],
    )
    return MarketSnapshot(
        asset=asset,
        timeframe="1H",
        candles=candles,
        order_book=order_book,
        spread_bps=4.0,
        depth_usd=950_000.0,
        support_levels=[last * 0.995, last * 0.985, last * 0.972],
        resistance_levels=[last * 1.006, last * 1.013, last * 1.022],
        liquidation_clusters=[last * 0.99, last * 1.01],
        sentiment_score=-0.25,
        source_prices={
            "exchange_a": last,
            "exchange_b": last * 1.0008,
            "exchange_c": last * 0.9993,
        },
        correlation_to_open_positions={"BTC/USDT": 0.56, "SOL/USDT": 0.48},
        events=[EventRisk(name="US Jobless Claims", impact="LOW", minutes_to_event=120)],
    )


def build_demo_portfolio() -> PortfolioState:
    return PortfolioState(
        equity_usd=100_000.0,
        open_positions={"ADA/USDT": "LONG"},
        recent_results=["WIN", "LOSS", "WIN"],
    )
