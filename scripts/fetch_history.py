#!/usr/bin/env python
"""Download multi-year OHLCV history from any ccxt-supported exchange.

Paginates automatically so you can request 2-5+ years of candles regardless
of the exchange's per-request limit (typically 500-720 candles).

Data is saved to data/historical/<ASSET>_<TIMEFRAME>.csv

EXCHANGE COMPATIBILITY
----------------------
Binance / Bybit / OKX  — full historical pagination, recommended for training data
Kraken                  — OHLCV endpoint ignores the `since` parameter and always
                          returns the most recent 720 candles (~30 days for 1h).
                          Use Binance/Bybit for multi-year downloads instead, then
                          use your Kraken Pro account for live/paper trading.

Usage:
  python scripts/fetch_history.py --asset ETH/USDT --exchange binance --years 2
  python scripts/fetch_history.py --asset BTC/USDT --exchange binance --years 5
  python scripts/fetch_history.py --asset ETH/USDT --exchange bybit   --years 3
  python scripts/fetch_history.py --asset ETH/USDT --exchange binance --years 2 --timeframe 4h
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "historical"

_TIMEFRAME_MS: dict[str, int] = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


_KRAKEN_OHLCV_WARNING = """
  WARNING: Kraken's OHLCV API ignores the 'since' parameter and always
  returns the most recent ~720 candles (~30 days for 1h data), regardless
  of how far back you request. Multi-year downloads will NOT work on Kraken.

  Use Binance or Bybit for historical training data:
    make fetch-history exchange=binance asset=ETH/USDT years=2
    make fetch-history exchange=bybit   asset=ETH/USDT years=2

  You can still use your Kraken Pro account for live/paper trading.
"""


def fetch_full_history(
    exchange_id: str,
    asset: str,
    timeframe: str,
    years: float,
) -> list[dict]:
    from ai_trading_engine.market_data import MarketDataFetcher  # noqa: PLC0415

    if exchange_id.lower() == "kraken":
        print(_KRAKEN_OHLCV_WARNING)
        sys.exit(1)

    tf_ms = _TIMEFRAME_MS.get(timeframe)
    if tf_ms is None:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}. Options: {list(_TIMEFRAME_MS)}")

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - int(years * 365.25 * 24 * 3_600 * 1_000)
    batch_size = 500

    fetcher = MarketDataFetcher(exchange_id)
    all_rows: list[dict] = []
    seen_ts: set[int] = set()
    since = start_ms

    logger.info(
        "Fetching %.4g years of %s [%s] from %s  (from %s)",
        years, asset, timeframe, exchange_id,
        datetime.fromtimestamp(start_ms / 1000, tz=UTC).strftime("%Y-%m-%d"),
    )

    while since < now_ms:
        try:
            candles = fetcher.fetch_historical_candles(
                asset, timeframe, since_ms=since, limit=batch_size
            )
        except Exception as exc:
            logger.warning("Fetch error (since=%d): %s — retrying in 5 s", since, exc)
            time.sleep(5)
            continue

        if not candles:
            logger.info("Exchange returned no more candles — done.")
            break

        new_count = 0
        for c in candles:
            ts_ms = int(c.timestamp.timestamp() * 1000)
            if ts_ms in seen_ts:
                continue
            seen_ts.add(ts_ms)
            all_rows.append({
                "timestamp": c.timestamp.isoformat(),
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            })
            new_count += 1

        last_ts = int(candles[-1].timestamp.timestamp() * 1000)
        logger.info(
            "  +%d candles  (total %d)  last: %s",
            new_count,
            len(all_rows),
            candles[-1].timestamp.strftime("%Y-%m-%d %H:%M UTC"),
        )

        if last_ts <= since:
            # Exchange not advancing — we've reached the end of available history
            break

        since = last_ts + tf_ms
        time.sleep(0.4)  # respect rate limits

    return sorted(all_rows, key=lambda r: r["timestamp"])


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["timestamp", "open", "high", "low", "close", "volume"]
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved %d candles → %s", len(rows), path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download multi-year OHLCV history")
    parser.add_argument("--asset", default=os.getenv("DEFAULT_ASSET", "ETH/USDT"))
    parser.add_argument(
        "--timeframe", default=os.getenv("DEFAULT_TIMEFRAME", "1h"),
        choices=list(_TIMEFRAME_MS),
    )
    parser.add_argument("--years", type=float, default=2.0, help="Years of history to fetch")
    # Default to binance — do NOT fall back to EXCHANGE env var because that
    # is typically set to the user's live-trading exchange (e.g. kraken) and
    # its API credentials would be sent to a different exchange and rejected.
    parser.add_argument("--exchange", default="binance")
    args = parser.parse_args()

    # Historical OHLCV is a public endpoint on every exchange — no API key needed.
    rows = fetch_full_history(
        args.exchange, args.asset, args.timeframe, args.years
    )

    if not rows:
        print("ERROR: No data was retrieved. Check your asset name and exchange.")
        sys.exit(1)

    safe_asset = args.asset.replace("/", "_")
    filename = f"{safe_asset}_{args.timeframe}.csv"
    out_path = DATA_DIR / filename
    save_csv(rows, out_path)

    first = rows[0]["timestamp"][:10]
    last = rows[-1]["timestamp"][:10]
    print(f"\n  {len(rows):,} candles  |  {first} → {last}  |  saved → data/historical/{filename}\n")


if __name__ == "__main__":
    main()
