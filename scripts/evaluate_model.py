#!/usr/bin/env python
"""Evaluate ML signal accuracy at any historical date or right now.

Two modes:

  --as-of YYYY-MM-DD
      Trains the model on all data BEFORE that date and evaluates it on
      all data AFTER it. This tells you "if I had trained the model at
      this point in time, how would it have performed on unseen data?"
      In-sample accuracy will be high; out-of-sample is the real metric.

  --now
      Fetches the current market snapshot from the exchange, asks the
      engine for a signal, and scores it with the pre-trained model.

Usage:
  python scripts/evaluate_model.py --asset ETH/USD --as-of 2024-01-01
  python scripts/evaluate_model.py --asset ETH/USD --as-of 2023-06-01
  python scripts/evaluate_model.py --asset ETH/USD --now
  python scripts/evaluate_model.py --asset ETH/USD --now --exchange kraken
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.WARNING)

FEATURES_DIR = Path(__file__).parent.parent / "data" / "features"
MODELS_DIR = Path(__file__).parent.parent / "data" / "models"

_SEP = "─" * 54


def _load_dataset(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_xy(rows: list[dict], feature_names: list[str]):
    X = [[float(r.get(k, 0.0)) for k in feature_names] for r in rows]
    y = [r["outcome"] for r in rows]
    return X, y


# ── as-of mode ────────────────────────────────────────────────────────────────

def evaluate_as_of(asset: str, timeframe: str, as_of: str) -> None:
    try:
        import joblib  # noqa: PLC0415
        from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415
        from sklearn.metrics import classification_report  # noqa: PLC0415
    except ImportError:
        print("ERROR: pip install -e '.[ml]'")
        sys.exit(1)

    from ai_trading_engine.feature_extractor import FEATURE_NAMES  # noqa: PLC0415

    safe_asset = asset.replace("/", "_")
    dataset_path = FEATURES_DIR / f"{safe_asset}_{timeframe}_dataset.csv"
    if not dataset_path.exists():
        print(f"ERROR: {dataset_path} not found. Run build_dataset.py first.")
        sys.exit(1)

    rows = _load_dataset(dataset_path)
    # Rows are time-ordered; ISO timestamp string comparison works correctly
    cutoff = as_of
    train_rows = [r for r in rows if r["timestamp"] < cutoff]
    test_rows = [r for r in rows if r["timestamp"] >= cutoff]

    print(f"\n{_SEP}")
    print(f"  AS-OF EVALUATION   {asset}  [{timeframe}]")
    print(_SEP)
    print(f"  Cutoff:      {as_of}")
    print(f"  Train:       {len(train_rows)} signals", end="")
    if train_rows:
        print(f"  ({train_rows[0]['timestamp'][:10]} → {train_rows[-1]['timestamp'][:10]})", end="")
    print()
    print(f"  Test:        {len(test_rows)} signals", end="")
    if test_rows:
        print(f"  ({test_rows[0]['timestamp'][:10]} → {test_rows[-1]['timestamp'][:10]})", end="")
    print()

    if len(train_rows) < 20:
        print(f"\n  ERROR: Only {len(train_rows)} training samples before {as_of}.")
        print("  Try an earlier --as-of date or fetch more history.")
        sys.exit(1)
    if len(test_rows) < 5:
        print(f"\n  ERROR: Only {len(test_rows)} test samples after {as_of}.")
        print("  Try a later --as-of date.")
        sys.exit(1)

    X_train, y_train = _to_xy(train_rows, FEATURE_NAMES)
    X_test, y_test = _to_xy(test_rows, FEATURE_NAMES)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    baseline = max(y_test.count("WIN"), y_test.count("LOSS")) / len(y_test)

    print()
    print(f"  Train accuracy:   {train_acc:.1%}  (in-sample — expected to be inflated)")
    print(f"  Test accuracy:    {test_acc:.1%}  (out-of-sample — the real signal quality)")
    print(f"  Baseline guess:   {baseline:.1%}  (majority-class)")
    lift = test_acc - baseline
    lift_str = f"{lift:+.1%}"
    print(f"  Lift over base:   {lift_str}  {'✓ model is learning' if lift > 0.03 else '~ marginal' if lift > 0 else '✗ no lift'}")
    print(_SEP)
    print()
    print(classification_report(y_test, model.predict(X_test), target_names=["LOSS", "WIN"]))


# ── now mode ──────────────────────────────────────────────────────────────────

def evaluate_now(asset: str, timeframe: str, exchange_id: str) -> None:
    try:
        import joblib  # noqa: PLC0415
    except ImportError:
        print("ERROR: pip install -e '.[ml]'")
        sys.exit(1)

    from ai_trading_engine.config import EngineConfig  # noqa: PLC0415
    from ai_trading_engine.engine import HybridTradingEngine  # noqa: PLC0415
    from ai_trading_engine.feature_extractor import extract_features  # noqa: PLC0415
    from ai_trading_engine.market_data import MarketDataFetcher  # noqa: PLC0415
    from ai_trading_engine.models import PortfolioState  # noqa: PLC0415
    from ai_trading_engine.feature_extractor import FEATURE_NAMES  # noqa: PLC0415

    safe_asset = asset.replace("/", "_")
    model_path = MODELS_DIR / f"{safe_asset}_{timeframe}_rf.pkl"
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}.")
        print(f"  Run first: python scripts/train_model.py --asset {asset}")
        sys.exit(1)

    api_key = os.getenv("EXCHANGE_API_KEY", "")
    api_secret = os.getenv("EXCHANGE_API_SECRET", "")

    print(f"Fetching live snapshot: {asset} [{timeframe}] from {exchange_id} ...")
    fetcher = MarketDataFetcher(exchange_id, api_key, api_secret)
    snapshot = fetcher.fetch_snapshot(asset, timeframe, candle_limit=250)

    engine = HybridTradingEngine(EngineConfig())
    portfolio = PortfolioState(equity_usd=10_000.0)
    decision = engine.evaluate(snapshot, portfolio)

    model = joblib.load(model_path)
    classes = list(model.classes_)
    win_idx = classes.index("WIN") if "WIN" in classes else 1

    current_price = snapshot.candles[-1].close

    print(f"\n{_SEP}")
    print(f"  LIVE SIGNAL EVALUATION   {asset}  [{timeframe}]")
    print(_SEP)
    print(f"  Exchange:    {exchange_id}")
    print(f"  Price:       {current_price:.4f}")

    if not decision.is_trade or decision.signal is None:
        print(f"  Signal:      NO TRADE")
        print(f"  Reason:      {decision.no_trade_reason}")
        print(_SEP)
        return

    sig = decision.signal
    features = extract_features(sig)
    row = [[features.get(k, 0.0) for k in FEATURE_NAMES]]
    proba = model.predict_proba(row)[0]
    win_prob = float(proba[win_idx])

    rating = "STRONG" if win_prob >= 0.65 else "MODERATE" if win_prob >= 0.50 else "WEAK"
    rating_note = {
        "STRONG": "model confident — signal meets historical win pattern",
        "MODERATE": "model sees some signal — proceed with normal position sizing",
        "WEAK": "model uncertain — consider skipping or reducing size",
    }[rating]

    print(f"  Signal:      {sig.candidate.direction}")
    print(f"  Entry:       {sig.candidate.entry:.4f}")
    print(f"  Stop Loss:   {sig.candidate.stop_loss:.4f}")
    if sig.candidate.take_profits:
        for i, tp in enumerate(sig.candidate.take_profits[:3], 1):
            print(f"  TP{i}:         {tp:.4f}")
    print(f"  Regime:      {sig.candidate.regime.regime}")
    print(f"  Confluence:  {sig.confluence.total_score:.1f}%")
    print(f"  Risk/Reward: {sig.candidate.risk_reward:.2f}x")
    print()
    print(f"  ★ ML Win Probability:  {win_prob:.1%}")
    print(f"  Signal Quality:        {rating}")
    print(f"  Note:                  {rating_note}")
    print(_SEP)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ML signal accuracy at a historical date or right now"
    )
    parser.add_argument("--asset", default=os.getenv("DEFAULT_ASSET", "ETH/USD"))
    parser.add_argument("--timeframe", default=os.getenv("DEFAULT_TIMEFRAME", "1h"))
    parser.add_argument("--exchange", default=os.getenv("EXCHANGE", "kraken"))

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--as-of", metavar="YYYY-MM-DD",
        help="Train on data before this date, evaluate on data after it"
    )
    group.add_argument(
        "--now", action="store_true",
        help="Fetch current market data and score the live signal"
    )
    args = parser.parse_args()

    if args.now:
        evaluate_now(args.asset, args.timeframe, args.exchange)
    else:
        evaluate_as_of(args.asset, args.timeframe, args.as_of)


if __name__ == "__main__":
    main()
