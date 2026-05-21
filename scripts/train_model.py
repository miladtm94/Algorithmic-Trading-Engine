#!/usr/bin/env python
"""Train a RandomForest classifier on the backtested signal dataset.

Uses a strict time-ordered split — the most recent `--test-pct` fraction
of signals is held out for validation so there is zero look-ahead bias.

Output:
  data/models/<ASSET>_<TIMEFRAME>_rf.pkl   — serialised sklearn model
  data/models/<ASSET>_<TIMEFRAME>_rf.json  — metadata + top features

Usage:
  python scripts/train_model.py --asset ETH/USDT --timeframe 1h
  python scripts/train_model.py --asset ETH/USDT --timeframe 1h --test-pct 0.25
"""
from __future__ import annotations

import argparse
import csv
import json
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


def load_dataset(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _threshold_stats(model, X_test: list[list[float]], test_rows: list[dict]) -> list[dict]:
    """Summarise realised quality when only high-probability WIN rows are taken."""
    if not hasattr(model, "predict_proba") or "WIN" not in list(model.classes_):
        return []

    win_idx = list(model.classes_).index("WIN")
    probabilities = [float(row[win_idx]) for row in model.predict_proba(X_test)]
    stats: list[dict] = []
    for threshold in (0.55, 0.60, 0.65, 0.70):
        selected = [
            (prob, row)
            for prob, row in zip(probabilities, test_rows, strict=True)
            if prob >= threshold
        ]
        if not selected:
            stats.append(
                {
                    "threshold": threshold,
                    "count": 0,
                    "precision": 0.0,
                    "avg_pnl_pct": 0.0,
                }
            )
            continue
        wins = sum(1 for _, row in selected if row["outcome"] == "WIN")
        pnl_values = [float(row.get("pnl_pct", 0.0)) for _, row in selected]
        stats.append(
            {
                "threshold": threshold,
                "count": len(selected),
                "precision": round(wins / len(selected), 4),
                "avg_pnl_pct": round(sum(pnl_values) / len(pnl_values), 6),
            }
        )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML model on backtested signal dataset")
    parser.add_argument("--asset", default=os.getenv("DEFAULT_ASSET", "ETH/USDT"))
    parser.add_argument("--timeframe", default=os.getenv("DEFAULT_TIMEFRAME", "1h"))
    parser.add_argument(
        "--test-pct", type=float, default=0.20,
        help="Fraction of data (most recent) held out for testing. Default: 0.20"
    )
    parser.add_argument("--trees", type=int, default=200, help="Number of trees in the forest")
    args = parser.parse_args()

    try:
        import joblib  # noqa: PLC0415
        from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415
        from sklearn.metrics import classification_report  # noqa: PLC0415
    except ImportError:
        print("ERROR: ML dependencies missing. Install with: pip install -e '.[ml]'")
        sys.exit(1)

    from ai_trading_engine.feature_extractor import FEATURE_NAMES  # noqa: PLC0415

    safe_asset = args.asset.replace("/", "_")
    dataset_path = FEATURES_DIR / f"{safe_asset}_{args.timeframe}_dataset.csv"
    if not dataset_path.exists():
        print(f"ERROR: {dataset_path} not found.")
        print(f"  Run first: python scripts/build_dataset.py --asset {args.asset}")
        sys.exit(1)

    rows = load_dataset(dataset_path)
    n = len(rows)
    if n < 30:
        print(f"ERROR: Only {n} samples. Need at least 30. Try lowering --confluence in build_dataset.py.")
        sys.exit(1)
    classes = sorted({row["outcome"] for row in rows})
    if len(classes) < 2:
        print(f"ERROR: Dataset only contains one class: {classes[0]}. Need both WIN and LOSS rows.")
        sys.exit(1)

    # Time-ordered split — no shuffle, preserves temporal structure
    split = int(n * (1 - args.test_pct))
    train_rows, test_rows = rows[:split], rows[split:]

    def to_xy(data: list[dict]):
        X = [[float(r.get(k, 0.0)) for k in FEATURE_NAMES] for r in data]
        y = [r["outcome"] for r in data]
        return X, y

    X_train, y_train = to_xy(train_rows)
    X_test, y_test = to_xy(test_rows)

    train_period = f"{train_rows[0]['timestamp'][:10]} → {train_rows[-1]['timestamp'][:10]}"
    test_period = f"{test_rows[0]['timestamp'][:10]} → {test_rows[-1]['timestamp'][:10]}"
    train_win_rate = sum(1 for y in y_train if y == "WIN") / len(y_train)
    test_win_rate = sum(1 for y in y_test if y == "WIN") / len(y_test)
    has_side = "side" in rows[0]

    print(f"\n  Asset:         {args.asset}  [{args.timeframe}]")
    print(f"  Train set:     {len(X_train)} samples  ({train_period})  win rate {train_win_rate:.1%}")
    print(f"  Test set:      {len(X_test)} samples   ({test_period})  win rate {test_win_rate:.1%}")
    if has_side:
        train_long = sum(1 for row in train_rows if row.get("side") == "LONG")
        test_long = sum(1 for row in test_rows if row.get("side") == "LONG")
        print(
            f"  Sides:         train {train_long} LONG / {len(train_rows) - train_long} SHORT, "
            f"test {test_long} LONG / {len(test_rows) - test_long} SHORT"
        )
    print(f"  Features:      {len(FEATURE_NAMES)}")
    print(f"  Trees:         {args.trees}")
    print()

    model = RandomForestClassifier(
        n_estimators=args.trees,
        max_depth=6,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    baseline = max(y_test.count("WIN"), y_test.count("LOSS")) / len(y_test)
    predictions = model.predict(X_test)

    print(f"  Train accuracy:  {train_acc:.1%}  (in-sample)")
    print(f"  Test accuracy:   {test_acc:.1%}  (out-of-sample — this is the real number)")
    print(f"  Baseline:        {baseline:.1%}  (majority-class guess)")
    print(f"  Lift over base:  {test_acc - baseline:+.1%}")
    print()
    print(
        classification_report(
            y_test,
            predictions,
            labels=["LOSS", "WIN"],
            target_names=["LOSS", "WIN"],
            zero_division=0,
        )
    )

    threshold_stats = _threshold_stats(model, X_test, test_rows)
    if threshold_stats:
        print("  High-confidence WIN filters:")
        for stat in threshold_stats:
            print(
                f"    p(WIN) >= {stat['threshold']:.0%}: "
                f"{stat['count']:4d} rows, precision {stat['precision']:.1%}, "
                f"avg realised PnL {stat['avg_pnl_pct']:+.2%}"
            )
        print()

    # Feature importance ranking
    importances = sorted(
        zip(FEATURE_NAMES, model.feature_importances_, strict=True), key=lambda x: -x[1]
    )
    print("  Top 10 most predictive features:")
    for name, imp in importances[:10]:
        bar = "█" * max(1, int(imp * 300))
        print(f"    {name:30s}  {imp:.4f}  {bar}")

    # Save model and metadata
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{safe_asset}_{args.timeframe}_rf.pkl"
    meta_path = MODELS_DIR / f"{safe_asset}_{args.timeframe}_rf.json"

    joblib.dump(model, model_path)
    meta = {
        "asset": args.asset,
        "timeframe": args.timeframe,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "train_period": train_period,
        "test_period": test_period,
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "baseline_accuracy": round(baseline, 4),
        "lift": round(test_acc - baseline, 4),
        "feature_names": FEATURE_NAMES,
        "threshold_stats": threshold_stats,
        "top_features": [
            {"name": name, "importance": round(imp, 4)} for name, imp in importances[:10]
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\n  Model → {model_path}")
    print(f"  Meta  → {meta_path}\n")


if __name__ == "__main__":
    main()
