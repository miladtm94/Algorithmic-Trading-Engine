#!/usr/bin/env python
"""Train a sequence-aware deep MLP on the generated signal dataset.

This trainer turns each dataset row into a rolling sequence of prior feature
vectors from the same side (LONG or SHORT), flattens that window, standardizes
it, then fits a multi-layer perceptron.

By default it uses a 70/30 random split by timestamp group, as requested.
That produces a more optimistic estimate than a strict time-ordered split, so
`--split-mode temporal` is also available.

Outputs:
  data/models/<ASSET>_<TIMEFRAME>_dnn.pkl
  data/models/<ASSET>_<TIMEFRAME>_dnn.json
  data/models/<ASSET>_<TIMEFRAME>_dnn_test_predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

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
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_float_list(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def parse_int_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(value.strip()) for value in raw.split(",") if value.strip())


def build_sequence_samples(
    rows: list[dict],
    feature_names: list[str],
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    histories: dict[str, list[list[float]]] = defaultdict(list)
    X: list[list[float]] = []
    y: list[int] = []
    sample_rows: list[dict] = []

    for row in rows:
        side = str(row.get("side", row.get("direction", "UNKNOWN")))
        history = histories[side]
        current = [float(row.get(name, 0.0)) for name in feature_names]
        history.append(current)
        if len(history) < sequence_length:
            continue
        if len(history) > sequence_length:
            history.pop(0)

        flattened = [value for window in history for value in window]
        X.append(flattened)
        y.append(1 if row["outcome"] == "WIN" else 0)
        sample_rows.append(row)

    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), sample_rows


def split_indices(
    sample_rows: list[dict],
    y: np.ndarray,
    *,
    test_pct: float,
    split_mode: str,
    random_state: int,
) -> tuple[list[int], list[int]]:
    if split_mode == "temporal":
        split = int(len(sample_rows) * (1 - test_pct))
        return list(range(split)), list(range(split, len(sample_rows)))

    timestamp_to_indices: dict[str, list[int]] = defaultdict(list)
    timestamp_to_labels: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(sample_rows):
        timestamp = row["timestamp"]
        timestamp_to_indices[timestamp].append(idx)
        timestamp_to_labels[timestamp].append(int(y[idx]))

    groups = list(timestamp_to_indices)
    rng = random.Random(random_state)
    rng.shuffle(groups)

    label_buckets: dict[int, list[str]] = defaultdict(list)
    for timestamp, labels in timestamp_to_labels.items():
        majority = 1 if sum(labels) >= len(labels) / 2 else 0
        label_buckets[majority].append(timestamp)

    test_groups: set[str] = set()
    for timestamps in label_buckets.values():
        rng.shuffle(timestamps)
        n_test = max(1, int(round(len(timestamps) * test_pct)))
        test_groups.update(timestamps[:n_test])

    train_idx: list[int] = []
    test_idx: list[int] = []
    for timestamp, indices in timestamp_to_indices.items():
        target = test_idx if timestamp in test_groups else train_idx
        target.extend(indices)

    train_idx.sort()
    test_idx.sort()
    return train_idx, test_idx


def oversample_minority(X: np.ndarray, y: np.ndarray, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    counts = Counter(y.tolist())
    if len(counts) < 2 or counts[0] == counts[1]:
        return X, y

    majority_class = counts.most_common(1)[0][0]
    minority_class = 1 - majority_class
    deficit = counts[majority_class] - counts[minority_class]
    minority_indices = np.where(y == minority_class)[0]
    rng = np.random.default_rng(random_state)
    sampled = rng.choice(minority_indices, size=deficit, replace=True)

    X_balanced = np.concatenate([X, X[sampled]], axis=0)
    y_balanced = np.concatenate([y, y[sampled]], axis=0)
    return X_balanced, y_balanced


def classification_report_text(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    from sklearn.metrics import classification_report  # noqa: PLC0415

    return classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["LOSS", "WIN"],
        zero_division=0,
    )


def threshold_stats(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    rows: list[dict],
    thresholds: list[float],
) -> list[dict]:
    stats: list[dict] = []
    for threshold in thresholds:
        selected = [
            (prob, label, row)
            for prob, label, row in zip(probabilities, y_true, rows, strict=True)
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
        wins = sum(1 for _, label, _ in selected if label == 1)
        avg_pnl = sum(float(row.get("pnl_pct", 0.0)) for _, _, row in selected) / len(selected)
        stats.append(
            {
                "threshold": threshold,
                "count": len(selected),
                "precision": round(wins / len(selected), 4),
                "avg_pnl_pct": round(avg_pnl, 6),
            }
        )
    return stats


def signal_level_stats(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    rows: list[dict],
    threshold: float,
) -> dict:
    by_timestamp: dict[str, list[tuple[float, int, dict]]] = defaultdict(list)
    for prob, label, row in zip(probabilities, y_true, rows, strict=True):
        by_timestamp[row["timestamp"]].append((float(prob), int(label), row))

    selected: list[tuple[float, int, dict]] = []
    for entries in by_timestamp.values():
        best = max(entries, key=lambda item: item[0])
        if best[0] >= threshold:
            selected.append(best)

    if not selected:
        return {"count": 0, "win_rate": 0.0, "avg_pnl_pct": 0.0}

    wins = sum(1 for _, label, _ in selected if label == 1)
    avg_pnl = sum(float(row.get("pnl_pct", 0.0)) for _, _, row in selected) / len(selected)
    return {
        "count": len(selected),
        "win_rate": round(wins / len(selected), 4),
        "avg_pnl_pct": round(avg_pnl, 6),
    }


def save_predictions(
    path: Path,
    rows: list[dict],
    probabilities: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
) -> None:
    fieldnames = [
        "timestamp",
        "side",
        "actual",
        "predicted",
        "win_probability",
        "selected_at_threshold",
        "entry",
        "stop_loss",
        "take_profit",
        "exit_price",
        "exit_reason",
        "bars_held",
        "pnl_pct",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, prob, pred in zip(rows, probabilities, predictions, strict=True):
            writer.writerow(
                {
                    "timestamp": row["timestamp"],
                    "side": row.get("side", row.get("direction", "UNKNOWN")),
                    "actual": row["outcome"],
                    "predicted": "WIN" if int(pred) == 1 else "LOSS",
                    "win_probability": round(float(prob), 6),
                    "selected_at_threshold": int(float(prob) >= threshold),
                    "entry": row.get("entry", ""),
                    "stop_loss": row.get("stop_loss", ""),
                    "take_profit": row.get("take_profit", ""),
                    "exit_price": row.get("exit_price", ""),
                    "exit_reason": row.get("exit_reason", ""),
                    "bars_held": row.get("bars_held", ""),
                    "pnl_pct": row.get("pnl_pct", ""),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sequence-aware deep neural model")
    parser.add_argument("--asset", default=os.getenv("DEFAULT_ASSET", "ETH/USDT"))
    parser.add_argument("--timeframe", default=os.getenv("DEFAULT_TIMEFRAME", "1h"))
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--test-pct", type=float, default=0.30)
    parser.add_argument(
        "--split-mode",
        choices=["random", "temporal"],
        default="random",
        help="random = 70/30 random timestamp split, temporal = holdout most recent rows",
    )
    parser.add_argument("--hidden-layers", default="256,128,64")
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--thresholds",
        default="0.55,0.60,0.65,0.70",
        help="Probability thresholds for signal quality summaries",
    )
    parser.add_argument("--selection-threshold", type=float, default=0.65)
    parser.add_argument("--balance", choices=["oversample", "none"], default="oversample")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    try:
        import joblib  # noqa: PLC0415
        from sklearn.neural_network import MLPClassifier  # noqa: PLC0415
        from sklearn.preprocessing import StandardScaler  # noqa: PLC0415
    except ImportError:
        print("ERROR: ML dependencies missing. Install with: pip install -e '.[ml]'")
        sys.exit(1)

    from ai_trading_engine.feature_extractor import FEATURE_NAMES  # noqa: PLC0415

    hidden_layers = parse_int_tuple(args.hidden_layers)
    thresholds = parse_float_list(args.thresholds)

    safe_asset = args.asset.replace("/", "_")
    dataset_path = FEATURES_DIR / f"{safe_asset}_{args.timeframe}_dataset.csv"
    if not dataset_path.exists():
        print(f"ERROR: {dataset_path} not found.")
        print(f"  Run first: python scripts/build_dataset.py --asset {args.asset} --timeframe {args.timeframe}")
        sys.exit(1)

    rows = load_dataset(dataset_path)
    X, y, sample_rows = build_sequence_samples(rows, FEATURE_NAMES, args.sequence_length)
    if len(sample_rows) < 100:
        print(f"ERROR: Only {len(sample_rows)} sequence samples. Reduce --sequence-length or build more history.")
        sys.exit(1)

    train_idx, test_idx = split_indices(
        sample_rows,
        y,
        test_pct=args.test_pct,
        split_mode=args.split_mode,
        random_state=args.random_state,
    )
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    train_rows = [sample_rows[idx] for idx in train_idx]
    test_rows = [sample_rows[idx] for idx in test_idx]

    if args.balance == "oversample":
        X_train, y_train = oversample_minority(X_train, y_train, args.random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=args.alpha,
        batch_size=args.batch_size,
        learning_rate_init=args.learning_rate,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )
    model.fit(X_train_scaled, y_train)

    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    train_acc = float((train_pred == y_train).mean())
    test_acc = float((test_pred == y_test).mean())
    baseline = max(int((y_test == 0).sum()), int((y_test == 1).sum())) / len(y_test)

    test_prob = model.predict_proba(X_test_scaled)[:, 1]
    threshold_report = threshold_stats(test_prob, y_test, test_rows, thresholds)
    signal_report = signal_level_stats(test_prob, y_test, test_rows, args.selection_threshold)

    print(f"\n  Asset:          {args.asset} [{args.timeframe}]")
    print(f"  Model:          sequence MLP {hidden_layers}")
    print(f"  Sequence len:   {args.sequence_length}")
    print(f"  Split:          {args.split_mode} ({int((1 - args.test_pct) * 100)}/{int(args.test_pct * 100)})")
    if args.split_mode == "random":
        print("  Note:           random timestamp split is more optimistic than true walk-forward testing")
    print(f"  Train samples:  {len(train_rows)}")
    print(f"  Test samples:   {len(test_rows)}")
    print(f"  Train WIN rate: {(y_train == 1).mean():.1%}")
    print(f"  Test WIN rate:  {(y_test == 1).mean():.1%}")
    print()
    print(f"  Train accuracy: {train_acc:.1%}")
    print(f"  Test accuracy:  {test_acc:.1%}")
    print(f"  Baseline:       {baseline:.1%}")
    print(f"  Lift:           {test_acc - baseline:+.1%}")
    print()
    print(classification_report_text(y_test, test_pred))

    print("  Threshold WIN filters:")
    for stat in threshold_report:
        print(
            f"    p(WIN) >= {stat['threshold']:.0%}: "
            f"{stat['count']:4d} rows, precision {stat['precision']:.1%}, "
            f"avg realised PnL {stat['avg_pnl_pct']:+.2%}"
        )
    print()
    print(
        f"  Best-side signal view @ {args.selection_threshold:.0%}: "
        f"{signal_report['count']} signals, "
        f"win rate {signal_report['win_rate']:.1%}, "
        f"avg realised PnL {signal_report['avg_pnl_pct']:+.2%}"
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    split_suffix = f"_dnn_{args.split_mode}"
    model_path = MODELS_DIR / f"{safe_asset}_{args.timeframe}{split_suffix}.pkl"
    meta_path = MODELS_DIR / f"{safe_asset}_{args.timeframe}{split_suffix}.json"
    preds_path = MODELS_DIR / f"{safe_asset}_{args.timeframe}{split_suffix}_test_predictions.csv"
    alias_model_path = MODELS_DIR / f"{safe_asset}_{args.timeframe}_dnn.pkl"
    alias_meta_path = MODELS_DIR / f"{safe_asset}_{args.timeframe}_dnn.json"
    alias_preds_path = MODELS_DIR / f"{safe_asset}_{args.timeframe}_dnn_test_predictions.csv"

    bundle = {
        "scaler": scaler,
        "model": model,
        "feature_names": FEATURE_NAMES,
        "sequence_length": args.sequence_length,
        "hidden_layers": hidden_layers,
    }
    joblib.dump(bundle, model_path)

    meta = {
        "asset": args.asset,
        "timeframe": args.timeframe,
        "model_type": "sequence_mlp",
        "sequence_length": args.sequence_length,
        "hidden_layers": list(hidden_layers),
        "split_mode": args.split_mode,
        "test_pct": args.test_pct,
        "random_state": args.random_state,
        "balance": args.balance,
        "train_samples": len(train_rows),
        "test_samples": len(test_rows),
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "baseline_accuracy": round(baseline, 4),
        "lift": round(test_acc - baseline, 4),
        "selection_threshold": args.selection_threshold,
        "threshold_stats": threshold_report,
        "best_side_signal_stats": signal_report,
        "predictions_csv": str(preds_path),
        "alias_written": args.split_mode == "random",
        "feature_names": FEATURE_NAMES,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    save_predictions(preds_path, test_rows, test_prob, test_pred, args.selection_threshold)

    if args.split_mode == "random":
        joblib.dump(bundle, alias_model_path)
        alias_meta_path.write_text(json.dumps({**meta, "predictions_csv": str(alias_preds_path)}, indent=2))
        save_predictions(alias_preds_path, test_rows, test_prob, test_pred, args.selection_threshold)

    print(f"\n  Model → {model_path}")
    print(f"  Meta  → {meta_path}")
    print(f"  Test  → {preds_path}\n")


if __name__ == "__main__":
    main()
