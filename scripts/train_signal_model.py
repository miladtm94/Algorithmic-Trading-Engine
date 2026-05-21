#!/usr/bin/env python
"""Train a sparse signal selector model on engine-generated candidates."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

FEATURES_DIR = Path(__file__).parent.parent / "data" / "features"
MODELS_DIR = Path(__file__).parent.parent / "data" / "models"


def load_dataset(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_xy(
    rows: list[dict[str, str]], feature_names: list[str]
) -> tuple[list[list[float]], list[str]]:
    X = [[float(row.get(name, 0.0)) for name in feature_names] for row in rows]
    y = [row["outcome"] for row in rows]
    return X, y


def period(rows: list[dict[str, str]]) -> str:
    return f"{rows[0]['timestamp'][:10]} → {rows[-1]['timestamp'][:10]}"


def split_rows_random(
    rows: list[dict[str, str]],
    *,
    validation_pct: float,
    test_pct: float,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["timestamp"], []).append(row)

    timestamps = list(grouped)
    rng = random.Random(42)
    rng.shuffle(timestamps)

    n_groups = len(timestamps)
    n_test = max(1, int(n_groups * test_pct))
    n_valid = max(1, int(n_groups * validation_pct))
    if n_test + n_valid >= n_groups:
        raise ValueError("Random split produced no training groups.")

    test_keys = set(timestamps[:n_test])
    valid_keys = set(timestamps[n_test : n_test + n_valid])
    train_keys = set(timestamps[n_test + n_valid :])

    train_rows = [row for row in rows if row["timestamp"] in train_keys]
    valid_rows = [row for row in rows if row["timestamp"] in valid_keys]
    test_rows = [row for row in rows if row["timestamp"] in test_keys]
    train_rows.sort(key=lambda row: row["timestamp"])
    valid_rows.sort(key=lambda row: row["timestamp"])
    test_rows.sort(key=lambda row: row["timestamp"])
    return train_rows, valid_rows, test_rows


def fit_probability_calibrator(
    probabilities: list[float],
    labels: list[str],
    *,
    method: str,
):
    binary = [1 if label == "WIN" else 0 for label in labels]
    if method == "none" or len(set(binary)) < 2:
        return None

    if method == "platt":
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

        calibrator = LogisticRegression(max_iter=1000, random_state=42)
        calibrator.fit([[probability] for probability in probabilities], binary)
        return calibrator

    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415

        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(probabilities, binary)
        return calibrator

    raise ValueError(f"Unsupported calibration method: {method}")


def apply_probability_calibrator(
    probabilities: list[float],
    *,
    calibrator,
    method: str,
) -> list[float]:
    if calibrator is None or method == "none":
        return probabilities
    if method == "platt":
        calibrated = calibrator.predict_proba([[probability] for probability in probabilities])
        return [float(row[1]) for row in calibrated]
    if method == "isotonic":
        return [float(value) for value in calibrator.transform(probabilities)]
    raise ValueError(f"Unsupported calibration method: {method}")


def labels_from_probabilities(probabilities: list[float]) -> list[str]:
    return ["WIN" if probability >= 0.5 else "LOSS" for probability in probabilities]


def probability_range(probabilities: list[float]) -> dict[str, float]:
    if not probabilities:
        return {"min": 0.0, "max": 0.0, "spread": 0.0}
    low = min(probabilities)
    high = max(probabilities)
    return {
        "min": round(low, 6),
        "max": round(high, 6),
        "spread": round(high - low, 6),
    }


def _float_value(value: str | None) -> float:
    try:
        return float(value) if value not in (None, "") else 0.0
    except ValueError:
        return 0.0


def family_stats(rows: list[dict[str, str]]) -> list[dict[str, float | int | str]]:
    grouped: dict[str, dict[str, float | int | str]] = defaultdict(
        lambda: {
            "family": "",
            "count": 0,
            "wins": 0,
            "losses": 0,
            "pnl_sum": 0.0,
            "net_r_sum": 0.0,
            "quality_sum": 0.0,
        }
    )
    for row in rows:
        family = row.get("setup_family", "UNKNOWN") or "UNKNOWN"
        stats = grouped[family]
        stats["family"] = family
        stats["count"] = int(stats["count"]) + 1
        if row.get("outcome") == "WIN":
            stats["wins"] = int(stats["wins"]) + 1
        else:
            stats["losses"] = int(stats["losses"]) + 1
        stats["pnl_sum"] = float(stats["pnl_sum"]) + _float_value(row.get("pnl_pct"))
        stats["net_r_sum"] = float(stats["net_r_sum"]) + _float_value(row.get("net_r"))
        stats["quality_sum"] = float(stats["quality_sum"]) + _float_value(row.get("setup_quality"))

    summary: list[dict[str, float | int | str]] = []
    for family, stats in grouped.items():
        count = int(stats["count"])
        wins = int(stats["wins"])
        summary.append(
            {
                "family": family,
                "count": count,
                "wins": wins,
                "losses": int(stats["losses"]),
                "win_rate": wins / count if count else 0.0,
                "avg_pnl_pct": float(stats["pnl_sum"]) / count if count else 0.0,
                "avg_net_r": float(stats["net_r_sum"]) / count if count else 0.0,
                "avg_setup_quality": float(stats["quality_sum"]) / count if count else 0.0,
            }
        )
    return sorted(summary, key=lambda item: int(item["count"]), reverse=True)


def print_family_stats(label: str, rows: list[dict[str, str]]) -> None:
    print(f"  {label}:")
    for stats in family_stats(rows):
        print(
            f"    {stats['family']}: "
            f"{int(stats['count']):4d} rows, "
            f"{float(stats['win_rate']):.1%} win rate, "
            f"{float(stats['avg_pnl_pct']):+.2%} avg PnL, "
            f"{float(stats['avg_net_r']):+.2f} avg net R, "
            f"{float(stats['avg_setup_quality']):.1f} avg quality"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sparse signal selector model")
    parser.add_argument("--asset", default=os.getenv("DEFAULT_ASSET", "ETH/USDT"))
    parser.add_argument("--timeframe", default=os.getenv("DEFAULT_TIMEFRAME", "1h"))
    parser.add_argument(
        "--test-pct",
        type=float,
        default=0.20,
        help="Final holdout fraction. Default 0.20.",
    )
    parser.add_argument(
        "--validation-pct",
        type=float,
        default=0.15,
        help="Validation fraction used for threshold selection and calibration.",
    )
    parser.add_argument(
        "--split-mode",
        choices=["temporal", "random"],
        default="temporal",
        help="Temporal is the honest default. Random is available for diagnostics.",
    )
    parser.add_argument(
        "--purge-rows",
        type=int,
        default=24,
        help="Rows purged before validation and test to reduce overlap leakage.",
    )
    parser.add_argument("--weekly-cap", type=int, default=10)
    parser.add_argument("--thresholds", default="0.55,0.60,0.65,0.70,0.75")
    parser.add_argument("--min-threshold-count", type=int, default=12)
    parser.add_argument(
        "--calibration",
        choices=["none", "platt", "isotonic"],
        default="platt",
        help="Fit the probability calibrator on the validation slice.",
    )
    args = parser.parse_args()

    try:
        import joblib  # noqa: PLC0415
        from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415
        from sklearn.inspection import permutation_importance  # noqa: PLC0415
        from sklearn.metrics import classification_report  # noqa: PLC0415
        from sklearn.preprocessing import LabelEncoder  # noqa: PLC0415
    except ImportError:
        print("ERROR: ML dependencies missing. Install with: pip install -e '.[ml]'")
        sys.exit(1)

    from ai_trading_engine.signal_learning import SIGNAL_FEATURE_NAMES  # noqa: PLC0415
    from ai_trading_engine.validation import (  # noqa: PLC0415
        brier_score,
        calibration_bins,
        choose_best_threshold,
        selected_flags,
        temporal_train_validation_test_split,
        threshold_metrics,
    )

    safe_asset = args.asset.replace("/", "_")
    dataset_path = FEATURES_DIR / f"{safe_asset}_{args.timeframe}_signal_dataset.csv"
    if not dataset_path.exists():
        print(f"ERROR: {dataset_path} not found.")
        print(f"  Run first: python scripts/build_signal_dataset.py --asset {args.asset}")
        sys.exit(1)

    rows = load_dataset(dataset_path)
    if len(rows) < 40:
        print(f"ERROR: Only {len(rows)} candidate rows found. Need at least 40.")
        sys.exit(1)

    classes = sorted({row["outcome"] for row in rows})
    if len(classes) < 2:
        print(f"ERROR: Dataset only contains one class: {classes[0]}")
        sys.exit(1)

    if args.split_mode == "temporal":
        try:
            split = temporal_train_validation_test_split(
                rows,
                validation_pct=args.validation_pct,
                test_pct=args.test_pct,
                purge_rows=args.purge_rows,
            )
        except ValueError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
        train_rows = split.train_rows
        valid_rows = split.validation_rows
        test_rows = split.test_rows
    else:
        try:
            train_rows, valid_rows, test_rows = split_rows_random(
                rows,
                validation_pct=args.validation_pct,
                test_pct=args.test_pct,
            )
        except ValueError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)

    if len(train_rows) < 20 or len(valid_rows) < 10 or len(test_rows) < 10:
        print("ERROR: Split produced too little train, validation, or test data.")
        sys.exit(1)

    X_train, y_train = to_xy(train_rows, SIGNAL_FEATURE_NAMES)
    X_valid, y_valid = to_xy(valid_rows, SIGNAL_FEATURE_NAMES)
    X_test, y_test = to_xy(test_rows, SIGNAL_FEATURE_NAMES)

    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_valid_enc = encoder.transform(y_valid)
    y_test_enc = encoder.transform(y_test)
    class_counts = Counter(y_train_enc)
    sample_weights = [
        len(y_train_enc) / (len(class_counts) * class_counts[y_value]) for y_value in y_train_enc
    ]

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=4,
        max_iter=350,
        min_samples_leaf=12,
        l2_regularization=0.2,
        random_state=42,
    )
    model.fit(X_train, y_train_enc, sample_weight=sample_weights)

    train_acc = float(model.score(X_train, y_train_enc))
    valid_acc_raw = float(model.score(X_valid, y_valid_enc))
    test_acc_raw = float(model.score(X_test, y_test_enc))
    baseline = max(y_test.count("WIN"), y_test.count("LOSS")) / len(y_test)
    win_idx = list(encoder.classes_).index("WIN")
    raw_valid_prob = [float(row[win_idx]) for row in model.predict_proba(X_valid)]
    raw_test_prob = [float(row[win_idx]) for row in model.predict_proba(X_test)]

    calibrator = fit_probability_calibrator(raw_valid_prob, y_valid, method=args.calibration)
    calibrated_valid_prob = apply_probability_calibrator(
        raw_valid_prob,
        calibrator=calibrator,
        method=args.calibration,
    )
    calibrated_test_prob = apply_probability_calibrator(
        raw_test_prob,
        calibrator=calibrator,
        method=args.calibration,
    )

    valid_predictions = labels_from_probabilities(calibrated_valid_prob)
    test_predictions = labels_from_probabilities(calibrated_test_prob)
    valid_acc = sum(pred == actual for pred, actual in zip(valid_predictions, y_valid, strict=True)) / len(
        y_valid
    )
    test_acc = sum(pred == actual for pred, actual in zip(test_predictions, y_test, strict=True)) / len(
        y_test
    )

    thresholds = [float(value) for value in args.thresholds.split(",") if value.strip()]
    validation_selection_stats = threshold_metrics(
        valid_rows,
        calibrated_valid_prob,
        thresholds=thresholds,
        weekly_cap=args.weekly_cap,
    )
    recommended = choose_best_threshold(
        validation_selection_stats,
        min_count=args.min_threshold_count,
    )
    test_selection_stats = threshold_metrics(
        test_rows,
        calibrated_test_prob,
        thresholds=thresholds,
        weekly_cap=args.weekly_cap,
    )
    recommended_threshold = float(recommended["threshold"]) if recommended else None
    recommended_test = (
        next(
            (
                stat
                for stat in test_selection_stats
                if recommended_threshold is not None
                and abs(float(stat["threshold"]) - recommended_threshold) < 1e-9
            ),
            None,
        )
        if recommended_threshold is not None
        else None
    )

    valid_brier = brier_score(valid_rows, calibrated_valid_prob)
    test_brier = brier_score(test_rows, calibrated_test_prob)
    valid_bins = calibration_bins(valid_rows, calibrated_valid_prob)
    test_bins = calibration_bins(test_rows, calibrated_test_prob)
    raw_valid_range = probability_range(raw_valid_prob)
    raw_test_range = probability_range(raw_test_prob)
    calibrated_valid_range = probability_range(calibrated_valid_prob)
    calibrated_test_range = probability_range(calibrated_test_prob)
    overfit_gap = train_acc - valid_acc
    lowest_threshold = min(thresholds) if thresholds else 0.0

    print(f"\n  Asset:         {args.asset}  [{args.timeframe}]")
    print(f"  Dataset:       {len(rows)} sparse candidate rows")
    print(
        f"  Split:         {args.split_mode} "
        f"(train {1 - args.validation_pct - args.test_pct:.0%} / "
        f"valid {args.validation_pct:.0%} / test {args.test_pct:.0%})"
    )
    print(f"  Train set:     {len(train_rows)} rows  ({period(train_rows)})")
    print(f"  Validation:    {len(valid_rows)} rows  ({period(valid_rows)})")
    print(f"  Test set:      {len(test_rows)} rows   ({period(test_rows)})")
    print(f"  Purge rows:    {args.purge_rows if args.split_mode == 'temporal' else 0}")
    print(f"  Weekly cap:    {args.weekly_cap}")
    print(f"  Calibration:   {args.calibration}")
    print()
    print_family_stats("Family mix (all rows)", rows)
    print_family_stats("Family mix (train)", train_rows)
    print_family_stats("Family mix (validation)", valid_rows)
    print_family_stats("Family mix (test)", test_rows)
    print()
    print(f"  Train accuracy:  {train_acc:.1%}")
    print(f"  Validation acc:  {valid_acc:.1%}  (raw model {valid_acc_raw:.1%})")
    print(f"  Test accuracy:   {test_acc:.1%}  (raw model {test_acc_raw:.1%})")
    print(f"  Baseline:        {baseline:.1%}")
    print(f"  Lift over base:  {test_acc - baseline:+.1%}")
    print(f"  Validation Brier:{valid_brier:.4f}")
    print(f"  Test Brier:      {test_brier:.4f}")
    print(
        "  Probability range:"
        f" valid raw {raw_valid_range['min']:.1%}–{raw_valid_range['max']:.1%},"
        f" calibrated {calibrated_valid_range['min']:.1%}–{calibrated_valid_range['max']:.1%};"
        f" test raw {raw_test_range['min']:.1%}–{raw_test_range['max']:.1%},"
        f" calibrated {calibrated_test_range['min']:.1%}–{calibrated_test_range['max']:.1%}"
    )
    if overfit_gap > 0.20:
        print(
            f"  Overfit warning: train accuracy exceeds validation by {overfit_gap:.1%}. "
            "Treat threshold results as exploratory until candidate quality improves."
        )
    if calibrated_valid_prob and max(calibrated_valid_prob) < lowest_threshold:
        print(
            f"  Probability warning: max validation p(WIN) is below the lowest threshold "
            f"({lowest_threshold:.0%}); calibrated selection will abstain."
        )
    print()
    print(
        classification_report(
            y_test,
            test_predictions,
            labels=["LOSS", "WIN"],
            target_names=["LOSS", "WIN"],
            zero_division=0,
        )
    )

    print("  Weekly-capped validation stats (threshold selection source):")
    for stat in validation_selection_stats:
        print(
            f"    p(WIN) >= {float(stat['threshold']):.0%}: "
            f"{int(stat['count']):4d} trades, "
            f"{float(stat['trades_per_week']):.1f}/week, "
            f"precision {float(stat['precision']):.1%}, "
            f"avg PnL {float(stat['avg_pnl_pct']):+.2%}"
        )
    if recommended:
        print(
            f"\n  Recommended threshold from validation: p(WIN) >= {float(recommended['threshold']):.0%} "
            f"({int(recommended['count'])} validation trades, "
            f"{float(recommended['trades_per_week']):.1f}/week, "
            f"{float(recommended['precision']):.1%} precision, "
            f"{float(recommended['avg_pnl_pct']):+.2%} avg PnL)"
        )
    else:
        print(
            "\n  No threshold met the minimum trade count. "
            "The model is still too sparse or too uncertain on this holdout."
        )
    print()
    print("  Weekly-capped test stats:")
    for stat in test_selection_stats:
        print(
            f"    p(WIN) >= {float(stat['threshold']):.0%}: "
            f"{int(stat['count']):4d} trades, "
            f"{float(stat['trades_per_week']):.1f}/week, "
            f"precision {float(stat['precision']):.1%}, "
            f"avg PnL {float(stat['avg_pnl_pct']):+.2%}"
        )
    if recommended_test:
        print(
            f"\n  Recommended-threshold test result: "
            f"{int(recommended_test['count'])} trades, "
            f"{float(recommended_test['trades_per_week']):.1f}/week, "
            f"{float(recommended_test['precision']):.1%} precision, "
            f"{float(recommended_test['avg_pnl_pct']):+.2%} avg PnL"
        )

    importances = permutation_importance(
        model,
        X_test,
        y_test_enc,
        n_repeats=5,
        random_state=42,
        n_jobs=1,
    )
    ranked_importances = sorted(
        zip(SIGNAL_FEATURE_NAMES, importances.importances_mean, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{safe_asset}_{args.timeframe}_signal_selector.pkl"
    meta_path = MODELS_DIR / f"{safe_asset}_{args.timeframe}_signal_selector.json"
    predictions_path = (
        MODELS_DIR / f"{safe_asset}_{args.timeframe}_signal_selector_test_predictions.csv"
    )

    bundle = {
        "model": model,
        "calibrator": calibrator,
        "calibration_method": args.calibration,
        "classes": list(encoder.classes_),
        "feature_names": SIGNAL_FEATURE_NAMES,
        "model_type": "HistGradientBoostingClassifier",
    }
    joblib.dump(bundle, model_path)

    selected_recommended = (
        selected_flags(
            test_rows,
            calibrated_test_prob,
            threshold=recommended_threshold,
            weekly_cap=args.weekly_cap,
        )
        if recommended_threshold is not None
        else [0 for _ in test_rows]
    )
    prediction_rows: list[dict[str, object]] = []
    for row, pred, raw_prob, prob, selected in zip(
        test_rows,
        test_predictions,
        raw_test_prob,
        calibrated_test_prob,
        selected_recommended,
        strict=True,
    ):
        prediction_rows.append(
            {
                **row,
                "predicted_outcome": pred,
                "predicted_win_prob_raw": round(raw_prob, 6),
                "predicted_win_prob": round(prob, 6),
                "selected_recommended": int(selected),
            }
        )

    with open(predictions_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prediction_rows)

    meta = {
        "asset": args.asset,
        "timeframe": args.timeframe,
        "dataset_rows": len(rows),
        "n_train": len(train_rows),
        "n_validation": len(valid_rows),
        "n_test": len(test_rows),
        "split_mode": args.split_mode,
        "validation_pct": args.validation_pct,
        "test_pct": args.test_pct,
        "purge_rows": args.purge_rows if args.split_mode == "temporal" else 0,
        "weekly_cap": args.weekly_cap,
        "calibration_method": args.calibration,
        "train_accuracy": round(train_acc, 4),
        "validation_accuracy": round(valid_acc, 4),
        "validation_accuracy_raw": round(valid_acc_raw, 4),
        "test_accuracy": round(test_acc, 4),
        "test_accuracy_raw": round(test_acc_raw, 4),
        "baseline_accuracy": round(baseline, 4),
        "lift": round(test_acc - baseline, 4),
        "validation_brier": round(valid_brier, 6),
        "test_brier": round(test_brier, 6),
        "raw_probability_range_validation": raw_valid_range,
        "raw_probability_range_test": raw_test_range,
        "calibrated_probability_range_validation": calibrated_valid_range,
        "calibrated_probability_range_test": calibrated_test_range,
        "overfit_gap_train_validation": round(overfit_gap, 6),
        "feature_names": SIGNAL_FEATURE_NAMES,
        "classes": list(encoder.classes_),
        "train_period": period(train_rows),
        "validation_period": period(valid_rows),
        "test_period": period(test_rows),
        "family_stats_all": family_stats(rows),
        "family_stats_train": family_stats(train_rows),
        "family_stats_validation": family_stats(valid_rows),
        "family_stats_test": family_stats(test_rows),
        "validation_selection_stats": validation_selection_stats,
        "selection_stats": test_selection_stats,
        "recommended_threshold": recommended_threshold,
        "recommended_validation_stats": recommended,
        "recommended_stats": recommended_test,
        "recommended_test_stats": recommended_test,
        "threshold_selection_source": "validation",
        "calibration_bins_validation": valid_bins,
        "calibration_bins_test": test_bins,
        "top_features": [
            {"name": name, "importance": round(float(score), 6)}
            for name, score in ranked_importances[:15]
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\n  Model → {model_path}")
    print(f"  Meta  → {meta_path}")
    print(f"  Test predictions → {predictions_path}\n")


if __name__ == "__main__":
    main()
