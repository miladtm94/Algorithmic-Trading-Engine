#!/usr/bin/env python
"""Optimize dataset labeling settings, then train and save the final model.

This script searches across lookahead / stop ATR / reward-risk combinations
using a time-ordered validation split. It ranks trials primarily by realised
PnL among high-confidence WIN predictions, then rebuilds the best dataset and
trains a final RandomForest on the full dataset.

Outputs:
  data/features/<ASSET>_<TIMEFRAME>_dataset.csv
  data/models/<ASSET>_<TIMEFRAME>_rf.pkl
  data/models/<ASSET>_<TIMEFRAME>_rf.json
  data/models/<ASSET>_<TIMEFRAME>_optimization.json

Usage:
  python scripts/optimize_model.py --asset ETH/USDT --timeframe 1h
  python scripts/optimize_model.py --asset ETH/USDT --lookaheads 12,24,48 --stop-atrs 1.2,1.5,2.0 --reward-risks 1.5,2.0
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.WARNING)

HISTORICAL_DIR = Path(__file__).parent.parent / "data" / "historical"
FEATURES_DIR = Path(__file__).parent.parent / "data" / "features"
MODELS_DIR = Path(__file__).parent.parent / "data" / "models"


def load_candles(csv_path: Path):
    from ai_trading_engine.models import Candle  # noqa: PLC0415

    candles: list[Candle] = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            candles.append(
                Candle(
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
    return candles


def parse_number_list(raw: str, cast):
    return [cast(value.strip()) for value in raw.split(",") if value.strip()]


def to_xy(rows: list[dict], feature_names: list[str]):
    X = [[float(row.get(name, 0.0)) for name in feature_names] for row in rows]
    y = [row["outcome"] for row in rows]
    return X, y


def threshold_stats(model, X_test: list[list[float]], test_rows: list[dict], thresholds: list[float]) -> list[dict]:
    if not hasattr(model, "predict_proba") or "WIN" not in list(model.classes_):
        return []

    win_idx = list(model.classes_).index("WIN")
    probabilities = [float(row[win_idx]) for row in model.predict_proba(X_test)]
    stats: list[dict] = []
    for threshold in thresholds:
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


def evaluate_rows(
    rows: list[dict],
    feature_names: list[str],
    *,
    test_pct: float,
    trees: int,
    thresholds: list[float],
):
    from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415

    n_rows = len(rows)
    if n_rows < 50:
        raise ValueError(f"Need at least 50 rows, got {n_rows}")

    split = int(n_rows * (1 - test_pct))
    train_rows, test_rows = rows[:split], rows[split:]
    if len(train_rows) < 30 or len(test_rows) < 20:
        raise ValueError(
            f"Split produced too little data: train={len(train_rows)}, test={len(test_rows)}"
        )

    train_classes = {row["outcome"] for row in train_rows}
    test_classes = {row["outcome"] for row in test_rows}
    if len(train_classes) < 2 or len(test_classes) < 2:
        raise ValueError(
            f"Need both WIN and LOSS in train/test; got train={sorted(train_classes)} test={sorted(test_classes)}"
        )

    X_train, y_train = to_xy(train_rows, feature_names)
    X_test, y_test = to_xy(test_rows, feature_names)

    model = RandomForestClassifier(
        n_estimators=trees,
        max_depth=6,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    train_acc = float(model.score(X_train, y_train))
    test_acc = float(model.score(X_test, y_test))
    baseline = max(y_test.count("WIN"), y_test.count("LOSS")) / len(y_test)
    stats = threshold_stats(model, X_test, test_rows, thresholds)

    return {
        "model": model,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "baseline_accuracy": baseline,
        "lift": test_acc - baseline,
        "threshold_stats": stats,
        "train_period": f"{train_rows[0]['timestamp'][:10]} → {train_rows[-1]['timestamp'][:10]}",
        "test_period": f"{test_rows[0]['timestamp'][:10]} → {test_rows[-1]['timestamp'][:10]}",
    }


def save_dataset(path: Path, rows: list[dict], feature_names: list[str]) -> None:
    fieldnames = [
        "timestamp",
        "asset",
        "timeframe",
        "side",
        "outcome",
        "exit_reason",
        "entry",
        "stop_loss",
        "take_profit",
        "exit_price",
        "bars_held",
        "pnl_pct",
        "max_favorable_pct",
        "max_adverse_pct",
    ] + feature_names
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_best_trial(trials: list[dict], target_threshold: float, min_count: int) -> dict:
    eligible = [
        trial
        for trial in trials
        if trial["target_stat"]["count"] >= min_count
    ]
    pool = eligible or trials
    return max(
        pool,
        key=lambda trial: (
            trial["target_stat"]["avg_pnl_pct"],
            trial["target_stat"]["precision"],
            trial["target_stat"]["count"],
            trial["lift"],
            trial["test_accuracy"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize labeling settings and train a model using the best combo"
    )
    parser.add_argument("--asset", default=os.getenv("DEFAULT_ASSET", "ETH/USDT"))
    parser.add_argument("--timeframe", default=os.getenv("DEFAULT_TIMEFRAME", "1h"))
    parser.add_argument("--window-size", type=int, default=220)
    parser.add_argument("--lookaheads", default="12,24,48")
    parser.add_argument("--stop-atrs", default="1.2,1.5,2.0")
    parser.add_argument("--reward-risks", default="1.5,2.0")
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--min-profit-pct", type=float, default=0.0)
    parser.add_argument("--test-pct", type=float, default=0.20)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--thresholds", default="0.55,0.60,0.65,0.70")
    parser.add_argument(
        "--selection-threshold",
        type=float,
        default=0.65,
        help="Probability threshold used to rank trials. Default: 0.65",
    )
    parser.add_argument(
        "--min-threshold-count",
        type=int,
        default=30,
        help="Minimum number of holdout rows at selection threshold before a trial is considered well-supported.",
    )
    args = parser.parse_args()

    try:
        import joblib  # noqa: PLC0415
        from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415
    except ImportError:
        print("ERROR: ML dependencies missing. Install with: pip install -e '.[ml]'")
        sys.exit(1)

    from ai_trading_engine.config import EngineConfig  # noqa: PLC0415
    from ai_trading_engine.dataset import (  # noqa: PLC0415
        build_opportunity_rows_from_frames,
        build_research_frames,
    )
    from ai_trading_engine.feature_extractor import FEATURE_NAMES  # noqa: PLC0415

    lookaheads = parse_number_list(args.lookaheads, int)
    stop_atrs = parse_number_list(args.stop_atrs, float)
    reward_risks = parse_number_list(args.reward_risks, float)
    thresholds = parse_number_list(args.thresholds, float)

    safe_asset = args.asset.replace("/", "_")
    candle_path = HISTORICAL_DIR / f"{safe_asset}_{args.timeframe}.csv"
    if not candle_path.exists():
        print(f"ERROR: {candle_path} not found.")
        print(f"  Run first: python scripts/fetch_history.py --asset {args.asset} --timeframe {args.timeframe}")
        sys.exit(1)

    print(f"Loading candles from {candle_path} ...")
    candles = load_candles(candle_path)
    print(f"  {len(candles):,} candles  ({candles[0].timestamp.date()} → {candles[-1].timestamp.date()})")

    cfg = EngineConfig()
    print("Precomputing research frames ...")
    frames = build_research_frames(
        candles,
        asset=args.asset,
        timeframe=args.timeframe,
        window_size=args.window_size,
        config=cfg,
    )
    print(f"  {len(frames):,} reusable frames ready")

    combos = list(product(lookaheads, stop_atrs, reward_risks))
    print(f"Running {len(combos)} optimization trials ...\n")

    trials: list[dict] = []
    for idx, (lookahead, stop_atr, reward_risk) in enumerate(combos, start=1):
        rows, summary = build_opportunity_rows_from_frames(
            frames,
            candles,
            config=cfg,
            lookahead=lookahead,
            stop_atr=stop_atr,
            reward_risk=reward_risk,
            fee_bps=args.fee_bps,
            min_profit_pct=args.min_profit_pct,
        )
        evaluation = evaluate_rows(
            rows,
            FEATURE_NAMES,
            test_pct=args.test_pct,
            trees=args.trees,
            thresholds=thresholds,
        )
        target_stat = next(
            (stat for stat in evaluation["threshold_stats"] if abs(stat["threshold"] - args.selection_threshold) < 1e-9),
            None,
        )
        if target_stat is None:
            raise ValueError(
                f"Selection threshold {args.selection_threshold} missing from thresholds {thresholds}"
            )

        trial = {
            "lookahead": lookahead,
            "stop_atr": stop_atr,
            "reward_risk": reward_risk,
            "rows": len(rows),
            "win_rate": round(summary.win_rate, 4),
            "long_win_rate": round(summary.long_win_rate, 4),
            "short_win_rate": round(summary.short_win_rate, 4),
            "train_accuracy": round(evaluation["train_accuracy"], 4),
            "test_accuracy": round(evaluation["test_accuracy"], 4),
            "baseline_accuracy": round(evaluation["baseline_accuracy"], 4),
            "lift": round(evaluation["lift"], 4),
            "target_stat": target_stat,
            "threshold_stats": evaluation["threshold_stats"],
            "train_period": evaluation["train_period"],
            "test_period": evaluation["test_period"],
        }
        trials.append(trial)
        print(
            f"[{idx:>2}/{len(combos)}] lookahead={lookahead:<2} stop_atr={stop_atr:<3} rr={reward_risk:<3} | "
            f"test={trial['test_accuracy']:.1%} lift={trial['lift']:+.1%} | "
            f"p(WIN)>={args.selection_threshold:.0%}: {target_stat['count']:>4} rows "
            f"prec={target_stat['precision']:.1%} pnl={target_stat['avg_pnl_pct']:+.2%}"
        )

    best_trial = select_best_trial(trials, args.selection_threshold, args.min_threshold_count)
    sorted_trials = sorted(
        trials,
        key=lambda trial: (
            trial["target_stat"]["avg_pnl_pct"],
            trial["target_stat"]["precision"],
            trial["target_stat"]["count"],
            trial["lift"],
            trial["test_accuracy"],
        ),
        reverse=True,
    )

    print("\nTop trials:")
    for rank, trial in enumerate(sorted_trials[:5], start=1):
        stat = trial["target_stat"]
        print(
            f"  {rank}. lookahead={trial['lookahead']} stop_atr={trial['stop_atr']} rr={trial['reward_risk']} | "
            f"test={trial['test_accuracy']:.1%} lift={trial['lift']:+.1%} | "
            f"{stat['count']} rows at {args.selection_threshold:.0%}, "
            f"precision {stat['precision']:.1%}, pnl {stat['avg_pnl_pct']:+.2%}"
        )

    print("\nSelected best setting:")
    print(
        f"  lookahead={best_trial['lookahead']}  stop_atr={best_trial['stop_atr']}  "
        f"reward_risk={best_trial['reward_risk']}"
    )

    best_rows, best_summary = build_opportunity_rows_from_frames(
        frames,
        candles,
        config=cfg,
        lookahead=best_trial["lookahead"],
        stop_atr=best_trial["stop_atr"],
        reward_risk=best_trial["reward_risk"],
        fee_bps=args.fee_bps,
        min_profit_pct=args.min_profit_pct,
    )

    X_all, y_all = to_xy(best_rows, FEATURE_NAMES)
    final_model = RandomForestClassifier(
        n_estimators=args.trees,
        max_depth=6,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X_all, y_all)

    importances = sorted(
        zip(FEATURE_NAMES, final_model.feature_importances_, strict=True),
        key=lambda item: -item[1],
    )

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = FEATURES_DIR / f"{safe_asset}_{args.timeframe}_dataset.csv"
    model_path = MODELS_DIR / f"{safe_asset}_{args.timeframe}_rf.pkl"
    meta_path = MODELS_DIR / f"{safe_asset}_{args.timeframe}_rf.json"
    report_path = MODELS_DIR / f"{safe_asset}_{args.timeframe}_optimization.json"

    save_dataset(dataset_path, best_rows, FEATURE_NAMES)
    joblib.dump(final_model, model_path)

    meta = {
        "asset": args.asset,
        "timeframe": args.timeframe,
        "n_rows": len(best_rows),
        "train_period": best_trial["train_period"],
        "test_period": best_trial["test_period"],
        "train_accuracy": best_trial["train_accuracy"],
        "test_accuracy": best_trial["test_accuracy"],
        "baseline_accuracy": best_trial["baseline_accuracy"],
        "lift": best_trial["lift"],
        "trained_on_full_dataset": True,
        "best_params": {
            "lookahead": best_trial["lookahead"],
            "stop_atr": best_trial["stop_atr"],
            "reward_risk": best_trial["reward_risk"],
            "fee_bps": args.fee_bps,
            "min_profit_pct": args.min_profit_pct,
            "window_size": args.window_size,
        },
        "selection_threshold": args.selection_threshold,
        "selection_threshold_stat": best_trial["target_stat"],
        "threshold_stats": best_trial["threshold_stats"],
        "feature_names": FEATURE_NAMES,
        "top_features": [
            {"name": name, "importance": round(importance, 4)}
            for name, importance in importances[:10]
        ],
        "optimization_report": str(report_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    report = {
        "asset": args.asset,
        "timeframe": args.timeframe,
        "candles": len(candles),
        "search_space": {
            "lookaheads": lookaheads,
            "stop_atrs": stop_atrs,
            "reward_risks": reward_risks,
            "fee_bps": args.fee_bps,
            "min_profit_pct": args.min_profit_pct,
            "test_pct": args.test_pct,
            "trees": args.trees,
            "thresholds": thresholds,
            "selection_threshold": args.selection_threshold,
            "min_threshold_count": args.min_threshold_count,
        },
        "best_trial": best_trial,
        "final_dataset_rows": len(best_rows),
        "final_win_rate": round(best_summary.win_rate, 4),
        "trials": sorted_trials,
    }
    report_path.write_text(json.dumps(report, indent=2))

    print("\nSaved outputs:")
    print(f"  Dataset → {dataset_path}")
    print(f"  Model   → {model_path}")
    print(f"  Meta    → {meta_path}")
    print(f"  Report  → {report_path}")


if __name__ == "__main__":
    main()
