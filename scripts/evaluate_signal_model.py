#!/usr/bin/env python
"""Audit the sparse signal selector at a specific historical candle."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from build_dataset import load_candles

HISTORICAL_DIR = Path(__file__).parent.parent / "data" / "historical"
MODELS_DIR = Path(__file__).parent.parent / "data" / "models"
_SEP = "─" * 58


def timestamp_matches(actual: str, requested: str) -> bool:
    needle = requested.strip().replace(" ", "T")
    if len(needle) == 16:
        needle = f"{needle}:00"
    return actual.startswith(needle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit sparse signal selector at one candle")
    parser.add_argument("--asset", default=os.getenv("DEFAULT_ASSET", "ETH/USDT"))
    parser.add_argument("--timeframe", default=os.getenv("DEFAULT_TIMEFRAME", "1h"))
    parser.add_argument("--at", required=True, help="Historical candle timestamp")
    parser.add_argument("--lookahead", type=int, default=24)
    parser.add_argument("--window-size", type=int, default=220)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--min-profit-pct", type=float, default=0.0)
    parser.add_argument("--min-confluence", type=float, default=55.0)
    parser.add_argument(
        "--threshold",
        default="auto",
        help="WIN probability threshold. Default uses saved recommended threshold when available.",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Retrain on strictly prior labeled rows only.",
    )
    args = parser.parse_args()

    try:
        import joblib  # noqa: PLC0415
        from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415
    except ImportError:
        print("ERROR: ML dependencies missing. Install with: pip install -e '.[ml]'")
        sys.exit(1)

    from ai_trading_engine.config import EngineConfig  # noqa: PLC0415
    from ai_trading_engine.confluence import score_candidate  # noqa: PLC0415
    from ai_trading_engine.dataset import build_research_snapshot, label_trade_path  # noqa: PLC0415
    from ai_trading_engine.indicators import compute_indicators  # noqa: PLC0415
    from ai_trading_engine.regime import classify_regime  # noqa: PLC0415
    from ai_trading_engine.signal_generation import generate_candidate  # noqa: PLC0415
    from ai_trading_engine.signal_learning import (  # noqa: PLC0415
        SIGNAL_FEATURE_NAMES,
        build_learning_features,
        build_signal_rows,
        effective_label_lookahead,
        features_to_row,
    )
    from ai_trading_engine.validation import apply_probability_calibrator  # noqa: PLC0415

    safe_asset = args.asset.replace("/", "_")
    candle_path = HISTORICAL_DIR / f"{safe_asset}_{args.timeframe}.csv"
    if not candle_path.exists():
        print(f"ERROR: {candle_path} not found.")
        sys.exit(1)

    candles = load_candles(candle_path)
    try:
        index = next(
            idx
            for idx, candle in enumerate(candles)
            if timestamp_matches(candle.timestamp.isoformat(), args.at)
        )
    except StopIteration:
        print(f"ERROR: No candle found at {args.at!r}.")
        sys.exit(1)

    if index < args.window_size:
        print("ERROR: Not enough warmup candles before requested timestamp.")
        sys.exit(1)
    if index + 1 >= len(candles):
        print("ERROR: Not enough future candles to realize the label at this timestamp.")
        sys.exit(1)

    cfg = EngineConfig()
    window = candles[index - args.window_size : index + 1]
    snapshot = build_research_snapshot(window, args.asset, args.timeframe)
    indicators = compute_indicators(window, snapshot.order_book)
    regime = classify_regime(indicators, cfg.regime)
    candidate = generate_candidate(snapshot, regime, indicators)

    print(f"\n{_SEP}")
    print(f"  SPARSE SIGNAL AUDIT   {args.asset}  [{args.timeframe}]")
    print(_SEP)
    print(f"  Timestamp:   {candles[index].timestamp.isoformat()}")
    print(f"  Walk-forward: {'yes' if args.walk_forward else 'no'}")
    print(f"  Dataset gate: confluence >= {args.min_confluence:.1f}")
    print()

    if candidate is None:
        print("  Engine candidate: none")
        print("  Recommendation:  NO TRADE")
        print(_SEP)
        return

    confluence = score_candidate(candidate, snapshot, cfg.confluence_weights)
    print(
        f"  Candidate:   {candidate.direction}  "
        f"score {confluence.total_score:.1f}  RR {candidate.risk_reward:.2f}"
    )
    print(f"  Regime:      {regime.regime} / {regime.strategy}  confidence {regime.confidence:.1%}")

    if confluence.total_score < args.min_confluence:
        print("  Candidate failed sparse dataset gate.")
        print("  Recommendation:  NO TRADE")
        print(_SEP)
        return

    label_lookahead = effective_label_lookahead(candidate, args.lookahead)
    if index + label_lookahead >= len(candles):
        print(
            "ERROR: Not enough future candles to realize the family-aware label "
            f"({label_lookahead} bars) at this timestamp."
        )
        sys.exit(1)
    label = label_trade_path(
        side=candidate.direction,
        entry=candidate.entry,
        stop_loss=candidate.stop_loss,
        take_profit=candidate.take_profits[0],
        future_candles=candles[index + 1 : index + label_lookahead + 1],
        fee_bps=args.fee_bps,
        min_profit_pct=args.min_profit_pct,
    )
    features = build_learning_features(
        candidate,
        confluence,
        snapshot,
        engine_threshold=cfg.confluence_threshold,
    )
    row = features_to_row(features)

    if args.walk_forward:
        prior_rows, _ = build_signal_rows(
            candles[: index + 1],
            asset=args.asset,
            timeframe=args.timeframe,
            config=cfg,
            window_size=args.window_size,
            lookahead=args.lookahead,
            fee_bps=args.fee_bps,
            min_profit_pct=args.min_profit_pct,
            min_confluence=args.min_confluence,
        )
        if len(prior_rows) < 30:
            print(f"ERROR: Only {len(prior_rows)} strictly prior rows are available.")
            sys.exit(1)

        from sklearn.preprocessing import LabelEncoder  # noqa: PLC0415

        X_train = [
            [float(prior_row.get(name, 0.0)) for name in SIGNAL_FEATURE_NAMES]
            for prior_row in prior_rows
        ]
        y_train = [prior_row["outcome"] for prior_row in prior_rows]
        encoder = LabelEncoder()
        y_train_enc = encoder.fit_transform(y_train)
        class_counts = {
            label_value: y_train_enc.tolist().count(label_value) for label_value in set(y_train_enc)
        }
        sample_weights = [
            len(y_train_enc) / (len(class_counts) * class_counts[value]) for value in y_train_enc
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
        classes = list(encoder.classes_)
        threshold = 0.65 if args.threshold == "auto" else float(args.threshold)
        model_note = f"walk-forward retrained on {len(prior_rows)} prior rows"
        prediction = int(model.predict([row])[0])
        raw_win_prob = float(model.predict_proba([row])[0][classes.index("WIN")])
        win_prob = raw_win_prob
        predicted_label = str(encoder.inverse_transform([prediction])[0])
    else:
        bundle_path = MODELS_DIR / f"{safe_asset}_{args.timeframe}_signal_selector.pkl"
        meta_path = MODELS_DIR / f"{safe_asset}_{args.timeframe}_signal_selector.json"
        if not bundle_path.exists():
            print(f"ERROR: {bundle_path} not found.")
            print("  Run train_signal_model.py first or use --walk-forward.")
            sys.exit(1)
        bundle = joblib.load(bundle_path)
        model = bundle["model"] if isinstance(bundle, dict) else bundle
        classes = list(bundle.get("classes", [])) if isinstance(bundle, dict) else list(model.classes_)
        calibrator = bundle.get("calibrator") if isinstance(bundle, dict) else None
        calibration_method = (
            str(bundle.get("calibration_method", "none")) if isinstance(bundle, dict) else "none"
        )
        meta = {}
        if meta_path.exists():
            import json  # noqa: PLC0415

            meta = json.loads(meta_path.read_text())
        threshold = (
            float(meta["recommended_threshold"])
            if args.threshold == "auto" and meta.get("recommended_threshold") is not None
            else (0.65 if args.threshold == "auto" else float(args.threshold))
        )
        model_note = (
            str(bundle.get("model_type", "saved selector model"))
            if isinstance(bundle, dict)
            else type(model).__name__
        )
        if calibration_method != "none":
            model_note = f"{model_note} + {calibration_method} calibration"
        prediction = int(model.predict([row])[0])
        raw_win_prob = float(model.predict_proba([row])[0][classes.index("WIN")])
        win_prob = apply_probability_calibrator(
            [raw_win_prob],
            calibrator=calibrator,
            method=calibration_method,
        )[0]
        predicted_label = "WIN" if classes[prediction] == "WIN" else "LOSS"

    take_trade = win_prob >= threshold
    print(f"  Model:       {model_note}")
    if abs(win_prob - raw_win_prob) > 1e-9:
        print(
            f"  Prediction:  {predicted_label}  "
            f"raw p(WIN)={raw_win_prob:.1%}  calibrated={win_prob:.1%}"
        )
    else:
        print(f"  Prediction:  {predicted_label}  p(WIN)={win_prob:.1%}")
    print(f"  Threshold:   {threshold:.0%}")
    print(f"  Decision:    {'TAKE TRADE' if take_trade else 'NO TRADE'}")
    print(
        f"  Trade:       entry {candidate.entry:.2f}  "
        f"SL {candidate.stop_loss:.2f}  TP {candidate.take_profits[0]:.2f}"
    )
    if label_lookahead != args.lookahead:
        print(f"  Label path:  {label_lookahead} bars  (family max-hold)")
    print(f"  Actual:      {label.outcome} via {label.exit_reason}")
    print(
        f"  Result:      bars {label.bars_held}  "
        f"PnL {label.pnl_pct:+.2%}  NetR {label.net_r:+.2f}  "
        f"MFE {label.max_favorable_r:+.2f}R  "
        f"MAE {label.max_adverse_r:+.2f}R"
    )
    print(_SEP)


if __name__ == "__main__":
    main()
