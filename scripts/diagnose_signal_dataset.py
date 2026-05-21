#!/usr/bin/env python
"""Audit sparse signal datasets for family-level expectancy and oracle limits."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

FEATURES_DIR = Path(__file__).parent.parent / "data" / "features"


def _load_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: row["timestamp"])
    return rows


def _period(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "empty"
    return f"{rows[0]['timestamp'][:10]} -> {rows[-1]['timestamp'][:10]}"


def _fmt_pct(value: object) -> str:
    return f"{float(value):+.2%}"


def _fmt_rate(value: object) -> str:
    return f"{float(value):.1%}"


def _print_summary(label: str, summary: dict[str, object]) -> None:
    print(
        f"  {label:<14} "
        f"{int(summary['count']):>4d} rows | "
        f"win {_fmt_rate(summary['win_rate'])} | "
        f"avg pnl {_fmt_pct(summary['avg_pnl_pct'])} | "
        f"avg net R {float(summary['avg_net_r']):+.2f} | "
        f"TP/SL/H {_fmt_rate(summary['take_profit_rate'])}/"
        f"{_fmt_rate(summary['stop_loss_rate'])}/"
        f"{_fmt_rate(summary['horizon_rate'])} | "
        f"{float(summary['trades_per_week']):.2f}/week"
    )


def _print_group_table(
    title: str,
    summaries: list[dict[str, object]],
    *,
    limit: int,
) -> None:
    print(f"\n{title}:")
    if not summaries:
        print("  no rows")
        return
    for summary in summaries[:limit]:
        print(
            f"  {str(summary['label'])[:42]:<42} "
            f"{int(summary['count']):>4d} rows | "
            f"win {_fmt_rate(summary['win_rate'])} | "
            f"pnl {_fmt_pct(summary['avg_pnl_pct'])} | "
            f"net R {float(summary['avg_net_r']):+.2f} | "
            f"MFE {float(summary['avg_max_favorable_r']):+.2f}R | "
            f"MAE {float(summary['avg_max_adverse_r']):+.2f}R | "
            f"score {float(summary['avg_signal_score']):.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose sparse signal dataset quality")
    parser.add_argument("--asset", default=os.getenv("DEFAULT_ASSET", "ETH/USDT"))
    parser.add_argument("--timeframe", default=os.getenv("DEFAULT_TIMEFRAME", "1h"))
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional explicit signal dataset CSV path.",
    )
    parser.add_argument("--validation-pct", type=float, default=0.15)
    parser.add_argument("--test-pct", type=float, default=0.20)
    parser.add_argument("--purge-rows", type=int, default=24)
    parser.add_argument("--weekly-cap", type=int, default=10)
    parser.add_argument("--min-group-size", type=int, default=3)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path to write machine-readable diagnostics.",
    )
    args = parser.parse_args()

    from ai_trading_engine.signal_diagnostics import (  # noqa: PLC0415
        group_summaries,
        oracle_summary,
        oracle_top_k_rows,
        summarize_rows,
    )
    from ai_trading_engine.validation import temporal_train_validation_test_split  # noqa: PLC0415

    safe_asset = args.asset.replace("/", "_")
    dataset_path = args.dataset or FEATURES_DIR / f"{safe_asset}_{args.timeframe}_signal_dataset.csv"
    if not dataset_path.exists():
        print(f"ERROR: {dataset_path} not found.")
        print(f"  Run first: python scripts/build_signal_dataset.py --asset {args.asset}")
        sys.exit(1)

    rows = _load_rows(dataset_path)
    if not rows:
        print(f"ERROR: {dataset_path} has no rows.")
        sys.exit(1)

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

    split_rows = {
        "all": rows,
        "train": split.train_rows,
        "validation": split.validation_rows,
        "test": split.test_rows,
    }

    print(f"\nSignal dataset diagnosis for {args.asset} [{args.timeframe}]")
    print(f"Dataset: {dataset_path}")
    print(f"Rows: {len(rows):,}  Period: {_period(rows)}")
    print(
        "Split: "
        f"train {len(split.train_rows):,} ({_period(split.train_rows)}), "
        f"validation {len(split.validation_rows):,} ({_period(split.validation_rows)}), "
        f"test {len(split.test_rows):,} ({_period(split.test_rows)}), "
        f"purge_rows={args.purge_rows}"
    )

    print("\nOverall realized quality:")
    for label, current_rows in split_rows.items():
        _print_summary(label, summarize_rows(current_rows, label=label))

    print("\nWeekly-capped realized oracle:")
    oracle_by_split: dict[str, dict[str, object]] = {}
    for label, current_rows in split_rows.items():
        summary = oracle_summary(current_rows, weekly_cap=args.weekly_cap, score_key="net_r")
        oracle_by_split[label] = summary
        _print_summary(label, summary)

    test_oracle_rows = oracle_top_k_rows(
        split.test_rows,
        weekly_cap=args.weekly_cap,
        score_key="net_r",
    )

    grouped_outputs = {
        "family_all": group_summaries(
            rows,
            ["setup_family"],
            min_count=args.min_group_size,
        ),
        "family_test": group_summaries(
            split.test_rows,
            ["setup_family"],
            min_count=args.min_group_size,
        ),
        "family_side_test": group_summaries(
            split.test_rows,
            ["setup_family", "side"],
            min_count=args.min_group_size,
        ),
        "family_exit_test": group_summaries(
            split.test_rows,
            ["setup_family", "exit_reason"],
            min_count=args.min_group_size,
        ),
        "oracle_family_test": group_summaries(
            test_oracle_rows,
            ["setup_family"],
            min_count=1,
        ),
    }

    _print_group_table("Family quality (all rows)", grouped_outputs["family_all"], limit=args.limit)
    _print_group_table("Family quality (test rows)", grouped_outputs["family_test"], limit=args.limit)
    _print_group_table(
        "Family + side quality (test rows)",
        grouped_outputs["family_side_test"],
        limit=args.limit,
    )
    _print_group_table(
        "Family + exit reason damage (test rows)",
        grouped_outputs["family_exit_test"],
        limit=args.limit,
    )
    _print_group_table(
        "Oracle top-k family mix (test rows)",
        grouped_outputs["oracle_family_test"],
        limit=args.limit,
    )

    if args.json_output is not None:
        payload = {
            "asset": args.asset,
            "timeframe": args.timeframe,
            "dataset": str(dataset_path),
            "rows": len(rows),
            "period": _period(rows),
            "split": {
                label: {
                    "rows": len(current_rows),
                    "period": _period(current_rows),
                    "summary": summarize_rows(current_rows, label=label),
                }
                for label, current_rows in split_rows.items()
            },
            "weekly_cap": args.weekly_cap,
            "oracle": oracle_by_split,
            "groups": grouped_outputs,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote diagnostics JSON -> {args.json_output}")

    print()


if __name__ == "__main__":
    main()
