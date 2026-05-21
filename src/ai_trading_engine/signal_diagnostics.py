"""Diagnostics for sparse engine-generated signal datasets."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .validation import iso_week

DatasetRow = Mapping[str, Any]
SummaryValue = float | int | str


def _float_value(value: Any) -> float:
    try:
        return float(value) if value not in (None, "") else 0.0
    except (TypeError, ValueError):
        return 0.0


def _text_value(row: DatasetRow, key: str, default: str = "UNKNOWN") -> str:
    value = row.get(key, default)
    if value in (None, ""):
        return default
    return str(value)


def summarize_rows(rows: Sequence[DatasetRow], *, label: str = "all") -> dict[str, SummaryValue]:
    """Summarize realized quality for a slice of sparse candidate rows."""
    count = len(rows)
    if count == 0:
        return {
            "label": label,
            "count": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "long_rows": 0,
            "short_rows": 0,
            "avg_pnl_pct": 0.0,
            "sum_pnl_pct": 0.0,
            "avg_net_r": 0.0,
            "sum_net_r": 0.0,
            "avg_max_favorable_r": 0.0,
            "avg_max_adverse_r": 0.0,
            "avg_bars_held": 0.0,
            "avg_signal_score": 0.0,
            "avg_setup_quality": 0.0,
            "take_profit_rate": 0.0,
            "stop_loss_rate": 0.0,
            "horizon_rate": 0.0,
            "weeks": 0,
            "trades_per_week": 0.0,
        }

    wins = sum(1 for row in rows if row.get("outcome") == "WIN")
    pnl_values = [_float_value(row.get("pnl_pct")) for row in rows]
    net_r_values = [_float_value(row.get("net_r")) for row in rows]
    exit_reasons = Counter(_text_value(row, "exit_reason") for row in rows)
    weeks = {iso_week(_text_value(row, "timestamp")) for row in rows if row.get("timestamp")}

    return {
        "label": label,
        "count": count,
        "wins": wins,
        "losses": count - wins,
        "win_rate": wins / count,
        "long_rows": sum(1 for row in rows if row.get("side") == "LONG"),
        "short_rows": sum(1 for row in rows if row.get("side") == "SHORT"),
        "avg_pnl_pct": sum(pnl_values) / count,
        "sum_pnl_pct": sum(pnl_values),
        "avg_net_r": sum(net_r_values) / count,
        "sum_net_r": sum(net_r_values),
        "avg_max_favorable_r": sum(_float_value(row.get("max_favorable_r")) for row in rows)
        / count,
        "avg_max_adverse_r": sum(_float_value(row.get("max_adverse_r")) for row in rows)
        / count,
        "avg_bars_held": sum(_float_value(row.get("bars_held")) for row in rows) / count,
        "avg_signal_score": sum(_float_value(row.get("signal_score")) for row in rows) / count,
        "avg_setup_quality": sum(_float_value(row.get("setup_quality")) for row in rows) / count,
        "take_profit_rate": exit_reasons["TAKE_PROFIT"] / count,
        "stop_loss_rate": exit_reasons["STOP_LOSS"] / count,
        "horizon_rate": exit_reasons["HORIZON"] / count,
        "weeks": len(weeks),
        "trades_per_week": count / len(weeks) if weeks else 0.0,
    }


def group_summaries(
    rows: Sequence[DatasetRow],
    group_keys: Sequence[str],
    *,
    min_count: int = 1,
) -> list[dict[str, SummaryValue]]:
    """Summarize rows grouped by one or more categorical keys."""
    grouped: dict[tuple[str, ...], list[DatasetRow]] = defaultdict(list)
    for row in rows:
        key = tuple(_text_value(row, group_key) for group_key in group_keys)
        grouped[key].append(row)

    summaries: list[dict[str, SummaryValue]] = []
    for key, group_rows in grouped.items():
        if len(group_rows) < min_count:
            continue
        label = " / ".join(key)
        summary = summarize_rows(group_rows, label=label)
        for group_key, value in zip(group_keys, key, strict=True):
            summary[group_key] = value
        summaries.append(summary)

    return sorted(
        summaries,
        key=lambda summary: (
            -int(summary["count"]),
            str(summary["label"]),
        ),
    )


def oracle_top_k_rows(
    rows: Sequence[DatasetRow],
    *,
    weekly_cap: int,
    score_key: str = "net_r",
    secondary_score_key: str = "signal_score",
) -> list[DatasetRow]:
    """Select the ex-post best rows per ISO week using realized score values."""
    if weekly_cap <= 0:
        return []

    grouped: dict[str, list[tuple[int, DatasetRow, float, float]]] = defaultdict(list)
    for index, row in enumerate(rows):
        timestamp = _text_value(row, "timestamp", default="")
        if not timestamp:
            continue
        grouped[iso_week(timestamp)].append(
            (
                index,
                row,
                _float_value(row.get(score_key)),
                _float_value(row.get(secondary_score_key)),
            )
        )

    selected: list[tuple[int, DatasetRow]] = []
    for week_rows in grouped.values():
        week_rows.sort(
            key=lambda item: (
                -item[2],
                -item[3],
                _text_value(item[1], "timestamp", default=""),
            )
        )
        selected.extend((index, row) for index, row, _, _ in week_rows[:weekly_cap])

    selected.sort(key=lambda item: item[0])
    return [row for _, row in selected]


def oracle_summary(
    rows: Sequence[DatasetRow],
    *,
    weekly_cap: int,
    score_key: str = "net_r",
) -> dict[str, SummaryValue]:
    """Summarize an ex-post weekly-capped oracle selection."""
    selected = oracle_top_k_rows(rows, weekly_cap=weekly_cap, score_key=score_key)
    summary = summarize_rows(selected, label=f"oracle_top_{weekly_cap}_{score_key}")
    summary["score_key"] = score_key
    return summary
