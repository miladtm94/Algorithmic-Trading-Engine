"""Validation utilities for sparse signal-selection research."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class TemporalSplit:
    train_rows: list[dict[str, str]]
    validation_rows: list[dict[str, str]]
    test_rows: list[dict[str, str]]


def iso_week(timestamp: str) -> str:
    dt = datetime.fromisoformat(timestamp)
    year, week, _ = dt.isocalendar()
    return f"{year}-W{week:02d}"


def temporal_train_validation_test_split(
    rows: Sequence[dict[str, str]],
    *,
    validation_pct: float,
    test_pct: float,
    purge_rows: int = 0,
) -> TemporalSplit:
    """Build a simple ordered train/validation/test split with purging gaps."""
    if not 0.0 < validation_pct < 1.0:
        raise ValueError(f"validation_pct must be in (0, 1), got {validation_pct}")
    if not 0.0 < test_pct < 1.0:
        raise ValueError(f"test_pct must be in (0, 1), got {test_pct}")
    if validation_pct + test_pct >= 1.0:
        raise ValueError(
            f"validation_pct + test_pct must be < 1.0, got {validation_pct + test_pct:.2f}"
        )

    n_rows = len(rows)
    validation_start = int(n_rows * (1.0 - validation_pct - test_pct))
    test_start = int(n_rows * (1.0 - test_pct))
    train_end = max(0, validation_start - purge_rows)
    validation_end = max(validation_start, test_start - purge_rows)

    split = TemporalSplit(
        train_rows=list(rows[:train_end]),
        validation_rows=list(rows[validation_start:validation_end]),
        test_rows=list(rows[test_start:]),
    )
    if not split.train_rows or not split.validation_rows or not split.test_rows:
        raise ValueError(
            "Temporal split produced an empty partition. "
            "Reduce validation/test percentages or lower purge_rows."
        )
    return split


def brier_score(
    rows: Sequence[dict[str, str]],
    probabilities: Sequence[float],
    *,
    label_key: str = "outcome",
    positive_label: str = "WIN",
) -> float:
    if len(rows) != len(probabilities):
        raise ValueError("rows and probabilities must have the same length")
    if not rows:
        return 0.0

    total = 0.0
    for row, probability in zip(rows, probabilities, strict=True):
        target = 1.0 if row.get(label_key) == positive_label else 0.0
        total += (float(probability) - target) ** 2
    return total / len(rows)


def select_top_k_per_week(
    rows: Sequence[dict[str, str]],
    scores: Sequence[float],
    *,
    weekly_cap: int,
    threshold: float | None = None,
    secondary_score_key: str = "signal_score",
) -> list[int]:
    if len(rows) != len(scores):
        raise ValueError("rows and scores must have the same length")
    if weekly_cap <= 0:
        return []

    grouped: dict[str, list[tuple[int, dict[str, str], float]]] = defaultdict(list)
    for idx, (row, score) in enumerate(zip(rows, scores, strict=True)):
        if threshold is not None and float(score) < threshold:
            continue
        grouped[iso_week(row["timestamp"])].append((idx, row, float(score)))

    selected: list[int] = []
    for items in grouped.values():
        items.sort(
            key=lambda item: (
                -item[2],
                -float(item[1].get(secondary_score_key, 0.0)),
                item[1]["timestamp"],
            )
        )
        selected.extend(idx for idx, _, _ in items[:weekly_cap])
    return selected


def selected_flags(
    rows: Sequence[dict[str, str]],
    scores: Sequence[float],
    *,
    threshold: float,
    weekly_cap: int,
    secondary_score_key: str = "signal_score",
) -> list[int]:
    selected = set(
        select_top_k_per_week(
            rows,
            scores,
            weekly_cap=weekly_cap,
            threshold=threshold,
            secondary_score_key=secondary_score_key,
        )
    )
    return [1 if idx in selected else 0 for idx in range(len(rows))]


def threshold_metrics(
    rows: Sequence[dict[str, str]],
    scores: Sequence[float],
    *,
    thresholds: Sequence[float],
    weekly_cap: int,
    label_key: str = "outcome",
    positive_label: str = "WIN",
    pnl_key: str = "pnl_pct",
    secondary_score_key: str = "signal_score",
) -> list[dict[str, float | int]]:
    metrics: list[dict[str, float | int]] = []

    for threshold in thresholds:
        selected_idx = select_top_k_per_week(
            rows,
            scores,
            weekly_cap=weekly_cap,
            threshold=float(threshold),
            secondary_score_key=secondary_score_key,
        )
        if not selected_idx:
            metrics.append(
                {
                    "threshold": float(threshold),
                    "count": 0,
                    "precision": 0.0,
                    "avg_pnl_pct": 0.0,
                    "sum_pnl_pct": 0.0,
                    "avg_signal_score": 0.0,
                    "weeks": 0,
                    "trades_per_week": 0.0,
                }
            )
            continue

        selected_rows = [rows[idx] for idx in selected_idx]
        wins = sum(1 for row in selected_rows if row.get(label_key) == positive_label)
        pnl_values = [float(row.get(pnl_key, 0.0)) for row in selected_rows]
        signal_scores = [float(row.get(secondary_score_key, 0.0)) for row in selected_rows]
        week_count = len({iso_week(row["timestamp"]) for row in selected_rows})
        metrics.append(
            {
                "threshold": float(threshold),
                "count": len(selected_rows),
                "precision": round(wins / len(selected_rows), 4),
                "avg_pnl_pct": round(sum(pnl_values) / len(pnl_values), 6),
                "sum_pnl_pct": round(sum(pnl_values), 6),
                "avg_signal_score": round(sum(signal_scores) / len(signal_scores), 4),
                "weeks": week_count,
                "trades_per_week": round(len(selected_rows) / week_count, 3)
                if week_count
                else 0.0,
            }
        )
    return metrics


def choose_best_threshold(
    stats: Sequence[dict[str, float | int]],
    *,
    min_count: int,
) -> dict[str, float | int] | None:
    eligible = [stat for stat in stats if int(stat["count"]) >= min_count]
    if not eligible:
        return None

    eligible.sort(
        key=lambda stat: (
            float(stat["avg_pnl_pct"]),
            float(stat["precision"]),
            -float(stat["trades_per_week"]),
            int(stat["count"]),
        ),
        reverse=True,
    )
    return eligible[0]


def calibration_bins(
    rows: Sequence[dict[str, str]],
    probabilities: Sequence[float],
    *,
    label_key: str = "outcome",
    positive_label: str = "WIN",
    pnl_key: str = "pnl_pct",
    bin_edges: Sequence[float] | None = None,
) -> list[dict[str, float | int]]:
    if len(rows) != len(probabilities):
        raise ValueError("rows and probabilities must have the same length")

    edges = list(bin_edges or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    bins: list[dict[str, float | int]] = []
    for idx, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        include_high = idx == len(edges) - 2
        selected: list[tuple[dict[str, str], float]] = []
        for row, probability in zip(rows, probabilities, strict=True):
            probability = float(probability)
            if low <= probability <= high if include_high else low <= probability < high:
                selected.append((row, probability))

        if not selected:
            bins.append(
                {
                    "low": low,
                    "high": high,
                    "count": 0,
                    "avg_pred": 0.0,
                    "win_rate": 0.0,
                    "avg_pnl_pct": 0.0,
                }
            )
            continue

        avg_pred = sum(probability for _, probability in selected) / len(selected)
        win_rate = (
            sum(1 for row, _ in selected if row.get(label_key) == positive_label) / len(selected)
        )
        avg_pnl = sum(float(row.get(pnl_key, 0.0)) for row, _ in selected) / len(selected)
        bins.append(
            {
                "low": low,
                "high": high,
                "count": len(selected),
                "avg_pred": round(avg_pred, 6),
                "win_rate": round(win_rate, 6),
                "avg_pnl_pct": round(avg_pnl, 6),
            }
        )
    return bins


def apply_probability_calibrator(
    probabilities: Sequence[float],
    *,
    calibrator,
    method: str,
) -> list[float]:
    if calibrator is None or method == "none":
        return [float(probability) for probability in probabilities]
    if method == "platt":
        calibrated = calibrator.predict_proba([[float(probability)] for probability in probabilities])
        return [float(row[1]) for row in calibrated]
    if method == "isotonic":
        return [float(value) for value in calibrator.transform(probabilities)]
    raise ValueError(f"Unsupported calibration method: {method}")
