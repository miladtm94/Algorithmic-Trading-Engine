#!/usr/bin/env python
"""Generate an HTML dashboard comparing saved model artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from html import escape
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

MODELS_DIR = Path(__file__).parent.parent / "data" / "models"
REPORTS_DIR = Path(__file__).parent.parent / "data" / "reports"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def detect_artifact_jsons(asset: str, timeframe: str) -> list[Path]:
    safe_asset = asset.replace("/", "_")
    pattern = f"{safe_asset}_{timeframe}_*.json"
    paths = sorted(MODELS_DIR.glob(pattern))

    alias = MODELS_DIR / f"{safe_asset}_{timeframe}_dnn.json"
    random_json = MODELS_DIR / f"{safe_asset}_{timeframe}_dnn_random.json"
    if alias in paths and random_json.exists():
        paths.remove(alias)
    return [path for path in paths if not path.name.endswith("_optimization.json")]


def infer_model_label(path: Path, meta: dict) -> str:
    name = path.stem
    if name.endswith("_rf"):
        return "Random Forest"
    if name.endswith("_dnn_random"):
        return "Deep NN (Random 70/30)"
    if name.endswith("_dnn_temporal"):
        return "Deep NN (Temporal 70/30)"
    if name.endswith("_dnn"):
        return "Deep NN"
    model_type = meta.get("model_type")
    return str(model_type or name)


def load_prediction_stats(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {}

    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    if not rows:
        return {}

    confusion = Counter((row["actual"], row["predicted"]) for row in rows)
    selected = [row for row in rows if row.get("selected_at_threshold") == "1"]
    by_side = Counter(row.get("side", "UNKNOWN") for row in rows)
    selected_by_side = Counter(row.get("side", "UNKNOWN") for row in selected)

    def avg_pnl(items: list[dict]) -> float:
        return sum(float(row.get("pnl_pct", 0.0)) for row in items) / len(items) if items else 0.0

    return {
        "rows": len(rows),
        "selected_rows": len(selected),
        "selected_win_rate": (
            sum(1 for row in selected if row["actual"] == "WIN") / len(selected) if selected else 0.0
        ),
        "selected_avg_pnl_pct": avg_pnl(selected),
        "overall_avg_pnl_pct": avg_pnl(rows),
        "confusion": {
            "LOSS->LOSS": confusion[("LOSS", "LOSS")],
            "LOSS->WIN": confusion[("LOSS", "WIN")],
            "WIN->LOSS": confusion[("WIN", "LOSS")],
            "WIN->WIN": confusion[("WIN", "WIN")],
        },
        "rows_by_side": dict(by_side),
        "selected_by_side": dict(selected_by_side),
    }


def collect_models(asset: str, timeframe: str) -> list[dict]:
    models: list[dict] = []
    for path in detect_artifact_jsons(asset, timeframe):
        meta = load_json(path)
        predictions_csv = meta.get("predictions_csv")
        prediction_stats = load_prediction_stats(Path(predictions_csv)) if predictions_csv else {}

        threshold_stats = meta.get("threshold_stats", [])
        selection_threshold = float(meta.get("selection_threshold", 0.65))
        selection_stat = next(
            (
                stat
                for stat in threshold_stats
                if abs(float(stat.get("threshold", 0.0)) - selection_threshold) < 1e-9
            ),
            meta.get("selection_threshold_stat", {}),
        )

        models.append(
            {
                "path": path,
                "meta": meta,
                "label": infer_model_label(path, meta),
                "selection_threshold": selection_threshold,
                "selection_stat": selection_stat,
                "threshold_stats": threshold_stats,
                "prediction_stats": prediction_stats,
            }
        )
    return models


def metric_card(model: dict) -> dict:
    meta = model["meta"]
    selection = model["selection_stat"] or {}
    prediction_stats = model["prediction_stats"]
    return {
        "label": model["label"],
        "test_accuracy": float(meta.get("test_accuracy", 0.0)),
        "baseline_accuracy": float(meta.get("baseline_accuracy", 0.0)),
        "lift": float(meta.get("lift", 0.0)),
        "threshold_precision": float(selection.get("precision", 0.0)),
        "threshold_avg_pnl_pct": float(selection.get("avg_pnl_pct", 0.0)),
        "threshold_count": int(selection.get("count", 0)),
        "signal_win_rate": float(meta.get("best_side_signal_stats", {}).get("win_rate", prediction_stats.get("selected_win_rate", 0.0))),
        "signal_avg_pnl_pct": float(meta.get("best_side_signal_stats", {}).get("avg_pnl_pct", prediction_stats.get("selected_avg_pnl_pct", 0.0))),
    }


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_signed_pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def bar_svg(items: list[tuple[str, float]], title: str, *, signed: bool = False, percent: bool = True) -> str:
    if not items:
        return ""

    width = 520
    row_height = 34
    left = 180
    chart_width = 300
    top = 18
    height = top + row_height * len(items) + 12

    if signed:
        max_abs = max(abs(value) for _, value in items) or 1.0
        scale = chart_width / (2 * max_abs)
        zero_x = left + chart_width / 2
    else:
        max_value = max(value for _, value in items) or 1.0
        scale = chart_width / max_value
        zero_x = left

    def label_text(value: float) -> str:
        if percent:
            return format_signed_pct(value) if signed else format_pct(value)
        return f"{value:+.4f}" if signed else f"{value:.4f}"

    rows = [f'<text x="0" y="14" class="chart-title">{escape(title)}</text>']
    if signed:
        rows.append(f'<line x1="{zero_x:.1f}" y1="{top - 6}" x2="{zero_x:.1f}" y2="{height - 8}" class="axis" />')
    for idx, (name, value) in enumerate(items):
        y = top + idx * row_height
        rows.append(f'<text x="0" y="{y + 18}" class="label">{escape(name)}</text>')
        if signed:
            bar_width = abs(value) * scale
            x = zero_x - bar_width if value < 0 else zero_x
            css = "bar-negative" if value < 0 else "bar"
        else:
            bar_width = value * scale
            x = zero_x
            css = "bar"
        rows.append(
            f'<rect x="{x:.1f}" y="{y + 4}" width="{bar_width:.1f}" height="18" rx="4" class="{css}" />'
        )
        rows.append(
            f'<text x="{left + chart_width + 8}" y="{y + 18}" class="value">{escape(label_text(value))}</text>'
        )
    return f'<svg viewBox="0 0 {width} {height}" class="chart">{"".join(rows)}</svg>'


def threshold_table(models: list[dict]) -> str:
    all_thresholds = sorted(
        {
            float(stat.get("threshold", 0.0))
            for model in models
            for stat in model["threshold_stats"]
        }
    )
    if not all_thresholds:
        return "<p>No threshold stats found.</p>"

    header = "".join(f"<th>{int(th * 100)}%</th>" for th in all_thresholds)
    rows = []
    for model in models:
        stats_by_threshold = {
            float(stat.get("threshold", 0.0)): stat
            for stat in model["threshold_stats"]
        }
        precision_cells = []
        pnl_cells = []
        count_cells = []
        for threshold in all_thresholds:
            stat = stats_by_threshold.get(threshold, {})
            precision_cells.append(f"<td>{format_pct(float(stat.get('precision', 0.0)))}</td>")
            pnl_cells.append(f"<td>{format_signed_pct(float(stat.get('avg_pnl_pct', 0.0)))}</td>")
            count_cells.append(f"<td>{int(stat.get('count', 0))}</td>")
        rows.append(
            "<tr>"
            f"<th rowspan='3'>{escape(model['label'])}</th>"
            f"<td class='metric-name'>Precision</td>{''.join(precision_cells)}"
            "</tr>"
            "<tr>"
            f"<td class='metric-name'>Avg PnL</td>{''.join(pnl_cells)}"
            "</tr>"
            "<tr>"
            f"<td class='metric-name'>Rows</td>{''.join(count_cells)}"
            "</tr>"
        )
    return (
        "<table class='threshold-table'>"
        "<thead><tr><th>Model</th><th>Metric</th>"
        f"{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def prediction_sections(models: list[dict]) -> str:
    blocks: list[str] = []
    for model in models:
        stats = model["prediction_stats"]
        if not stats:
            continue
        confusion = stats["confusion"]
        blocks.append(
            "<section class='subsection'>"
            f"<h3>{escape(model['label'])} Predictions</h3>"
            "<div class='grid two'>"
            "<div class='card'>"
            "<h4>Selected Signals</h4>"
            f"<p>Rows: <strong>{stats['selected_rows']}</strong></p>"
            f"<p>Win rate: <strong>{format_pct(float(stats['selected_win_rate']))}</strong></p>"
            f"<p>Avg realised PnL: <strong>{format_signed_pct(float(stats['selected_avg_pnl_pct']))}</strong></p>"
            "</div>"
            "<div class='card'>"
            "<h4>Confusion Matrix</h4>"
            "<table class='compact'><tbody>"
            f"<tr><td>LOSS -> LOSS</td><td>{confusion['LOSS->LOSS']}</td></tr>"
            f"<tr><td>LOSS -> WIN</td><td>{confusion['LOSS->WIN']}</td></tr>"
            f"<tr><td>WIN -> LOSS</td><td>{confusion['WIN->LOSS']}</td></tr>"
            f"<tr><td>WIN -> WIN</td><td>{confusion['WIN->WIN']}</td></tr>"
            "</tbody></table>"
            "</div>"
            "</div>"
            "</section>"
        )
    return "".join(blocks)


def optimization_section(asset: str, timeframe: str) -> str:
    safe_asset = asset.replace("/", "_")
    path = MODELS_DIR / f"{safe_asset}_{timeframe}_optimization.json"
    if not path.exists():
        return ""

    data = load_json(path)
    trials = data.get("trials", [])[:8]
    if not trials:
        return ""

    rows = []
    for idx, trial in enumerate(trials, start=1):
        stat = trial.get("target_stat", {})
        rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td>{trial.get('lookahead')}</td>"
            f"<td>{trial.get('stop_atr')}</td>"
            f"<td>{trial.get('reward_risk')}</td>"
            f"<td>{format_pct(float(trial.get('test_accuracy', 0.0)))}</td>"
            f"<td>{format_signed_pct(float(trial.get('lift', 0.0)))}</td>"
            f"<td>{int(stat.get('count', 0))}</td>"
            f"<td>{format_pct(float(stat.get('precision', 0.0)))}</td>"
            f"<td>{format_signed_pct(float(stat.get('avg_pnl_pct', 0.0)))}</td>"
            "</tr>"
        )
    return (
        "<section>"
        "<h2>Optimization Trials</h2>"
        "<table class='compact'>"
        "<thead><tr><th>#</th><th>Lookahead</th><th>Stop ATR</th><th>RR</th>"
        "<th>Test Acc</th><th>Lift</th><th>Rows @ Thr</th><th>Precision</th><th>Avg PnL</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "</section>"
    )


def build_html(asset: str, timeframe: str, models: list[dict]) -> str:
    cards = [metric_card(model) for model in models]
    metric_bars = [
        ("Test Accuracy", [(card["label"], card["test_accuracy"]) for card in cards], False),
        ("Lift Over Baseline", [(card["label"], card["lift"]) for card in cards], True),
        (
            "Precision at Selection Threshold",
            [(card["label"], card["threshold_precision"]) for card in cards],
            False,
        ),
        (
            "Avg PnL at Selection Threshold",
            [(card["label"], card["threshold_avg_pnl_pct"]) for card in cards],
            True,
        ),
        (
            "Signal Win Rate",
            [(card["label"], card["signal_win_rate"]) for card in cards],
            False,
        ),
    ]

    overview_rows = []
    for model, card in zip(models, cards, strict=True):
        meta = model["meta"]
        selection = model["selection_stat"] or {}
        overview_rows.append(
            "<tr>"
            f"<th>{escape(model['label'])}</th>"
            f"<td>{escape(meta.get('model_type', 'tree' if model['label'] == 'Random Forest' else 'model'))}</td>"
            f"<td>{format_pct(card['test_accuracy'])}</td>"
            f"<td>{format_pct(card['baseline_accuracy'])}</td>"
            f"<td>{format_signed_pct(card['lift'])}</td>"
            f"<td>{format_pct(card['threshold_precision'])}</td>"
            f"<td>{format_signed_pct(card['threshold_avg_pnl_pct'])}</td>"
            f"<td>{int(selection.get('count', 0))}</td>"
            "</tr>"
        )

    charts = "".join(
        f"<div class='card'>{bar_svg(items, title, signed=signed)}</div>"
        for title, items, signed in metric_bars
    )

    generated_models = "".join(
        "<li>"
        f"{escape(model['label'])}: <code>{escape(str(model['path']))}</code>"
        "</li>"
        for model in models
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model Performance Report - {escape(asset)} {escape(timeframe)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f7fb;
      --surface: #ffffff;
      --border: #d7e0ea;
      --text: #16202a;
      --muted: #607487;
      --accent: #0d9488;
      --accent-2: #0f766e;
      --negative: #dc2626;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1, h2, h3, h4 {{ margin: 0 0 10px; }}
    p {{ margin: 0 0 10px; color: var(--muted); }}
    section {{ margin-bottom: 28px; }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
    }}
    .grid {{
      display: grid;
      gap: 16px;
    }}
    .grid.two {{ grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
    .grid.charts {{ grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      font-size: 14px;
    }}
    thead th {{
      background: #edf4fb;
      color: var(--text);
    }}
    tbody tr:last-child td, tbody tr:last-child th {{ border-bottom: none; }}
    .metric-name {{
      color: var(--muted);
      width: 100px;
    }}
    .threshold-table th[rowspan] {{
      min-width: 180px;
      background: #f8fbff;
    }}
    .compact td, .compact th {{ font-size: 13px; }}
    .chart {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .chart-title {{
      fill: var(--text);
      font-size: 13px;
      font-weight: 600;
    }}
    .label {{
      fill: var(--text);
      font-size: 12px;
    }}
    .value {{
      fill: var(--muted);
      font-size: 12px;
    }}
    .bar {{
      fill: var(--accent);
    }}
    .bar-negative {{
      fill: var(--negative);
    }}
    .axis {{
      stroke: #90a4b6;
      stroke-width: 1;
      stroke-dasharray: 3 3;
    }}
    code {{
      background: #eef3f7;
      padding: 2px 6px;
      border-radius: 6px;
      font-size: 12px;
    }}
    ul {{
      margin: 8px 0 0 18px;
      padding: 0;
      color: var(--muted);
    }}
    @media (max-width: 640px) {{
      main {{ padding: 16px; }}
      th, td {{ padding: 8px; font-size: 12px; }}
    }}
  </style>
</head>
<body>
  <main>
    <section>
      <h1>Model Performance Report</h1>
      <p>{escape(asset)} [{escape(timeframe)}]</p>
      <ul>{generated_models}</ul>
    </section>

    <section>
      <h2>Overview</h2>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Type</th>
            <th>Test Accuracy</th>
            <th>Baseline</th>
            <th>Lift</th>
            <th>Precision @ Threshold</th>
            <th>Avg PnL @ Threshold</th>
            <th>Rows @ Threshold</th>
          </tr>
        </thead>
        <tbody>{''.join(overview_rows)}</tbody>
      </table>
    </section>

    <section>
      <h2>Comparison Charts</h2>
      <div class="grid charts">{charts}</div>
    </section>

    <section>
      <h2>Threshold Comparison</h2>
      {threshold_table(models)}
    </section>

    {prediction_sections(models)}
    {optimization_section(asset, timeframe)}
  </main>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HTML report comparing saved model performance")
    parser.add_argument("--asset", default="ETH/USDT")
    parser.add_argument("--timeframe", default="1h")
    args = parser.parse_args()

    models = collect_models(args.asset, args.timeframe)
    if not models:
        print(f"ERROR: No model JSON files found for {args.asset} [{args.timeframe}] in {MODELS_DIR}")
        sys.exit(1)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_asset = args.asset.replace("/", "_")
    report_path = REPORTS_DIR / f"{safe_asset}_{args.timeframe}_model_report.html"
    report_path.write_text(build_html(args.asset, args.timeframe, models), encoding="utf-8")

    print(f"Report saved → {report_path}")


if __name__ == "__main__":
    main()
