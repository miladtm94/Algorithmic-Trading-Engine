"""SQLite-backed persistence for trading decisions and outcomes.

Schema:
  decisions  — every EngineDecision (trade or no-trade) with full context
  outcomes   — win/loss result linked back to a decision row
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .models import EngineDecision

logger = logging.getLogger(__name__)

_DB_PATH = Path("data/trades.db")


def set_db_path(path: str | Path) -> None:
    global _DB_PATH
    _DB_PATH = Path(path)


def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS decisions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at       TEXT NOT NULL,
                asset            TEXT NOT NULL,
                direction        TEXT,
                confluence_score REAL,
                regime           TEXT,
                strategy         TEXT,
                entry            REAL,
                stop_loss        REAL,
                take_profit_1    REAL,
                take_profit_2    REAL,
                take_profit_3    REAL,
                quantity         REAL,
                risk_usd         REAL,
                risk_pct         REAL,
                order_type       TEXT,
                reasons          TEXT,
                no_trade_reason  TEXT,
                llm_note         TEXT,
                news_note        TEXT
            );

            CREATE TABLE IF NOT EXISTS outcomes (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_id   INTEGER NOT NULL REFERENCES decisions(id),
                closed_at     TEXT NOT NULL,
                exit_price    REAL NOT NULL,
                pnl_usd       REAL NOT NULL,
                outcome       TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_decisions_asset ON decisions(asset);
            CREATE INDEX IF NOT EXISTS idx_decisions_created ON decisions(created_at);
        """)
    logger.info("Database initialised at %s", _DB_PATH)


def save_decision(decision: EngineDecision) -> int:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        if decision.signal:
            s = decision.signal
            tps = s.candidate.take_profits
            reasons_json = json.dumps(s.candidate.reasons)
            cursor = conn.execute(
                """
                INSERT INTO decisions (
                    created_at, asset, direction, confluence_score, regime, strategy,
                    entry, stop_loss, take_profit_1, take_profit_2, take_profit_3,
                    quantity, risk_usd, risk_pct, order_type, reasons, llm_note, news_note
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    now,
                    s.candidate.asset,
                    s.candidate.direction,
                    s.confluence.total_score,
                    s.candidate.regime.regime,
                    s.candidate.regime.strategy,
                    s.candidate.entry,
                    s.candidate.stop_loss,
                    tps[0] if len(tps) > 0 else None,
                    tps[1] if len(tps) > 1 else None,
                    tps[2] if len(tps) > 2 else None,
                    s.position.quantity,
                    s.position.risk_usd,
                    s.position.risk_pct,
                    s.execution.order_type,
                    reasons_json,
                    s.llm_note,
                    s.news_note,
                ),
            )
        else:
            cursor = conn.execute(
                "INSERT INTO decisions (created_at, asset, no_trade_reason) VALUES (?,?,?)",
                (now, "N/A", decision.no_trade_reason),
            )
        conn.commit()
        row_id: int = cursor.lastrowid  # type: ignore[assignment]
        logger.debug("Decision saved: id=%d", row_id)
        return row_id


def save_outcome(
    decision_id: int,
    exit_price: float,
    pnl_usd: float,
) -> None:
    outcome = "WIN" if pnl_usd > 0 else "LOSS"
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO outcomes (decision_id, closed_at, exit_price, pnl_usd, outcome) VALUES (?,?,?,?,?)",
            (decision_id, now, exit_price, pnl_usd, outcome),
        )
        conn.commit()
    logger.info("Outcome saved: decision_id=%d %s pnl=$%.2f", decision_id, outcome, pnl_usd)


def query_stats(asset: Optional[str] = None) -> dict:
    """Return aggregate performance stats from stored outcomes."""
    where = "WHERE d.asset = ?" if asset else ""
    params = (asset,) if asset else ()
    with _connect() as conn:
        rows = conn.execute(
            f"""
            SELECT o.outcome, o.pnl_usd, d.confluence_score, d.regime
            FROM outcomes o
            JOIN decisions d ON d.id = o.decision_id
            {where}
            """,
            params,
        ).fetchall()

    if not rows:
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "total_pnl": 0.0}

    wins = [r for r in rows if r["outcome"] == "WIN"]
    losses = [r for r in rows if r["outcome"] == "LOSS"]
    total_pnl = sum(r["pnl_usd"] for r in rows)
    return {
        "total": len(rows),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(rows),
        "total_pnl": total_pnl,
        "avg_win": sum(r["pnl_usd"] for r in wins) / len(wins) if wins else 0.0,
        "avg_loss": sum(r["pnl_usd"] for r in losses) / len(losses) if losses else 0.0,
    }
