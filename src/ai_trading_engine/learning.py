from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

from .config import ConfluenceWeights
from .models import FinalSignal


class TradeLearningStore:
    def __init__(self, path: str = "trade_log.csv") -> None:
        self.path = Path(path)

    def log_trade(
        self,
        signal: FinalSignal,
        outcome: str,
        slippage_bps: float,
    ) -> None:
        row = {
            "asset": signal.candidate.asset,
            "direction": signal.candidate.direction,
            "outcome": outcome,
            "slippage_bps": round(slippage_bps, 4),
            "market_regime": signal.candidate.regime.regime,
            "strategy": signal.candidate.regime.strategy,
            "signal_score": round(signal.confluence.total_score, 2),
            "risk_reward": round(signal.candidate.risk_reward, 2),
        }
        file_exists = self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


def adapt_weights(
    current: ConfluenceWeights,
    wins_by_factor: dict[str, float],
    losses_by_factor: dict[str, float],
    learning_rate: float = 0.05,
) -> ConfluenceWeights:
    raw = asdict(current)
    for key in raw.keys():
        wins = wins_by_factor.get(key, 0.0)
        losses = losses_by_factor.get(key, 0.0)
        net = wins - losses
        raw[key] = max(0.05, raw[key] + (learning_rate * net))

    updated = ConfluenceWeights(**raw).normalized()
    return updated
