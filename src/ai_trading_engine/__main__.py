from __future__ import annotations

import argparse

from .config import EngineConfig
from .demo_data import build_demo_portfolio, build_demo_snapshot
from .engine import HybridTradingEngine
from .formatters import format_decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hybrid trading engine demo.")
    parser.add_argument("--asset", default="ETH/USDT", help="Asset symbol")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM validation if OPENAI_API_KEY is available.",
    )
    args = parser.parse_args()

    cfg = EngineConfig()
    cfg.llm.enabled = args.llm
    engine = HybridTradingEngine(cfg)

    snapshot = build_demo_snapshot(args.asset)
    portfolio = build_demo_portfolio()
    decision = engine.evaluate(snapshot, portfolio)
    print(format_decision(decision))


if __name__ == "__main__":
    main()
