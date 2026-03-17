from __future__ import annotations

import os

from ai_trading_engine import EngineConfig, HybridTradingEngine
from ai_trading_engine.demo_data import build_demo_portfolio, build_demo_snapshot
from ai_trading_engine.formatters import format_decision
from ai_trading_engine.telegram_bot import send_message


def main() -> int:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID environment variable.")
        return 1

    cfg = EngineConfig()
    engine = HybridTradingEngine(cfg)
    decision = engine.evaluate(build_demo_snapshot("ETH/USDT"), build_demo_portfolio())
    text = format_decision(decision)
    send_message(token=token, chat_id=chat_id, text=text)
    print("Signal sent to Telegram.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
