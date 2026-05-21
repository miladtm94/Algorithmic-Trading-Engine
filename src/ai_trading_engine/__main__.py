"""CLI entry point for AI Trading Engine.

Usage:
  python -m ai_trading_engine                     # demo, single run
  python -m ai_trading_engine --mode paper        # paper trading loop
  python -m ai_trading_engine --mode live         # live trading (real money)
  python -m ai_trading_engine --once              # one cycle and exit
  python -m ai_trading_engine --mode demo --llm  # demo + LLM validation
"""
from __future__ import annotations

import argparse
import logging
import os
import sys


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv()
    except ImportError:
        pass


def _setup_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    # Suppress noisy third-party loggers
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def main() -> None:
    _load_dotenv()

    parser = argparse.ArgumentParser(
        prog="ai-trading-engine",
        description="Production-grade hybrid algorithmic trading engine",
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "paper", "live"],
        default=os.getenv("TRADING_MODE", "demo"),
        help="Operating mode (default: demo)",
    )
    parser.add_argument(
        "--asset",
        default=os.getenv("DEFAULT_ASSET", "ETH/USDT"),
        help="Trading pair symbol (default: ETH/USDT)",
    )
    parser.add_argument(
        "--timeframe",
        default=os.getenv("DEFAULT_TIMEFRAME", "1h"),
        help="Candle timeframe (default: 1h)",
    )
    parser.add_argument(
        "--exchange",
        default=os.getenv("EXCHANGE", "binance"),
        help="ccxt exchange ID (default: binance)",
    )
    parser.add_argument(
        "--equity",
        type=float,
        default=float(os.getenv("INITIAL_EQUITY", "10000")),
        help="Starting equity in USD for paper mode (default: 10000)",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        default=os.getenv("LLM_ENABLED", "").lower() in ("1", "true", "yes"),
        help="Enable optional LLM signal validation (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single evaluation cycle and exit",
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        default=os.getenv("EXCHANGE_SANDBOX", "").lower() in ("1", "true", "yes"),
        help="Use exchange sandbox/testnet",
    )
    parser.add_argument(
        "--preset",
        choices=["default", "strict"],
        default=os.getenv("ENGINE_PRESET", "default"),
        help=(
            "Engine policy preset. 'strict' restricts signals to the "
            "RANGE_REJECTION_MEAN_REVERSION family, caps trades at 1 per ISO week, "
            "lowers the confluence threshold to 55 to match the sparse-research "
            "workflow, and halves per-trade risk. Matches the evidence as of 2026-04-25."
        ),
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )

    args = parser.parse_args()
    _setup_logging(args.log_level)

    from .config import EngineConfig  # noqa: PLC0415
    from .live_runner import TradingRunner  # noqa: PLC0415

    cfg = EngineConfig()
    cfg.llm.enabled = args.llm

    if args.preset == "strict":
        cfg.confluence_threshold = 55.0
        cfg.allowed_setup_families = frozenset({"RANGE_REJECTION_MEAN_REVERSION"})
        cfg.max_trades_per_iso_week = 1
        cfg.risk.risk_per_trade_pct = 0.005
        # Mean-reversion setups target TP1 at ~1.1R by design. A 2.0R floor
        # blocks the whole family and breaks research↔deployment alignment.
        cfg.risk.min_rr = 1.0
        # With weekly_cap=1 the engine trades at most ~52 times per year.
        # A 3-loss kill switch with that cadence locks out the engine for
        # months without informative recovery. 5 keeps a real safety brake
        # but matches the slower trade tempo of the strict preset.
        cfg.risk.kill_switch_after_losses = 5

    # Override risk params from environment if set (takes precedence over preset)
    if os.getenv("RISK_PER_TRADE_PCT"):
        cfg.risk.risk_per_trade_pct = float(os.environ["RISK_PER_TRADE_PCT"])
    if os.getenv("MAX_CONCURRENT_TRADES"):
        cfg.risk.max_concurrent_trades = int(os.environ["MAX_CONCURRENT_TRADES"])
    if os.getenv("CONFLUENCE_THRESHOLD"):
        cfg.confluence_threshold = float(os.environ["CONFLUENCE_THRESHOLD"])

    # Telegram bot (optional)
    telegram_bot = None
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if os.getenv("TELEGRAM_BOT_TOKEN") and telegram_chat_id:
        from .telegram_bot import TelegramBot  # noqa: PLC0415

        telegram_bot = TelegramBot(os.environ["TELEGRAM_BOT_TOKEN"])

    runner = TradingRunner(
        asset=args.asset,
        timeframe=args.timeframe,
        mode=args.mode,
        exchange_id=args.exchange,
        api_key=os.getenv("EXCHANGE_API_KEY", ""),
        api_secret=os.getenv("EXCHANGE_API_SECRET", ""),
        sandbox=args.sandbox,
        initial_equity=args.equity,
        engine_config=cfg,
        telegram_bot=telegram_bot,
        telegram_chat_id=telegram_chat_id,
    )

    if args.once or args.mode == "demo":
        runner.run_once()
    else:
        runner.run()


if __name__ == "__main__":
    main()
