from __future__ import annotations

from .models import EngineDecision, FinalSignal


def _sentiment_label(score: float) -> str:
    if score >= 0.35:
        return "Bullish"
    if score <= -0.35:
        return "Bearish"
    return "Neutral"


def format_telegram_signal(signal: FinalSignal) -> str:
    c = signal.candidate
    sentiment = _sentiment_label(c.indicators.order_book_imbalance * 0.4 + c.indicators.macd_hist * 10)
    sentiment_shift = _sentiment_label(c.indicators.macd_hist * 8 + c.indicators.order_book_imbalance * 0.2)
    lines = [
        'id="trading-signal"',
        f"📊 Asset: {c.asset}",
        f"📈 Direction: {c.direction}",
        "",
        f"🧠 Market Regime: {c.regime.regime.replace('_', ' ').title()} → {c.regime.strategy.replace('_', ' ').title()}",
        "",
        f"📊 Confluence Score: {signal.confluence.total_score:.1f}%",
        "",
        "🔍 Signal Factors:",
    ]
    lines.extend(f"- {reason}" for reason in c.reasons[:6])
    lines.extend(
        [
            "",
            "💰 Trade Setup:",
            f"- Entry: {c.entry:,.2f}",
            f"- Stop Loss: {c.stop_loss:,.2f}",
            "- Take Profit:",
            f"  - TP1: {c.take_profits[0]:,.2f}",
            f"  - TP2: {c.take_profits[1]:,.2f}",
            f"  - TP3: {c.take_profits[2]:,.2f}",
            "",
            f"📊 Risk/Reward: 1:{c.risk_reward:.2f}",
            f"📉 Risk per Trade: {signal.position.risk_pct * 100:.2f}%",
            "",
            "⚠️ Invalidation:",
            c.invalidation,
            "",
            f"📰 Sentiment: {sentiment} → {sentiment_shift}",
            f"🧾 Execution: {signal.execution.order_type} (slippage cap {signal.execution.max_slippage_bps:.1f} bps)",
            f"🛡️ Validation: {signal.llm_note}",
        ]
    )
    return "\n".join(lines)


def format_no_trade(reason: str) -> str:
    return (
        'id="no-trade"\n'
        "⚠️ No Trade Opportunity\n"
        f"{reason}"
    )


def format_decision(decision: EngineDecision) -> str:
    if decision.signal is None:
        return format_no_trade(decision.no_trade_reason or "Insufficient confluence or elevated uncertainty.")
    return format_telegram_signal(decision.signal)
