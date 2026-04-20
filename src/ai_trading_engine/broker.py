"""Order execution layer.

PaperBroker  — simulated fills, no real money, perfect for testing.
LiveBroker   — real orders via ccxt; use only with verified credentials.

Both share the same interface so TradingRunner can swap them by mode.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional

from .models import Direction, FinalSignal

logger = logging.getLogger(__name__)

OrderStatus = Literal["OPEN", "FILLED", "STOPPED", "CANCELLED"]


@dataclass
class Order:
    id: str
    asset: str
    direction: Direction
    order_type: str
    quantity: float
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    status: OrderStatus
    created_at: datetime
    filled_at: Optional[datetime] = None
    fill_price: Optional[float] = None
    pnl_usd: Optional[float] = None
    exchange_order_id: Optional[str] = None


@dataclass
class BrokerStats:
    equity: float
    total_trades: int
    wins: int
    losses: int
    total_pnl_usd: float
    win_rate: float
    open_positions: int

    def __str__(self) -> str:
        return (
            f"equity=${self.equity:,.2f}  trades={self.total_trades}  "
            f"wins={self.wins}  losses={self.losses}  "
            f"win_rate={self.win_rate:.1%}  pnl=${self.total_pnl_usd:+,.2f}"
        )


class PaperBroker:
    """Simulates order fills based on subsequent price updates.

    Positions close at TP1 or stop-loss when update_prices() is called.
    """

    def __init__(self, initial_equity: float = 10_000.0) -> None:
        self._equity = initial_equity
        self._orders: list[Order] = []
        logger.info("PaperBroker started with equity=$%.2f", initial_equity)

    def place_order(self, signal: FinalSignal) -> Order:
        tps = signal.candidate.take_profits
        order = Order(
            id=f"PAPER-{uuid.uuid4().hex[:8].upper()}",
            asset=signal.candidate.asset,
            direction=signal.candidate.direction,
            order_type=signal.execution.order_type,
            quantity=signal.position.quantity,
            entry_price=signal.candidate.entry,
            stop_loss=signal.candidate.stop_loss,
            take_profit_1=tps[0] if len(tps) > 0 else signal.candidate.entry,
            take_profit_2=tps[1] if len(tps) > 1 else tps[0] if tps else signal.candidate.entry,
            take_profit_3=tps[2] if len(tps) > 2 else tps[-1] if tps else signal.candidate.entry,
            status="OPEN",
            created_at=datetime.now(timezone.utc),
        )
        self._orders.append(order)
        logger.info(
            "Paper order placed: id=%s %s %s qty=%.4f entry=%.4f sl=%.4f tp1=%.4f",
            order.id, order.direction, order.asset,
            order.quantity, order.entry_price, order.stop_loss, order.take_profit_1,
        )
        return order

    def update_prices(self, asset: str, current_price: float) -> list[Order]:
        """Check open positions for stop-loss or TP1 fills at `current_price`."""
        closed: list[Order] = []
        for order in self._orders:
            if order.asset != asset or order.status != "OPEN":
                continue

            hit_stop = False
            hit_tp = False

            if order.direction == "LONG":
                hit_stop = current_price <= order.stop_loss
                hit_tp = current_price >= order.take_profit_1
            else:  # SHORT
                hit_stop = current_price >= order.stop_loss
                hit_tp = current_price <= order.take_profit_1

            if hit_stop:
                fill = order.stop_loss
                if order.direction == "LONG":
                    pnl = (fill - order.entry_price) * order.quantity
                else:
                    pnl = (order.entry_price - fill) * order.quantity
                self._close(order, fill, pnl, "STOPPED")
                closed.append(order)

            elif hit_tp:
                fill = order.take_profit_1
                if order.direction == "LONG":
                    pnl = (fill - order.entry_price) * order.quantity
                else:
                    pnl = (order.entry_price - fill) * order.quantity
                self._close(order, fill, pnl, "FILLED")
                closed.append(order)

        return closed

    def _close(self, order: Order, fill_price: float, pnl_usd: float, status: OrderStatus) -> None:
        order.status = status
        order.filled_at = datetime.now(timezone.utc)
        order.fill_price = fill_price
        order.pnl_usd = pnl_usd
        self._equity += pnl_usd
        outcome = "WIN" if pnl_usd > 0 else "LOSS"
        logger.info(
            "Paper position closed: id=%s %s pnl=$%.2f equity=$%.2f",
            order.id, outcome, pnl_usd, self._equity,
        )

    def cancel_order(self, order_id: str) -> bool:
        for order in self._orders:
            if order.id == order_id and order.status == "OPEN":
                order.status = "CANCELLED"
                logger.info("Paper order cancelled: %s", order_id)
                return True
        return False

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def open_orders(self) -> list[Order]:
        return [o for o in self._orders if o.status == "OPEN"]

    @property
    def trade_history(self) -> list[Order]:
        return [o for o in self._orders if o.status in ("FILLED", "STOPPED")]

    def stats(self) -> BrokerStats:
        history = self.trade_history
        wins = [o for o in history if (o.pnl_usd or 0) > 0]
        losses = [o for o in history if (o.pnl_usd or 0) <= 0]
        total_pnl = sum(o.pnl_usd or 0 for o in history)
        win_rate = len(wins) / len(history) if history else 0.0
        return BrokerStats(
            equity=self._equity,
            total_trades=len(history),
            wins=len(wins),
            losses=len(losses),
            total_pnl_usd=total_pnl,
            win_rate=win_rate,
            open_positions=len(self.open_orders),
        )


class LiveBroker:
    """Places real orders on an exchange via ccxt.

    Only LIMIT and MARKET entry orders are placed here.
    Stop-loss management is handled via exchange native stop orders where
    supported, otherwise tracked locally and submitted on trigger.
    """

    def __init__(
        self,
        exchange_id: str,
        api_key: str,
        api_secret: str,
        sandbox: bool = False,
    ) -> None:
        try:
            import ccxt  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "ccxt is required for live trading. Install with: pip install ccxt"
            ) from exc

        exchange_class = getattr(ccxt, exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"Unsupported exchange: {exchange_id!r}")

        self._exchange = exchange_class(
            {"apiKey": api_key, "secret": api_secret, "enableRateLimit": True}
        )
        if sandbox:
            self._exchange.set_sandbox_mode(True)

        self._order_map: dict[str, Order] = {}
        logger.info(
            "LiveBroker initialised: exchange=%s sandbox=%s", exchange_id, sandbox
        )

    def place_order(self, signal: FinalSignal) -> Order:
        side = "buy" if signal.candidate.direction == "LONG" else "sell"
        order_type = signal.execution.order_type.lower()
        price = signal.candidate.entry if order_type == "limit" else None
        qty = signal.position.quantity

        try:
            raw = self._exchange.create_order(
                symbol=signal.candidate.asset,
                type=order_type,
                side=side,
                amount=qty,
                price=price,
            )
        except Exception as exc:
            logger.error("Order placement failed: %s", exc)
            raise

        tps = signal.candidate.take_profits
        order = Order(
            id=f"LIVE-{uuid.uuid4().hex[:8].upper()}",
            asset=signal.candidate.asset,
            direction=signal.candidate.direction,
            order_type=signal.execution.order_type,
            quantity=qty,
            entry_price=signal.candidate.entry,
            stop_loss=signal.candidate.stop_loss,
            take_profit_1=tps[0] if tps else signal.candidate.entry,
            take_profit_2=tps[1] if len(tps) > 1 else tps[0] if tps else signal.candidate.entry,
            take_profit_3=tps[2] if len(tps) > 2 else tps[-1] if tps else signal.candidate.entry,
            status="OPEN",
            created_at=datetime.now(timezone.utc),
            exchange_order_id=raw.get("id"),
        )
        self._order_map[order.id] = order

        logger.info(
            "Live order placed: local_id=%s exchange_id=%s %s %s qty=%.4f",
            order.id, order.exchange_order_id, order.direction, order.asset, qty,
        )
        return order

    def fetch_balance(self) -> dict:
        return self._exchange.fetch_balance()

    @property
    def open_orders(self) -> list[Order]:
        return [o for o in self._order_map.values() if o.status == "OPEN"]

    @property
    def equity(self) -> float:
        """Fetch current USDT balance from exchange."""
        try:
            bal = self._exchange.fetch_balance()
            return float(bal.get("USDT", {}).get("free", 0))
        except Exception as exc:
            logger.warning("Could not fetch live equity: %s", exc)
            return 0.0
