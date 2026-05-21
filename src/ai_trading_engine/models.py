from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

Direction = Literal["LONG", "SHORT"]
RegimeType = Literal[
    "TRENDING_BULLISH",
    "TRENDING_BEARISH",
    "RANGE_BOUND",
    "HIGH_VOLATILITY",
]
StrategyType = Literal["TREND_FOLLOWING", "BREAKOUT", "MEAN_REVERSION"]
SetupFamily = Literal[
    "TREND_PULLBACK_CONTINUATION",
    "BREAKOUT_RETEST_CONTINUATION",
    "RANGE_REJECTION_MEAN_REVERSION",
    "FAILED_BREAKOUT_REVERSAL",
    "RESEARCH_TEMPLATE",
    "GENERIC",
]


@dataclass(slots=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(slots=True)
class EventRisk:
    name: str
    impact: Literal["LOW", "MEDIUM", "HIGH"]
    minutes_to_event: int


@dataclass(slots=True)
class OrderBookSnapshot:
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0.0


@dataclass(slots=True)
class MarketSnapshot:
    asset: str
    timeframe: str
    candles: list[Candle]
    order_book: OrderBookSnapshot
    spread_bps: float
    depth_usd: float
    support_levels: list[float]
    resistance_levels: list[float]
    liquidation_clusters: list[float]
    sentiment_score: float  # -1 (bearish) to +1 (bullish)
    source_prices: dict[str, float] = field(default_factory=dict)
    correlation_to_open_positions: dict[str, float] = field(default_factory=dict)
    events: list[EventRisk] = field(default_factory=list)


@dataclass(slots=True)
class PortfolioState:
    equity_usd: float
    open_positions: dict[str, Direction] = field(default_factory=dict)
    recent_results: list[Literal["WIN", "LOSS"]] = field(default_factory=list)
    # ISO-format UTC timestamps (YYYY-MM-DDTHH:MM:SS...) for recent entries,
    # used by engine-level selectivity gates such as weekly caps. Keep short:
    # only the current and recent ISO weeks need to be present.
    recent_trade_timestamps: list[str] = field(default_factory=list)

    @property
    def consecutive_losses(self) -> int:
        losses = 0
        for result in reversed(self.recent_results):
            if result != "LOSS":
                break
            losses += 1
        return losses


@dataclass(slots=True)
class IndicatorSet:
    ema20: float
    ema50: float
    ema200: float
    vwap: float
    rsi: float
    macd: float
    macd_signal: float
    macd_hist: float
    atr: float
    atr_pct: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_width_pct: float
    avg_volume: float
    volume_ratio: float
    order_book_imbalance: float


@dataclass(slots=True)
class RegimeResult:
    regime: RegimeType
    strategy: StrategyType
    confidence: float
    reason: str


@dataclass(slots=True)
class CandidateSignal:
    asset: str
    direction: Direction
    entry: float
    stop_loss: float
    take_profits: list[float]
    regime: RegimeResult
    indicators: IndicatorSet
    reasons: list[str]
    invalidation: str
    risk_reward: float
    setup_family: SetupFamily = "GENERIC"
    setup_quality: float = 0.0
    max_hold_bars: int = 24
    reference_level: float | None = None


@dataclass(slots=True)
class ConfluenceBreakdown:
    trend_alignment: float
    momentum: float
    volume_liquidity: float
    structure: float
    sentiment: float
    total_score: float


@dataclass(slots=True)
class ExecutionPlan:
    order_type: Literal["LIMIT", "MARKET"]
    max_slippage_bps: float
    expected_fill_note: str


@dataclass(slots=True)
class PositionPlan:
    quantity: float
    notional_usd: float
    risk_usd: float
    risk_pct: float


@dataclass(slots=True)
class FinalSignal:
    candidate: CandidateSignal
    confluence: ConfluenceBreakdown
    execution: ExecutionPlan
    position: PositionPlan
    news_note: str
    llm_note: str


@dataclass(slots=True)
class EngineDecision:
    signal: FinalSignal | None
    no_trade_reason: str | None = None

    @property
    def is_trade(self) -> bool:
        return self.signal is not None
