from dataclasses import dataclass, field


@dataclass(slots=True)
class ConfluenceWeights:
    trend_alignment: float = 0.25
    momentum: float = 0.20
    volume_liquidity: float = 0.20
    structure: float = 0.20
    sentiment: float = 0.15

    def normalized(self) -> "ConfluenceWeights":
        total = (
            self.trend_alignment
            + self.momentum
            + self.volume_liquidity
            + self.structure
            + self.sentiment
        )
        if total <= 0:
            return ConfluenceWeights()
        return ConfluenceWeights(
            trend_alignment=self.trend_alignment / total,
            momentum=self.momentum / total,
            volume_liquidity=self.volume_liquidity / total,
            structure=self.structure / total,
            sentiment=self.sentiment / total,
        )


@dataclass(slots=True)
class DataValidationConfig:
    min_candles: int = 210
    max_price_deviation_pct: float = 0.60
    stale_after_minutes: int = 5


@dataclass(slots=True)
class RegimeConfig:
    trend_strength_threshold: float = 0.0035
    high_volatility_atr_pct: float = 0.018
    range_bb_width_pct: float = 0.022


@dataclass(slots=True)
class RiskConfig:
    risk_per_trade_pct: float = 0.01
    min_rr: float = 2.0
    max_concurrent_trades: int = 5
    correlation_limit: float = 0.80
    kill_switch_after_losses: int = 3


@dataclass(slots=True)
class ExecutionConfig:
    max_spread_bps: float = 12.0
    min_depth_usd: float = 200_000.0
    breakout_market_order_allowed: bool = True


@dataclass(slots=True)
class EventConfig:
    pause_high_impact_minutes: int = 45
    reduce_medium_impact_minutes: int = 20
    reduced_risk_multiplier: float = 0.5
    high_impact_keywords: tuple[str, ...] = ("CPI", "FOMC", "NFP", "Fed", "SEC")


@dataclass(slots=True)
class LLMConfig:
    enabled: bool = False
    model: str = "gpt-5-mini"


@dataclass(slots=True)
class EngineConfig:
    confluence_threshold: float = 75.0
    # Optional selective-deployment gates. Leaving both as `None` preserves
    # the default research engine behavior.
    allowed_setup_families: frozenset[str] | None = None
    max_trades_per_iso_week: int | None = None
    data_validation: DataValidationConfig = field(default_factory=DataValidationConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    event: EventConfig = field(default_factory=EventConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    confluence_weights: ConfluenceWeights = field(default_factory=ConfluenceWeights)
