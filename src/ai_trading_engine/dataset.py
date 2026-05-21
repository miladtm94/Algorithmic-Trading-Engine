"""Dataset construction helpers for ML signal research."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .config import EngineConfig
from .confluence import score_candidate
from .feature_extractor import FEATURE_NAMES, extract_candidate_features
from .indicators import compute_indicators
from .models import (
    CandidateSignal,
    Candle,
    IndicatorSet,
    MarketSnapshot,
    OrderBookSnapshot,
    RegimeResult,
)
from .regime import classify_regime

Side = Literal["LONG", "SHORT"]
Outcome = Literal["WIN", "LOSS"]


def _compress_levels(levels: list[float], *, min_gap: float) -> list[float]:
    """Collapse nearby raw price levels into wider structural zones."""
    if not levels:
        return []
    ordered = sorted(levels)
    compressed: list[float] = [ordered[0]]
    for level in ordered[1:]:
        if abs(level - compressed[-1]) >= min_gap:
            compressed.append(level)
    return compressed


@dataclass(slots=True)
class LabelResult:
    outcome: Outcome
    exit_reason: Literal["TAKE_PROFIT", "STOP_LOSS", "HORIZON"]
    exit_price: float
    bars_held: int
    pnl_pct: float
    max_favorable_pct: float
    max_adverse_pct: float
    risk_pct: float
    net_return_pct: float
    net_r: float
    max_favorable_r: float
    max_adverse_r: float
    bars_to_target: int | None
    bars_to_stop: int | None
    r_bucket: str
    meta_label: int


@dataclass(slots=True)
class ResearchFrame:
    index: int
    timestamp: str
    asset: str
    timeframe: str
    entry: float
    indicators: IndicatorSet
    regime: RegimeResult
    support_levels: list[float]
    resistance_levels: list[float]


@dataclass(slots=True)
class OpportunitySummary:
    rows: int
    long_rows: int
    short_rows: int
    wins: int
    losses: int
    long_wins: int
    short_wins: int

    @property
    def win_rate(self) -> float:
        return self.wins / self.rows if self.rows else 0.0

    @property
    def long_win_rate(self) -> float:
        return self.long_wins / self.long_rows if self.long_rows else 0.0

    @property
    def short_win_rate(self) -> float:
        return self.short_wins / self.short_rows if self.short_rows else 0.0


def build_opportunity_rows(
    candles: list[Candle],
    asset: str,
    timeframe: str,
    *,
    config: EngineConfig | None = None,
    window_size: int = 220,
    lookahead: int = 24,
    stop_atr: float = 1.5,
    reward_risk: float = 2.0,
    fee_bps: float = 10.0,
    min_profit_pct: float = 0.0,
) -> tuple[list[dict[str, object]], OpportunitySummary]:
    """Create LONG and SHORT training examples across the full candle history.

    Each row represents a hypothetical setup at the candle close. The label is
    profitable if TP is reached first, or if the horizon exit is profitable
    after round-trip costs. If SL is reached first, the row is a LOSS.
    """
    if len(candles) < window_size + lookahead + 1:
        raise ValueError(
            f"Need at least {window_size + lookahead + 1} candles, got {len(candles)}"
        )

    frames = build_research_frames(
        candles,
        asset=asset,
        timeframe=timeframe,
        window_size=window_size,
        config=config,
    )
    return build_opportunity_rows_from_frames(
        frames,
        candles,
        config=config,
        lookahead=lookahead,
        stop_atr=stop_atr,
        reward_risk=reward_risk,
        fee_bps=fee_bps,
        min_profit_pct=min_profit_pct,
    )


def build_research_frames(
    candles: list[Candle],
    asset: str,
    timeframe: str,
    *,
    window_size: int = 220,
    config: EngineConfig | None = None,
) -> list[ResearchFrame]:
    """Precompute reusable research frames for repeated dataset sweeps."""
    if len(candles) < window_size + 2:
        raise ValueError(f"Need at least {window_size + 2} candles, got {len(candles)}")

    cfg = config or EngineConfig()
    frames: list[ResearchFrame] = []

    for idx in range(window_size, len(candles)):
        window = candles[idx - window_size : idx + 1]
        snapshot = build_research_snapshot(window, asset, timeframe)
        indicators = compute_indicators(window, snapshot.order_book)
        regime = classify_regime(indicators, cfg.regime)
        frames.append(
            ResearchFrame(
                index=idx,
                timestamp=candles[idx].timestamp.isoformat(),
                asset=asset,
                timeframe=timeframe,
                entry=candles[idx].close,
                indicators=indicators,
                regime=regime,
                support_levels=list(snapshot.support_levels),
                resistance_levels=list(snapshot.resistance_levels),
            )
        )
    return frames


def build_opportunity_rows_from_frames(
    frames: list[ResearchFrame],
    candles: list[Candle],
    *,
    config: EngineConfig | None = None,
    lookahead: int = 24,
    stop_atr: float = 1.5,
    reward_risk: float = 2.0,
    fee_bps: float = 10.0,
    min_profit_pct: float = 0.0,
) -> tuple[list[dict[str, object]], OpportunitySummary]:
    """Build labeled dataset rows from precomputed research frames."""
    if not frames:
        raise ValueError("Need at least one research frame to build dataset rows.")

    cfg = config or EngineConfig()
    rows: list[dict[str, object]] = []

    max_entry_index = len(candles) - lookahead - 1
    eligible_frames = [frame for frame in frames if frame.index <= max_entry_index]
    if not eligible_frames:
        raise ValueError(
            f"Lookahead {lookahead} is too large for {len(candles)} candles and {len(frames)} frames."
        )

    for frame in eligible_frames:
        snapshot = _build_scoring_snapshot(frame)

        for side in ("LONG", "SHORT"):
            candidate = _build_research_candidate(
                side=side,
                frame=frame,
                stop_atr=stop_atr,
                reward_risk=reward_risk,
            )
            confluence = score_candidate(candidate, snapshot, cfg.confluence_weights)
            label = label_trade_path(
                side=side,
                entry=candidate.entry,
                stop_loss=candidate.stop_loss,
                take_profit=candidate.take_profits[0],
                future_candles=candles[frame.index + 1 : frame.index + lookahead + 1],
                fee_bps=fee_bps,
                min_profit_pct=min_profit_pct,
            )
            features = extract_candidate_features(candidate, confluence)
            row: dict[str, object] = {
                "timestamp": frame.timestamp,
                "asset": frame.asset,
                "timeframe": frame.timeframe,
                "side": side,
                "outcome": label.outcome,
                "exit_reason": label.exit_reason,
                "entry": round(candidate.entry, 8),
                "stop_loss": round(candidate.stop_loss, 8),
                "take_profit": round(candidate.take_profits[0], 8),
                "exit_price": round(label.exit_price, 8),
                "bars_held": label.bars_held,
                "pnl_pct": round(label.pnl_pct, 8),
                "risk_pct": round(label.risk_pct, 8),
                "net_return_pct": round(label.net_return_pct, 8),
                "net_r": round(label.net_r, 8),
                "max_favorable_pct": round(label.max_favorable_pct, 8),
                "max_adverse_pct": round(label.max_adverse_pct, 8),
                "max_favorable_r": round(label.max_favorable_r, 8),
                "max_adverse_r": round(label.max_adverse_r, 8),
                "bars_to_target": label.bars_to_target,
                "bars_to_stop": label.bars_to_stop,
                "r_bucket": label.r_bucket,
                "meta_label": label.meta_label,
            }
            for name in FEATURE_NAMES:
                row[name] = round(features.get(name, 0.0), 8)
            rows.append(row)

    summary = _summarize(rows)
    return rows, summary


def build_research_snapshot(
    candles: list[Candle],
    asset: str,
    timeframe: str,
) -> MarketSnapshot:
    """Build a deterministic historical snapshot without live-only inputs."""
    current = candles[-1].close
    spread = current * 0.0004
    order_book = OrderBookSnapshot(
        bids=[(current - spread * level, 50_000.0) for level in range(1, 6)],
        asks=[(current + spread * level, 50_000.0) for level in range(1, 6)],
    )
    recent = candles[-80:] if len(candles) >= 80 else candles
    level_gap = current * 0.004
    supports = _compress_levels(
        [c.low for c in recent if c.low < current],
        min_gap=level_gap,
    )[-5:]
    resistances = _compress_levels(
        [c.high for c in recent if c.high > current],
        min_gap=level_gap,
    )[:5]

    return MarketSnapshot(
        asset=asset,
        timeframe=timeframe,
        candles=candles,
        order_book=order_book,
        spread_bps=4.0,
        depth_usd=1_000_000.0,
        support_levels=supports,
        resistance_levels=resistances,
        liquidation_clusters=[],
        sentiment_score=0.0,
        source_prices={"research": current},
        correlation_to_open_positions={},
        events=[],
    )


def label_trade_path(
    *,
    side: Side,
    entry: float,
    stop_loss: float,
    take_profit: float,
    future_candles: list[Candle],
    fee_bps: float = 10.0,
    min_profit_pct: float = 0.0,
) -> LabelResult:
    """Label a trade by walking future candles with conservative intrabar ordering."""
    risk_pct = abs(entry - stop_loss) / entry if entry else 0.0
    if not future_candles:
        net = -fee_bps / 10_000
        return _build_label_result(
            outcome="LOSS",
            exit_reason="HORIZON",
            exit_price=entry,
            bars_held=0,
            pnl_pct=net,
            max_favorable_pct=0.0,
            max_adverse_pct=0.0,
            risk_pct=risk_pct,
            bars_to_target=None,
            bars_to_stop=None,
        )

    cost = fee_bps / 10_000
    best = 0.0
    worst = 0.0
    bars_to_target: int | None = None
    bars_to_stop: int | None = None

    for offset, candle in enumerate(future_candles, start=1):
        if side == "LONG":
            favorable = (candle.high - entry) / entry
            adverse = (candle.low - entry) / entry
            hit_stop = candle.low <= stop_loss
            hit_target = candle.high >= take_profit
            stop_pnl = (stop_loss - entry) / entry - cost
            target_pnl = (take_profit - entry) / entry - cost
        else:
            favorable = (entry - candle.low) / entry
            adverse = (entry - candle.high) / entry
            hit_stop = candle.high >= stop_loss
            hit_target = candle.low <= take_profit
            stop_pnl = (entry - stop_loss) / entry - cost
            target_pnl = (entry - take_profit) / entry - cost

        best = max(best, favorable)
        worst = min(worst, adverse)
        if hit_target and bars_to_target is None:
            bars_to_target = offset
        if hit_stop and bars_to_stop is None:
            bars_to_stop = offset

        # If both levels are touched inside the same candle, assume the stop hit
        # first. OHLC data does not reveal intrabar path, so pessimism is safer.
        if hit_stop:
            return _build_label_result(
                outcome="LOSS",
                exit_reason="STOP_LOSS",
                exit_price=stop_loss,
                bars_held=offset,
                pnl_pct=stop_pnl,
                max_favorable_pct=best,
                max_adverse_pct=worst,
                risk_pct=risk_pct,
                bars_to_target=bars_to_target,
                bars_to_stop=bars_to_stop,
            )
        if hit_target:
            return _build_label_result(
                outcome="WIN" if target_pnl > min_profit_pct else "LOSS",
                exit_reason="TAKE_PROFIT",
                exit_price=take_profit,
                bars_held=offset,
                pnl_pct=target_pnl,
                max_favorable_pct=best,
                max_adverse_pct=worst,
                risk_pct=risk_pct,
                bars_to_target=bars_to_target,
                bars_to_stop=bars_to_stop,
            )

    last = future_candles[-1]
    if side == "LONG":
        horizon_pnl = (last.close - entry) / entry - cost
    else:
        horizon_pnl = (entry - last.close) / entry - cost

    return _build_label_result(
        outcome="WIN" if horizon_pnl > min_profit_pct else "LOSS",
        exit_reason="HORIZON",
        exit_price=last.close,
        bars_held=len(future_candles),
        pnl_pct=horizon_pnl,
        max_favorable_pct=best,
        max_adverse_pct=worst,
        risk_pct=risk_pct,
        bars_to_target=bars_to_target,
        bars_to_stop=bars_to_stop,
    )


def _build_label_result(
    *,
    outcome: Outcome,
    exit_reason: Literal["TAKE_PROFIT", "STOP_LOSS", "HORIZON"],
    exit_price: float,
    bars_held: int,
    pnl_pct: float,
    max_favorable_pct: float,
    max_adverse_pct: float,
    risk_pct: float,
    bars_to_target: int | None,
    bars_to_stop: int | None,
) -> LabelResult:
    net_r = pnl_pct / risk_pct if risk_pct > 0 else 0.0
    max_favorable_r = max_favorable_pct / risk_pct if risk_pct > 0 else 0.0
    max_adverse_r = max_adverse_pct / risk_pct if risk_pct > 0 else 0.0
    return LabelResult(
        outcome=outcome,
        exit_reason=exit_reason,
        exit_price=exit_price,
        bars_held=bars_held,
        pnl_pct=pnl_pct,
        max_favorable_pct=max_favorable_pct,
        max_adverse_pct=max_adverse_pct,
        risk_pct=risk_pct,
        net_return_pct=pnl_pct,
        net_r=net_r,
        max_favorable_r=max_favorable_r,
        max_adverse_r=max_adverse_r,
        bars_to_target=bars_to_target,
        bars_to_stop=bars_to_stop,
        r_bucket=_bucket_net_r(net_r),
        meta_label=int(outcome == "WIN"),
    )


def _bucket_net_r(net_r: float) -> str:
    if net_r <= -1.0:
        return "LEQ_NEG_1R"
    if net_r <= 0.0:
        return "NEG_1R_TO_0R"
    if net_r <= 1.0:
        return "0R_TO_1R"
    if net_r <= 2.0:
        return "1R_TO_2R"
    return "GT_2R"


def _build_research_candidate(
    *,
    side: Side,
    frame: ResearchFrame,
    stop_atr: float,
    reward_risk: float,
) -> CandidateSignal:
    entry = frame.entry
    risk = max(frame.indicators.atr * stop_atr, entry * 0.003)
    if side == "LONG":
        stop_loss = entry - risk
        take_profit = entry + reward_risk * risk
    else:
        stop_loss = entry + risk
        take_profit = entry - reward_risk * risk

    return CandidateSignal(
        asset=frame.asset,
        direction=side,
        entry=entry,
        stop_loss=stop_loss,
        take_profits=[take_profit],
        regime=frame.regime,
        indicators=frame.indicators,
        reasons=[f"Research {side.lower()} opportunity"],
        invalidation=f"Stop loss at {stop_loss:.2f}",
        risk_reward=reward_risk,
        setup_family="RESEARCH_TEMPLATE",
        setup_quality=0.0,
        max_hold_bars=24,
        reference_level=entry,
    )


def _build_scoring_snapshot(frame: ResearchFrame) -> MarketSnapshot:
    return MarketSnapshot(
        asset=frame.asset,
        timeframe=frame.timeframe,
        candles=[],
        order_book=OrderBookSnapshot(bids=[], asks=[]),
        spread_bps=4.0,
        depth_usd=1_000_000.0,
        support_levels=frame.support_levels,
        resistance_levels=frame.resistance_levels,
        liquidation_clusters=[],
        sentiment_score=0.0,
        source_prices={"research": frame.entry},
        correlation_to_open_positions={},
        events=[],
    )


def _summarize(rows: list[dict[str, object]]) -> OpportunitySummary:
    long_rows = [row for row in rows if row["side"] == "LONG"]
    short_rows = [row for row in rows if row["side"] == "SHORT"]
    wins = [row for row in rows if row["outcome"] == "WIN"]
    long_wins = [row for row in long_rows if row["outcome"] == "WIN"]
    short_wins = [row for row in short_rows if row["outcome"] == "WIN"]
    return OpportunitySummary(
        rows=len(rows),
        long_rows=len(long_rows),
        short_rows=len(short_rows),
        wins=len(wins),
        losses=len(rows) - len(wins),
        long_wins=len(long_wins),
        short_wins=len(short_wins),
    )
