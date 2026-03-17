from __future__ import annotations

from dataclasses import dataclass

from .config import EventConfig
from .models import EventRisk


@dataclass(slots=True)
class EventDecision:
    allow_trading: bool
    risk_multiplier: float
    note: str


def apply_event_filter(events: list[EventRisk], cfg: EventConfig) -> EventDecision:
    if not events:
        return EventDecision(True, 1.0, "No high-impact events in scope.")

    high_events = [
        e
        for e in events
        if e.impact == "HIGH"
        and e.minutes_to_event <= cfg.pause_high_impact_minutes
        and any(word.lower() in e.name.lower() for word in cfg.high_impact_keywords)
    ]
    if high_events:
        event_names = ", ".join(e.name for e in high_events[:2])
        return EventDecision(
            allow_trading=False,
            risk_multiplier=0.0,
            note=f"Trading paused: high-impact event window ({event_names}).",
        )

    medium_events = [
        e
        for e in events
        if e.impact == "MEDIUM" and e.minutes_to_event <= cfg.reduce_medium_impact_minutes
    ]
    if medium_events:
        event_names = ", ".join(e.name for e in medium_events[:2])
        return EventDecision(
            allow_trading=True,
            risk_multiplier=cfg.reduced_risk_multiplier,
            note=f"Risk reduced due to medium-impact event proximity ({event_names}).",
        )

    return EventDecision(True, 1.0, "Event risk acceptable.")
