from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class ActivityEvent:
    event_type: str
    summary: str


class ActivityService:
    def __init__(self) -> None:
        self._events: list[ActivityEvent] = []

    def add(self, event_type: str, summary: str) -> None:
        self._events.append(ActivityEvent(event_type=event_type, summary=summary))
        self._events = self._events[-500:]

    def summary(self) -> list[dict]:
        agg: dict[tuple[str, str], int] = defaultdict(int)
        for e in self._events:
            agg[(e.event_type, e.summary)] += 1
        return [
            {"event_type": k[0], "summary": k[1], "count": v}
            for k, v in sorted(agg.items(), key=lambda x: x[1], reverse=True)
        ]
