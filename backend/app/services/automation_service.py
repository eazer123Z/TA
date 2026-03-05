from __future__ import annotations


class AutomationService:
    def evaluate_light_rule(self, brightness: float, on_threshold: float = 0.35, off_threshold: float = 0.55) -> int | None:
        if brightness < on_threshold:
            return 1
        if brightness > off_threshold:
            return 0
        return None
