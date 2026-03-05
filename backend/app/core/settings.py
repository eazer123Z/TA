from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.json"


@dataclass
class AppConfig:
    mqtt_host: str = "broker.hivemq.com"
    mqtt_port: int = 1883
    camera_index: int = 0
    vision_model: str = "yolov8m.pt"


def load_config() -> AppConfig:
    if not CONFIG_PATH.exists():
        return AppConfig()
    data = json.loads(CONFIG_PATH.read_text())
    return AppConfig(**{**asdict(AppConfig()), **data})


def save_config(cfg: AppConfig) -> None:
    CONFIG_PATH.write_text(json.dumps(asdict(cfg), indent=2))
