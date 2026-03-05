from __future__ import annotations

import json
from dataclasses import dataclass

import paho.mqtt.client as mqtt


@dataclass
class MqttConfig:
    host: str
    port: int


class MqttService:
    def __init__(self) -> None:
        self.client: mqtt.Client | None = None
        self.connected = False

    def connect(self, cfg: MqttConfig) -> None:
        self.disconnect()
        client = mqtt.Client(client_id="iotzy_modular_backend")

        def _on_connect(_client, _userdata, _flags, rc):
            self.connected = rc == 0

        client.on_connect = _on_connect
        client.connect(cfg.host, cfg.port, keepalive=60)
        client.loop_start()
        self.client = client

    def disconnect(self) -> None:
        if self.client is not None:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass
        self.client = None
        self.connected = False

    def publish(self, topic: str, payload: dict) -> None:
        if self.client is None:
            return
        self.client.publish(topic, json.dumps(payload))
