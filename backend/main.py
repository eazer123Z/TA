from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import paho.mqtt.client as mqtt
from fastapi import FastAPI
from pydantic import BaseModel

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class Topics:
    temperature: str = "iotzy/sensor/temperature"
    presence: str = "iotzy/sensor/presence"
    lamp_control: str = "iotzy/device/lamp/control"


@dataclass
class Automation:
    lamp_enabled: bool = True
    lamp_auto_on: bool = True
    lamp_auto_off_delay: int = 30


@dataclass
class BackendConfig:
    mqtt_host: str = "broker.hivemq.com"
    mqtt_port: int = 1883
    mqtt_use_ssl: bool = False
    mqtt_username: str = ""
    mqtt_password: str = ""
    topics: Topics = field(default_factory=Topics)
    automation: Automation = field(default_factory=Automation)


class TopicsPayload(BaseModel):
    temperature: Optional[str] = None
    presence: Optional[str] = None
    lamp_control: Optional[str] = None


class AutomationPayload(BaseModel):
    lamp_enabled: Optional[bool] = None
    lamp_auto_on: Optional[bool] = None
    lamp_auto_off_delay: Optional[int] = None


class ConfigPayload(BaseModel):
    mqtt_host: Optional[str] = None
    mqtt_port: Optional[int] = None
    mqtt_use_ssl: Optional[bool] = None
    mqtt_username: Optional[str] = None
    mqtt_password: Optional[str] = None
    topics: Optional[TopicsPayload] = None
    automation: Optional[AutomationPayload] = None


class SensorState(BaseModel):
    temperature: Optional[float] = None
    presence: bool = False
    last_seen: Optional[str] = None
    lamp_state: int = 0


app = FastAPI(title="Smart Room IoT Backend")
config_lock = threading.Lock()
state_lock = threading.Lock()

config = BackendConfig()
state = SensorState()

mqtt_client: Optional[mqtt.Client] = None
last_presence_at: Optional[float] = None


def load_config() -> BackendConfig:
    if not CONFIG_PATH.exists():
        return BackendConfig()
    data = json.loads(CONFIG_PATH.read_text())
    topics = Topics(**data.get("topics", {}))
    automation = Automation(**data.get("automation", {}))
    return BackendConfig(
        mqtt_host=data.get("mqtt_host", "broker.hivemq.com"),
        mqtt_port=data.get("mqtt_port", 1883),
        mqtt_use_ssl=data.get("mqtt_use_ssl", False),
        mqtt_username=data.get("mqtt_username", ""),
        mqtt_password=data.get("mqtt_password", ""),
        topics=topics,
        automation=automation,
    )


def save_config(new_config: BackendConfig) -> None:
    CONFIG_PATH.write_text(json.dumps(asdict(new_config), indent=2))


def clone_config(source: BackendConfig) -> BackendConfig:
    return BackendConfig(
        mqtt_host=source.mqtt_host,
        mqtt_port=source.mqtt_port,
        mqtt_use_ssl=source.mqtt_use_ssl,
        mqtt_username=source.mqtt_username,
        mqtt_password=source.mqtt_password,
        topics=Topics(**asdict(source.topics)),
        automation=Automation(**asdict(source.automation)),
    )


def mqtt_on_connect(client: mqtt.Client, _userdata, _flags, _rc):
    with config_lock:
        topics = config.topics
    client.subscribe(topics.temperature)
    client.subscribe(topics.presence)


def mqtt_on_message(_client: mqtt.Client, _userdata, msg: mqtt.MQTTMessage):
    global last_presence_at
    with config_lock:
        topics = config.topics
        automation = config.automation

    if msg.topic == topics.temperature:
        try:
            payload = json.loads(msg.payload.decode())
            temp = payload.get("value") or payload.get("temperature") or payload.get("temp")
            if temp is None:
                return
            with state_lock:
                state.temperature = float(temp)
                state.last_seen = now_iso()
        except json.JSONDecodeError:
            return
        return

    if msg.topic == topics.presence:
        try:
            payload = json.loads(msg.payload.decode())
            presence = bool(payload.get("value") or payload.get("presence") or payload.get("detected"))
            with state_lock:
                state.presence = presence
                state.last_seen = now_iso()
            if presence:
                last_presence_at = time.time()
                if automation.lamp_enabled and automation.lamp_auto_on and state.lamp_state == 0:
                    publish(topics.lamp_control, {"state": 1, "source": "presence"})
                    with state_lock:
                        state.lamp_state = 1
        except json.JSONDecodeError:
            return


def start_mqtt() -> None:
    global mqtt_client
    client = mqtt.Client(client_id="smart_room_backend")
    client.on_connect = mqtt_on_connect
    client.on_message = mqtt_on_message
    with config_lock:
        cfg = config
    if cfg.mqtt_username:
        client.username_pw_set(cfg.mqtt_username, cfg.mqtt_password)
    if cfg.mqtt_use_ssl:
        client.tls_set()
    client.connect(cfg.mqtt_host, cfg.mqtt_port, keepalive=60)
    client.loop_start()
    mqtt_client = client


def restart_mqtt() -> None:
    global mqtt_client
    if mqtt_client is None:
        start_mqtt()
        return
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    mqtt_client = None
    start_mqtt()


def publish(topic: str, payload: dict) -> None:
    if mqtt_client is None:
        return
    mqtt_client.publish(topic, json.dumps(payload))


def automation_worker() -> None:
    global last_presence_at
    while True:
        with config_lock:
            topics = config.topics
            automation = config.automation
        with state_lock:
            presence = state.presence
            lamp_state = state.lamp_state
        if (
            automation.lamp_enabled
            and not presence
            and lamp_state == 1
            and last_presence_at is not None
        ):
            elapsed = time.time() - last_presence_at
            if elapsed >= automation.lamp_auto_off_delay:
                publish(topics.lamp_control, {"state": 0, "source": "presence"})
                with state_lock:
                    state.lamp_state = 0
        time.sleep(1)


@app.on_event("startup")
def on_startup() -> None:
    global config
    config = load_config()
    start_mqtt()
    threading.Thread(target=automation_worker, daemon=True).start()


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def get_config():
    with config_lock:
        return asdict(config)


@app.post("/api/config")
def update_config(payload: ConfigPayload):
    with config_lock:
        updated = clone_config(config)
        if payload.mqtt_host is not None:
            updated.mqtt_host = payload.mqtt_host
        if payload.mqtt_port is not None:
            updated.mqtt_port = payload.mqtt_port
        if payload.mqtt_use_ssl is not None:
            updated.mqtt_use_ssl = payload.mqtt_use_ssl
        if payload.mqtt_username is not None:
            updated.mqtt_username = payload.mqtt_username
        if payload.mqtt_password is not None:
            updated.mqtt_password = payload.mqtt_password
        if payload.topics is not None:
            topics_data = asdict(updated.topics)
            topics_data.update(payload.topics.model_dump(exclude_none=True))
            updated.topics = Topics(**topics_data)
        if payload.automation is not None:
            automation_data = asdict(updated.automation)
            automation_data.update(payload.automation.model_dump(exclude_none=True))
            updated.automation = Automation(**automation_data)
        save_config(updated)
        config = updated
    restart_mqtt()
    return {"status": "updated", "config": asdict(config)}


@app.get("/api/status")
def get_status():
    with state_lock:
        return state.model_dump()
