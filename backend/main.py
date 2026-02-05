from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import paho.mqtt.client as mqtt
from fastapi import FastAPI
from pydantic import BaseModel

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


@dataclass
class Topics:
    temperature: str = "iotzy/sensor/temperature"
    presence: str = "iotzy/sensor/presence"
    brightness: str = "iotzy/sensor/brightness"
    lamp_control: str = "iotzy/device/lamp/control"


@dataclass
class Automation:
    lamp_enabled: bool = True
    lamp_on_threshold: float = 0.35
    lamp_off_threshold: float = 0.5


@dataclass
class BackendConfig:
    mqtt_host: str = "broker.hivemq.com"
    mqtt_port: int = 1883
    mqtt_use_ssl: bool = False
    mqtt_username: str = ""
    mqtt_password: str = ""
    camera_index: int = 0
    topics: Topics = Topics()
    automation: Automation = Automation()


class ConfigPayload(BaseModel):
    mqtt_host: Optional[str] = None
    mqtt_port: Optional[int] = None
    mqtt_use_ssl: Optional[bool] = None
    mqtt_username: Optional[str] = None
    mqtt_password: Optional[str] = None
    camera_index: Optional[int] = None
    topics: Optional[Topics] = None
    automation: Optional[Automation] = None


class SensorState(BaseModel):
    temperature: Optional[float] = None
    presence: bool = False
    brightness: Optional[float] = None
    last_seen: Optional[str] = None
    lamp_state: int = 0


app = FastAPI(title="IoTzy Backend")
config_lock = threading.Lock()
state_lock = threading.Lock()


config = BackendConfig()
state = SensorState()

mqtt_client: Optional[mqtt.Client] = None
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


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
        camera_index=data.get("camera_index", 0),
        topics=topics,
        automation=automation,
    )


def save_config(new_config: BackendConfig) -> None:
    CONFIG_PATH.write_text(json.dumps(asdict(new_config), indent=2))


def mqtt_on_connect(client: mqtt.Client, _userdata, _flags, _rc):
    with config_lock:
        topic_temp = config.topics.temperature
    client.subscribe(topic_temp)


def mqtt_on_message(_client: mqtt.Client, _userdata, msg: mqtt.MQTTMessage):
    with config_lock:
        topic_temp = config.topics.temperature
    if msg.topic != topic_temp:
        return
    try:
        payload = json.loads(msg.payload.decode())
        temp = payload.get("value") or payload.get("temperature") or payload.get("temp")
        if temp is None:
            return
        with state_lock:
            state.temperature = float(temp)
    except json.JSONDecodeError:
        return


def start_mqtt() -> None:
    global mqtt_client
    client = mqtt.Client(client_id="iotzy_backend")
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


def publish(topic: str, payload: dict) -> None:
    if mqtt_client is None:
        return
    mqtt_client.publish(topic, json.dumps(payload))


def camera_worker() -> None:
    last_presence = None
    last_brightness = None
    last_lamp_action = "off"
    while True:
        with config_lock:
            camera_index = config.camera_index
            automation = config.automation
            topics = config.topics
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            time.sleep(3)
            continue
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(gray.mean() / 255.0)
            (rects, _weights) = hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)
            presence = len(rects) > 0
            if brightness != last_brightness:
                publish(topics.brightness, {"value": round(brightness, 3)})
                last_brightness = brightness
            if presence != last_presence:
                publish(topics.presence, {"value": 1 if presence else 0})
                last_presence = presence
            with state_lock:
                state.brightness = brightness
                state.presence = presence
                state.last_seen = time.strftime("%Y-%m-%d %H:%M:%S")
            if automation.lamp_enabled:
                if brightness < automation.lamp_on_threshold and last_lamp_action != "on":
                    publish(topics.lamp_control, {"state": 1, "source": "camera"})
                    last_lamp_action = "on"
                    with state_lock:
                        state.lamp_state = 1
                elif brightness > automation.lamp_off_threshold and last_lamp_action != "off":
                    publish(topics.lamp_control, {"state": 0, "source": "camera"})
                    last_lamp_action = "off"
                    with state_lock:
                        state.lamp_state = 0
            time.sleep(0.2)
        cap.release()
        time.sleep(1)


@app.on_event("startup")
def on_startup() -> None:
    global config
    config = load_config()
    start_mqtt()
    threading.Thread(target=camera_worker, daemon=True).start()


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
        updated = BackendConfig(**asdict(config))
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
        if payload.camera_index is not None:
            updated.camera_index = payload.camera_index
        if payload.topics is not None:
            updated.topics = payload.topics
        if payload.automation is not None:
            updated.automation = payload.automation
        save_config(updated)
        config = updated
    return {"status": "updated", "config": asdict(config)}


@app.get("/api/status")
def get_status():
    with state_lock:
        return state.model_dump()
