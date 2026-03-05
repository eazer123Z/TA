from __future__ import annotations

from dataclasses import asdict

from app.core.settings import AppConfig, load_config, save_config
from app.services.activity_service import ActivityService
from app.services.automation_service import AutomationService
from app.services.mqtt_service import MqttConfig, MqttService
from app.services.vision_service import VisionService


class AppState:
    def __init__(self) -> None:
        self.config: AppConfig = load_config()
        self.activity = ActivityService()
        self.automation = AutomationService()
        self.mqtt = MqttService()
        self.vision = VisionService(camera_index=self.config.camera_index, model_name=self.config.vision_model)

    def startup(self) -> None:
        self.mqtt.connect(MqttConfig(host=self.config.mqtt_host, port=self.config.mqtt_port))
        self.activity.add("system", "backend started")

    def shutdown(self) -> None:
        self.vision.stop()
        self.mqtt.disconnect()
        self.activity.add("system", "backend stopped")

    def update_config(self, **kwargs) -> AppConfig:
        for k, v in kwargs.items():
            if v is not None and hasattr(self.config, k):
                setattr(self.config, k, v)
        save_config(self.config)
        if "camera_index" in kwargs or "vision_model" in kwargs:
            was_running = self.vision.running
            self.vision.stop()
            self.vision = VisionService(camera_index=self.config.camera_index, model_name=self.config.vision_model)
            if was_running:
                self.vision.start()
        if "mqtt_host" in kwargs or "mqtt_port" in kwargs:
            self.mqtt.connect(MqttConfig(host=self.config.mqtt_host, port=self.config.mqtt_port))
        return self.config

    def config_dict(self) -> dict:
        return asdict(self.config)
