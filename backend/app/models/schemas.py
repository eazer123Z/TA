from pydantic import BaseModel


class ConfigPayload(BaseModel):
    mqtt_host: str | None = None
    mqtt_port: int | None = None
    camera_index: int | None = None
    vision_model: str | None = None


class ActivityItem(BaseModel):
    event_type: str
    summary: str
    count: int
