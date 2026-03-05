from fastapi import APIRouter, Request

from app.models.schemas import ConfigPayload

router = APIRouter(prefix="/api", tags=["config"])


@router.get("/config")
def get_config(request: Request):
    app_state = request.app.state.container
    return app_state.config_dict()


@router.post("/config")
def update_config(payload: ConfigPayload, request: Request):
    app_state = request.app.state.container
    cfg = app_state.update_config(**payload.model_dump())
    app_state.activity.add("config", "config updated")
    return {"ok": True, "config": app_state.config_dict()}
