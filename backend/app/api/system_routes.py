from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/system", tags=["system"])


@router.post("/start")
def start_system(request: Request):
    app_state = request.app.state.container
    app_state.vision.start()
    app_state.activity.add("vision", "vision started")
    return {"ok": True}


@router.post("/stop")
def stop_system(request: Request):
    app_state = request.app.state.container
    app_state.vision.stop()
    app_state.activity.add("vision", "vision stopped")
    return {"ok": True}


@router.get("/status")
def system_status(request: Request):
    app_state = request.app.state.container
    return {
        "vision": {
            "running": app_state.vision.status().running,
            "model": app_state.vision.status().model,
        },
        "mqtt": {"connected": app_state.mqtt.connected},
    }
