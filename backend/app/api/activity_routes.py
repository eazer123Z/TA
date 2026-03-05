from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/activity", tags=["activity"])


@router.get("/summary")
def activity_summary(request: Request):
    app_state = request.app.state.container
    return app_state.activity.summary()
