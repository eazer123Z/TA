from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.activity_routes import router as activity_router
from app.api.config_routes import router as config_router
from app.api.system_routes import router as system_router
from app.core.app_state import AppState


def create_app() -> FastAPI:
    app = FastAPI(title="IoTzy Modular Backend")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    container = AppState()
    app.state.container = container

    @app.on_event("startup")
    def _startup() -> None:
        container.startup()

    @app.on_event("shutdown")
    def _shutdown() -> None:
        container.shutdown()

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    app.include_router(system_router)
    app.include_router(config_router)
    app.include_router(activity_router)
    return app


app = create_app()
