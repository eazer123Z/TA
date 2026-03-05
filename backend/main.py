"""Compatibility shim. Use `python run.py` or `uvicorn app.main:app`."""

from app.main import app

__all__ = ["app"]
