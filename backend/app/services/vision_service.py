from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None


@dataclass
class VisionStatus:
    running: bool
    model: str


class VisionService:
    def __init__(self, camera_index: int = 0, model_name: str = "yolov8m.pt") -> None:
        self.camera_index = camera_index
        self.model_name = model_name
        self._model = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self.running = False

    def _ensure_model(self) -> None:
        if self._model is None and YOLO is not None:
            self._model = YOLO(self.model_name)

    def start(self) -> None:
        if self.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self.running = True

    def stop(self) -> None:
        if not self.running:
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)
        self.running = False

    def _worker(self) -> None:
        self._ensure_model()
        cap = cv2.VideoCapture(self.camera_index)
        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.2)
                continue
            if self._model is not None:
                _ = self._model(frame, verbose=False)
            time.sleep(0.1)
        cap.release()

    def status(self) -> VisionStatus:
        return VisionStatus(running=self.running, model=self.model_name)
