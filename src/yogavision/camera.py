"""Webcam capture with explicit open/release lifecycle."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .utils import get_logger

logger = get_logger()


class CameraError(RuntimeError):
    """Raised when a camera cannot be opened."""


def list_cameras(max_index: int = 6) -> list[int]:
    """Probe camera indices and return those that can be opened."""
    import cv2

    available: list[int] = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            available.append(idx)
            cap.release()
    return available


@dataclass
class FrameMeta:
    index: int
    width: int
    height: int


class CameraCapture:
    """Context-manager wrapper around ``cv2.VideoCapture``."""

    def __init__(
        self,
        index: int = 0,
        *,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
    ) -> None:
        self.index = index
        self.requested_width = width
        self.requested_height = height
        self.requested_fps = fps
        self._cap = None

    def open(self) -> "CameraCapture":
        import cv2

        self._cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            raise CameraError(
                f"Could not open camera index {self.index}. Check that a webcam is "
                "connected and that Windows camera privacy settings allow access "
                "for this application (Settings > Privacy & Security > Camera)."
            )
        if self.requested_width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        if self.requested_height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
        if self.requested_fps:
            self._cap.set(cv2.CAP_PROP_FPS, self.requested_fps)
        logger.info("Camera %s opened", self.index)
        return self

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._cap is None:
            raise CameraError("Camera is not open")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None
        return True, frame

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera %s released", self.index)

    def __enter__(self) -> "CameraCapture":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.release()


def run_capture_loop(
    index: int,
    frame_handler: Callable[[np.ndarray, int], bool],
    *,
    max_frames: int | None = None,
    target_fps: float = 30.0,
) -> int:
    """Run a blocking capture loop until the handler returns False.

    ``frame_handler`` receives (frame, frame_index) and returns False to stop.
    Returns the number of frames processed. This is the fallback path used when
    the UI framework cannot stream frames directly.
    """
    import time

    processed = 0
    min_dt = 1.0 / target_fps if target_fps > 0 else 0.0
    with CameraCapture(index) as cam:
        while True:
            ok, frame = cam.read()
            if not ok:
                logger.warning("Frame grab failed, stopping capture")
                break
            keep_going = frame_handler(frame, processed)
            processed += 1
            if not keep_going:
                break
            if max_frames is not None and processed >= max_frames:
                break
            if min_dt:
                time.sleep(min_dt)
    return processed
