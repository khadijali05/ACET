"""Validated runtime settings for the Smoke-Detection application."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .detection_pipeline import DetectionSettings


class SettingsError(ValueError):
    """Raised when an environment setting cannot be used safely."""


def _number(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise SettingsError(f"{name} must be a number.") from exc
    if not minimum <= value <= maximum:
        raise SettingsError(f"{name} must be between {minimum} and {maximum}.")
    return value


def _integer(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise SettingsError(f"{name} must be a whole number.") from exc
    if not minimum <= value <= maximum:
        raise SettingsError(f"{name} must be between {minimum} and {maximum}.")
    return value


@dataclass(frozen=True)
class ApplicationSettings:
    """Runtime values shared by video testing and live CCTV monitoring."""

    device: str = "cuda:0"
    smoke_confidence: float = 0.50
    vehicle_confidence: float = 0.25
    iou_threshold: float = 0.60
    image_size: int = 640
    tracker_algorithm: str = "botsort"
    analysis_fps: float = 5.0
    reconnect_seconds: float = 5.0
    open_timeout_ms: int = 5000
    read_timeout_ms: int = 5000
    event_cooldown_seconds: float = 10.0

    def __post_init__(self) -> None:
        if not self.device.strip().lower().startswith("cuda"):
            raise SettingsError("This GPU deployment requires a CUDA device such as cuda:0.")
        if self.tracker_algorithm not in {"botsort", "bytetrack"}:
            raise SettingsError("tracker_algorithm must be botsort or bytetrack.")
        if self.analysis_fps <= 0 or self.reconnect_seconds <= 0:
            raise SettingsError("analysis_fps and reconnect_seconds must be greater than zero.")
        if self.open_timeout_ms <= 0 or self.read_timeout_ms <= 0:
            raise SettingsError("Camera timeouts must be greater than zero.")
        if self.event_cooldown_seconds < 0:
            raise SettingsError("event_cooldown_seconds cannot be negative.")
        self.detection_settings()

    @classmethod
    def from_environment(cls) -> ApplicationSettings:
        """Read optional SMOKE_DETECTION_* overrides from the environment."""
        device = os.environ.get("SMOKE_DETECTION_DEVICE", "cuda:0").strip() or "cuda:0"
        tracker = os.environ.get("SMOKE_DETECTION_TRACKER", "botsort").strip().lower() or "botsort"
        return cls(
            device=device,
            smoke_confidence=_number("SMOKE_DETECTION_SMOKE_CONFIDENCE", 0.50, minimum=0.0, maximum=1.0),
            vehicle_confidence=_number("SMOKE_DETECTION_VEHICLE_CONFIDENCE", 0.25, minimum=0.0, maximum=1.0),
            iou_threshold=_number("SMOKE_DETECTION_IOU_THRESHOLD", 0.60, minimum=0.0, maximum=1.0),
            image_size=_integer("SMOKE_DETECTION_IMAGE_SIZE", 640, minimum=32, maximum=4096),
            tracker_algorithm=tracker,
            analysis_fps=_number("SMOKE_DETECTION_ANALYSIS_FPS", 5.0, minimum=0.1, maximum=120.0),
            reconnect_seconds=_number("SMOKE_DETECTION_RECONNECT_SECONDS", 5.0, minimum=0.1, maximum=3600.0),
            open_timeout_ms=_integer("SMOKE_DETECTION_OPEN_TIMEOUT_MS", 5000, minimum=100, maximum=120000),
            read_timeout_ms=_integer("SMOKE_DETECTION_READ_TIMEOUT_MS", 5000, minimum=100, maximum=120000),
            event_cooldown_seconds=_number("SMOKE_DETECTION_EVENT_COOLDOWN_SECONDS", 10.0, minimum=0.0, maximum=3600.0),
        )

    def detection_settings(self) -> DetectionSettings:
        """Convert the shared detection values to the pipeline's settings type."""
        return DetectionSettings(
            smoke_confidence=self.smoke_confidence,
            vehicle_confidence=self.vehicle_confidence,
            iou_threshold=self.iou_threshold,
            image_size=self.image_size,
        )
