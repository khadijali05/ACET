"""Frame-level inference using the project's smoke and vehicle checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO


VEHICLE_LABELS = {"car", "motorcycle", "bus", "truck"}


class ModelLoadError(RuntimeError):
    """Raised when a configured checkpoint cannot be loaded or validated."""


@dataclass(frozen=True)
class DetectionSettings:
    smoke_confidence: float = 0.50
    vehicle_confidence: float = 0.25
    iou_threshold: float = 0.60
    image_size: int = 640

    def __post_init__(self) -> None:
        for name in ("smoke_confidence", "vehicle_confidence", "iou_threshold"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1; received {value}.")
        if self.image_size < 32:
            raise ValueError("image_size must be at least 32 pixels.")


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class FrameDetections:
    smoke: tuple[Detection, ...]
    vehicles: tuple[Detection, ...]


class DetectionPipeline:
    """Loads the existing checkpoints once and runs both detectors per frame."""

    def __init__(
        self,
        project_root: str | Path,
        settings: DetectionSettings | None = None,
        device: str | int | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.settings = settings or DetectionSettings()
        self.device = device
        self.smoke_model: YOLO | None = None
        self.vehicle_model: YOLO | None = None

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    def load(self) -> None:
        """Load and validate both local YOLO checkpoints."""
        smoke_path = self.models_dir / "smoke_detector.pt"
        vehicle_path = self.models_dir / "vehicle_detector.pt"
        for path in (smoke_path, vehicle_path):
            if not path.is_file():
                raise ModelLoadError(f"Required model checkpoint is missing: {path}")

        try:
            smoke_model = YOLO(str(smoke_path))
            vehicle_model = YOLO(str(vehicle_path))
        except Exception as exc:
            raise ModelLoadError(f"Could not load a YOLO checkpoint: {exc}") from exc

        smoke_names = self._names(smoke_model)
        if "smoke" not in {label.lower() for label in smoke_names.values()}:
            raise ModelLoadError(
                "smoke_detector.pt does not expose the expected 'smoke' class. "
                f"Checkpoint classes: {list(smoke_names.values())}"
            )

        vehicle_names = {label.lower() for label in self._names(vehicle_model).values()}
        if not (vehicle_names & VEHICLE_LABELS):
            raise ModelLoadError(
                "vehicle_detector.pt does not contain a supported vehicle class "
                f"({', '.join(sorted(VEHICLE_LABELS))})."
            )

        self.smoke_model = smoke_model
        self.vehicle_model = vehicle_model

    @staticmethod
    def _names(model: YOLO) -> dict[int, str]:
        names: Any = model.names
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        return {index: str(value) for index, value in enumerate(names)}

    def process_frame(self, frame_bgr: np.ndarray) -> FrameDetections:
        """Run smoke and vehicle detection on one OpenCV BGR frame."""
        if self.smoke_model is None or self.vehicle_model is None:
            raise RuntimeError("Detection models are not loaded. Call load() first.")
        self._validate_frame(frame_bgr)
        smoke_detections = self.detect_smoke(frame_bgr)
        vehicle_names = self._names(self.vehicle_model)
        vehicle_class_ids = [
            class_id for class_id, label in vehicle_names.items()
            if label.lower() in VEHICLE_LABELS
        ]
        vehicle_results = self.vehicle_model.predict(
            source=frame_bgr,
            conf=self.settings.vehicle_confidence,
            iou=self.settings.iou_threshold,
            imgsz=self.settings.image_size,
            classes=vehicle_class_ids,
            device=self.device,
            verbose=False,
        )

        return FrameDetections(
            smoke=smoke_detections,
            vehicles=self._convert_results(vehicle_results, vehicle_names),
        )

    def detect_smoke(self, frame_bgr: np.ndarray) -> tuple[Detection, ...]:
        """Run only the smoke detector, for use alongside tracked vehicle inference."""
        if self.smoke_model is None:
            raise RuntimeError("Smoke model is not loaded. Call load() first.")
        self._validate_frame(frame_bgr)
        results = self.smoke_model.predict(
            source=frame_bgr,
            conf=self.settings.smoke_confidence,
            iou=self.settings.iou_threshold,
            imgsz=self.settings.image_size,
            device=self.device,
            verbose=False,
        )
        return self._convert_results(results, self._names(self.smoke_model))

    @staticmethod
    def _validate_frame(frame_bgr: np.ndarray) -> None:
        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("Expected a color image as a three-channel OpenCV BGR frame.")
        if frame_bgr.shape[0] == 0 or frame_bgr.shape[1] == 0:
            raise ValueError("Cannot process an empty video frame.")

    @staticmethod
    def _convert_results(results: list[Any], names: dict[int, str]) -> tuple[Detection, ...]:
        if not results or results[0].boxes is None:
            return ()

        converted: list[Detection] = []
        for box in results[0].boxes:
            class_id = int(box.cls.item())
            xyxy = tuple(int(round(value)) for value in box.xyxy[0].tolist())
            converted.append(
                Detection(
                    label=names.get(class_id, f"class_{class_id}"),
                    confidence=float(box.conf.item()),
                    xyxy=xyxy,  # type: ignore[arg-type]
                )
            )
        return tuple(converted)
