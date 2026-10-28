"""License plate location and approved OCR for associated vehicles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import easyocr
import numpy as np
import torch
from ultralytics import YOLO


class PlateReaderError(RuntimeError):
    """Raised when the existing plate checkpoint cannot be used."""


@dataclass(frozen=True)
class PlateResult:
    """Plate location and OCR evidence for one associated vehicle."""

    status: str
    confidence: float | None
    xyxy: tuple[int, int, int, int] | None
    explanation: str
    text: str | None = None
    ocr_confidence: float | None = None


class PlateReader:
    """Locate plates with the existing checkpoint and read them with approved EasyOCR."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        device: str | int | None = None,
        confidence: float = 0.45,
        minimum_ocr_confidence: float = 0.35,
        image_size: int = 640,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Plate detection confidence must be between 0 and 1.")
        if not 0.0 <= minimum_ocr_confidence <= 1.0:
            raise ValueError("Minimum OCR confidence must be between 0 and 1.")
        self.project_root = Path(project_root).resolve()
        self.device = device
        self.confidence = confidence
        self.minimum_ocr_confidence = minimum_ocr_confidence
        self.image_size = image_size
        self.model: YOLO | None = None
        self.ocr_reader: Any | None = None

    @property
    def checkpoint_path(self) -> Path:
        return self.project_root / "models" / "plate_detector.pt"

    @property
    def ocr_cache_dir(self) -> Path:
        return self.project_root / "models" / "easyocr_cache"

    def load(self) -> None:
        """Load the existing plate detector and approved OCR reader once."""
        if self.model is not None and self.ocr_reader is not None:
            return
        if not self.checkpoint_path.is_file():
            raise PlateReaderError(f"Required model checkpoint is missing: {self.checkpoint_path}")
        try:
            model = YOLO(str(self.checkpoint_path))
        except Exception as exc:
            raise PlateReaderError("Could not load plate_detector.pt.") from exc
        names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
        if "license_plate" not in {str(name).lower() for name in names.values()}:
            raise PlateReaderError(
                "plate_detector.pt does not expose the expected 'license_plate' class."
            )
        try:
            self.ocr_cache_dir.mkdir(parents=True, exist_ok=True)
            reader = easyocr.Reader(
                ["en"],
                gpu=torch.cuda.is_available(),
                model_storage_directory=str(self.ocr_cache_dir),
                verbose=False,
            )
        except Exception as exc:
            raise PlateReaderError("Could not load the approved EasyOCR plate reader.") from exc
        self.model = model
        self.ocr_reader = reader

    def locate(self, frame: np.ndarray, vehicle_box: tuple[int, int, int, int] | None) -> PlateResult:
        """Find the strongest plate inside one associated vehicle box.

        Returned coordinates refer to the complete video frame. Text is saved
        only when EasyOCR returns usable plate characters and confidence.
        """
        if vehicle_box is None:
            return PlateResult("not_checked", None, None, "No associated vehicle was available for plate detection.")
        if self.model is None or self.ocr_reader is None:
            raise PlateReaderError("Load the plate detector and OCR reader before locating plates.")
        if frame.size == 0:
            return PlateResult("not_detected", None, None, "The source frame is empty.")
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = self._clip_box(vehicle_box, frame_width, frame_height)
        vehicle_crop = frame[y1:y2, x1:x2]
        if vehicle_crop.size == 0:
            return PlateResult("not_detected", None, None, "The associated vehicle crop is empty.")
        try:
            result = self.model.predict(
                vehicle_crop,
                conf=self.confidence,
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
            )[0]
        except Exception as exc:
            raise PlateReaderError("Plate detector inference failed.") from exc
        if result.boxes is None or len(result.boxes) == 0:
            return PlateResult("not_detected", None, None, "No plate was detected inside the associated vehicle.")
        best_index = int(result.boxes.conf.argmax().item())
        confidence = float(result.boxes.conf[best_index].item())
        px1, py1, px2, py2 = (int(value) for value in result.boxes.xyxy[best_index].cpu().tolist())
        full_box = self._clip_box((x1 + px1, y1 + py1, x1 + px2, y1 + py2), frame_width, frame_height)
        plate_crop = vehicle_crop[max(0, py1):max(0, py2), max(0, px1):max(0, px2)]
        text, ocr_confidence = self._read_text(plate_crop)
        if text is None:
            return PlateResult(
                "detected_unreadable",
                confidence,
                full_box,
                "Plate location detected, but OCR could not read reliable plate text.",
                None,
                ocr_confidence,
            )
        return PlateResult(
            "text_read",
            confidence,
            full_box,
            "Plate location and text were detected.",
            text,
            ocr_confidence,
        )

    def _read_text(self, plate_crop: np.ndarray) -> tuple[str | None, float | None]:
        if plate_crop.size == 0:
            return None, None
        try:
            readings = self.ocr_reader.readtext(
                plate_crop,
                detail=1,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ",
                mag_ratio=2,
            )
        except Exception as exc:
            raise PlateReaderError("Plate OCR inference failed.") from exc
        usable = []
        for reading in readings:
            if len(reading) < 3:
                continue
            text = re.sub(r"[^A-Z0-9]", "", str(reading[1]).upper())
            confidence = float(reading[2])
            if len(text) >= 3:
                usable.append((text, confidence))
        if not usable:
            return None, None
        full_text = "".join(item[0] for item in usable)
        confidence = sum(item[1] for item in usable) / len(usable)
        if confidence < self.minimum_ocr_confidence:
            return None, confidence
        return full_text, confidence

    @staticmethod
    def _clip_box(
        box: tuple[int, int, int, int], width: int, height: int
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        left, right = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
        top, bottom = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
        return left, top, right, bottom
