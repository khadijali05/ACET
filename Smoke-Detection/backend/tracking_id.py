"""Per-source vehicle tracking with a shared YOLO detector."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
from ultralytics import YOLO
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker


SUPPORTED_TRACKERS = {"bytetrack", "botsort"}
VEHICLE_LABELS = {"car", "motorcycle", "bus", "truck"}


@dataclass(frozen=True)
class TrackedVehicle:
    track_id: int
    label: str
    confidence: float
    xyxy: tuple[int, int, int, int]


class VehicleTracker:
    """Use one shared YOLO checkpoint with independent state for each video source."""

    def __init__(
        self,
        vehicle_model: YOLO,
        tracker: str = "botsort",
        confidence: float = 0.15,
        iou_threshold: float = 0.45,
        device: str | int | None = None,
    ) -> None:
        tracker_key = tracker.lower().replace("-", "")
        if tracker_key not in SUPPORTED_TRACKERS:
            raise ValueError("tracker must be 'bytetrack' or 'botsort'.")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1.")

        self.vehicle_model = vehicle_model
        self.tracker_name = tracker_key
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = device
        self.class_names = self._names(vehicle_model)
        self.vehicle_class_ids = [
            class_id for class_id, label in self.class_names.items()
            if label.lower() in VEHICLE_LABELS
        ]
        if not self.vehicle_class_ids:
            raise ValueError("The supplied checkpoint has no supported vehicle classes.")
        self._tracker_args = self._make_tracker_args()
        self._tracker = self._new_tracker()

    def _make_tracker_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            track_high_thresh=max(0.5, self.confidence),
            track_low_thresh=self.confidence,
            new_track_thresh=max(0.5, self.confidence),
            track_buffer=90,
            match_thresh=0.8,
            fuse_score=True,
            gmc_method="sparseOptFlow",
            proximity_thresh=0.5,
            appearance_thresh=0.8,
            with_reid=False,
            model="auto",
            device=self.device,
        )

    def _new_tracker(self) -> BYTETracker | BOTSORT:
        if self.tracker_name == "bytetrack":
            return BYTETracker(self._tracker_args)
        return BOTSORT(self._tracker_args)

    @staticmethod
    def _names(model: YOLO) -> dict[int, str]:
        names: Any = model.names
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        return {index: str(value) for index, value in enumerate(names)}

    def update(self, frame_bgr: np.ndarray) -> tuple[TrackedVehicle, ...]:
        """Detect vehicles once and update this source's independent tracker state."""
        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("Expected a color image as a three-channel OpenCV BGR frame.")
        if frame_bgr.shape[0] == 0 or frame_bgr.shape[1] == 0:
            raise ValueError("Cannot track an empty video frame.")

        detections = self.vehicle_model.predict(
            source=frame_bgr,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=self.vehicle_class_ids,
            device=self.device,
            verbose=False,
        )
        boxes = detections[0].boxes if detections else None
        if boxes is None:
            return ()
        tracks = self._tracker.update(boxes.cpu(), img=frame_bgr)
        if tracks is None or len(tracks) == 0:
            return ()

        tracked: list[TrackedVehicle] = []
        for row in tracks:
            class_id = int(row[6])
            coords = tuple(int(round(value)) for value in row[:4])
            tracked.append(
                TrackedVehicle(
                    track_id=int(row[4]),
                    label=self.class_names.get(class_id, f"class_{class_id}"),
                    confidence=float(row[5]),
                    xyxy=coords,  # type: ignore[arg-type]
                )
            )
        return tuple(tracked)

    def reset(self) -> None:
        """Clear this source's tracking history before rerunning a video."""
        self._tracker = self._new_tracker()
