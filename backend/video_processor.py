"""Temporary, rerunnable video testing workflow."""

from __future__ import annotations

import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .color_classifier import ColorResult, SmokeColorClassifier
from .detection_pipeline import Detection, DetectionPipeline
from .plate_reader import PlateReader, PlateResult
from .smoke_vehicle_match import SmokeVehicleMatcher
from .tracking_id import VehicleTracker


@dataclass(frozen=True)
class VideoProgress:
    frames_read: int
    total_frames: int
    progress: float | None
    current_seconds: float


@dataclass(frozen=True)
class VideoEvent:
    frame_index: int
    timestamp_seconds: float
    smoke_box: tuple[int, int, int, int]
    smoke_detector_score: float
    colour: ColorResult
    association_status: str
    vehicle_track_id: int | None
    candidate_vehicle_track_id: int | None
    association_score: float
    plate: PlateResult


@dataclass(frozen=True)
class VideoProcessingResult:
    events: tuple[VideoEvent, ...]
    frames_read: int
    frames_analyzed: int
    fps: float
    stopped_early: bool
    output_path: str
    issues: tuple[str, ...]


class VideoSession:
    """A temporary session that retains source/results for reruns until close()."""

    ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

    def __init__(self, video_bytes: bytes, original_name: str) -> None:
        if not video_bytes:
            raise ValueError("The uploaded video is empty.")
        suffix = Path(original_name).suffix.lower()
        if suffix not in self.ALLOWED_EXTENSIONS:
            raise ValueError("Supported video formats are MP4, AVI, MOV, and MKV.")

        self._temporary_directory = tempfile.TemporaryDirectory(prefix="smoke-video-")
        self.directory = Path(self._temporary_directory.name)
        self.source_path = self.directory / f"source{suffix}"
        self.output_path = self.directory / "processed.mp4"
        self.source_path.write_bytes(video_bytes)
        self.closed = False

    def process(
        self,
        pipeline: DetectionPipeline,
        tracker: VehicleTracker,
        matcher: SmokeVehicleMatcher,
        classifier: SmokeColorClassifier,
        plate_reader: PlateReader,
        *,
        analysis_fps: float = 5.0,
        progress_callback: Callable[[VideoProgress], None] | None = None,
        stop_event: threading.Event | None = None,
    ) -> VideoProcessingResult:
        """Process the current temporary source; the same session can be rerun."""
        if self.closed:
            raise RuntimeError("This video session has ended. Upload the video again to rerun it.")
        if analysis_fps <= 0:
            raise ValueError("analysis_fps must be greater than zero.")
        if pipeline.smoke_model is None or pipeline.vehicle_model is None:
            raise RuntimeError("Detection models must be loaded before video processing.")

        self.output_path.unlink(missing_ok=True)
        tracker.reset()
        matcher.reset()

        capture = cv2.VideoCapture(str(self.source_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError("Could not open the uploaded video. Check its format and encoding.")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if width <= 0 or height <= 0:
            capture.release()
            raise RuntimeError("Could not read the uploaded video's frame dimensions.")

        frame_stride = max(1, int(round(fps / analysis_fps)))
        writer = cv2.VideoWriter(
            str(self.output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            self.output_path.unlink(missing_ok=True)
            raise RuntimeError("Could not create the temporary processed video output.")

        events: list[VideoEvent] = []
        issues: list[str] = []
        frames_read = 0
        frames_analyzed = 0
        last_smoke: tuple[Detection, ...] = ()
        last_vehicles = ()
        classifier_state: str = "not_loaded"
        stopped_early = False

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = frames_read / fps
                should_analyze = frames_read % frame_stride == 0

                if should_analyze:
                    last_smoke = pipeline.detect_smoke(frame)
                    last_vehicles = tracker.update(frame)
                    matches = matcher.update(frames_analyzed, last_smoke, last_vehicles)
                    frames_analyzed += 1
                    for match in matches:
                        x1, y1, x2, y2 = match.smoke.xyxy
                        crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                        if classifier_state != "unavailable":
                            try:
                                colour = classifier.classify(crop)
                                classifier_state = "loaded"
                            except Exception as exc:
                                classifier_state = "unavailable"
                                message = f"Smoke colour classification unavailable: {exc}"
                                if message not in issues:
                                    issues.append(message)
                                colour = ColorResult(
                                    None,
                                    "unavailable",
                                    0.0,
                                    "pixel_share_heuristic",
                                    None,
                                    {},
                                    "Classifier could not run; no colour was assigned.",
                                )
                        else:
                            colour = ColorResult(
                                None,
                                "unavailable",
                                0.0,
                                "pixel_share_heuristic",
                                None,
                                {},
                                "Classifier unavailable earlier in this processing run.",
                            )
                        associated_vehicle = next(
                            (vehicle for vehicle in last_vehicles if vehicle.track_id == match.vehicle_track_id),
                            None,
                        )
                        try:
                            plate = plate_reader.locate(
                                frame,
                                associated_vehicle.xyxy if associated_vehicle is not None else None,
                            )
                        except Exception as exc:
                            message = f"Plate detection unavailable: {type(exc).__name__}"
                            if message not in issues:
                                issues.append(message)
                            plate = PlateResult(
                                "unavailable",
                                None,
                                None,
                                "Plate detector could not run for this video event.",
                            )
                        events.append(
                            VideoEvent(
                                frame_index=frames_read,
                                timestamp_seconds=timestamp,
                                smoke_box=match.smoke.xyxy,
                                smoke_detector_score=match.smoke.confidence,
                                colour=colour,
                                association_status=match.status,
                                vehicle_track_id=match.vehicle_track_id,
                                candidate_vehicle_track_id=match.candidate_track_id,
                                association_score=match.association_score,
                                plate=plate,
                            )
                        )

                annotated = frame.copy()
                for vehicle in last_vehicles:
                    x1, y1, x2, y2 = vehicle.xyxy
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (220, 150, 40), 2)
                    cv2.putText(
                        annotated,
                        f"{vehicle.label} | ID {vehicle.track_id}",
                        (x1, max(18, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (220, 150, 40),
                        2,
                        cv2.LINE_AA,
                    )
                for smoke in last_smoke:
                    x1, y1, x2, y2 = smoke.xyxy
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (40, 60, 230), 2)
                    cv2.putText(
                        annotated,
                        f"Smoke {smoke.confidence:.2f}",
                        (x1, max(18, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (40, 60, 230),
                        2,
                        cv2.LINE_AA,
                    )
                writer.write(annotated)
                frames_read += 1

                if progress_callback is not None:
                    progress_callback(
                        VideoProgress(
                            frames_read=frames_read,
                            total_frames=total_frames,
                            progress=(frames_read / total_frames if total_frames > 0 else None),
                            current_seconds=timestamp,
                        )
                    )
                if stop_event is not None and stop_event.is_set():
                    stopped_early = True
                    break
        finally:
            capture.release()
            writer.release()

        return VideoProcessingResult(
            events=tuple(events),
            frames_read=frames_read,
            frames_analyzed=frames_analyzed,
            fps=fps,
            stopped_early=stopped_early,
            output_path=str(self.output_path),
            issues=tuple(issues),
        )

    def close(self) -> None:
        """Delete the uploaded video and processed copy from temporary storage."""
        if not self.closed:
            self.closed = True
            self._temporary_directory.cleanup()

    def __enter__(self) -> VideoSession:
        if self.closed:
            raise RuntimeError("This video session has already ended.")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
