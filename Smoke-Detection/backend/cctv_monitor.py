"""Dynamic CCTV stream workers using shared models and per-camera tracker state."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import cv2
import numpy as np

from .camera_manager import CameraConfig, CameraManager
from .color_classifier import ColorResult, SmokeColorClassifier
from .detection_pipeline import DetectionPipeline
from .finding_error import get_logger
from .plate_reader import PlateReader, PlateResult
from .smoke_vehicle_match import SmokeVehicleMatcher
from .tracking_id import TrackedVehicle, VehicleTracker


LOGGER = get_logger("cctv")


@dataclass(frozen=True)
class CCTVDetectionEvent:
    camera_id: str
    camera_name: str
    timestamp_utc: str
    frame_index: int
    smoke_box: tuple[int, int, int, int]
    smoke_detector_score: float
    colour: ColorResult
    association_status: str
    vehicle_track_id: int | None
    candidate_vehicle_track_id: int | None
    association_score: float
    plate: PlateResult


@dataclass(frozen=True)
class CameraViewState:
    camera: CameraConfig
    status: str
    latest_frame_jpeg: bytes | None
    message: str


@dataclass
class _Worker:
    stop_event: threading.Event
    thread: threading.Thread


class CCTVMonitor:
    """Start one capture worker per enabled camera and share loaded AI models."""

    def __init__(
        self,
        camera_manager: CameraManager,
        detection_pipeline: DetectionPipeline,
        color_classifier: SmokeColorClassifier,
        plate_reader: PlateReader,
        *,
        tracker_algorithm: str = "botsort",
        analysis_fps: float = 5.0,
        reconnect_seconds: float = 5.0,
        open_timeout_ms: int = 5000,
        read_timeout_ms: int = 5000,
        event_cooldown_seconds: float = 10.0,
        event_callback: Callable[[CCTVDetectionEvent], None] | None = None,
    ) -> None:
        if analysis_fps <= 0:
            raise ValueError("analysis_fps must be greater than zero.")
        if reconnect_seconds <= 0:
            raise ValueError("reconnect_seconds must be greater than zero.")
        if open_timeout_ms <= 0 or read_timeout_ms <= 0:
            raise ValueError("Camera stream timeouts must be greater than zero.")
        if event_cooldown_seconds < 0:
            raise ValueError("event_cooldown_seconds cannot be negative.")
        if detection_pipeline.smoke_model is None or detection_pipeline.vehicle_model is None:
            raise RuntimeError("Load the detection pipeline before starting CCTV monitoring.")

        self.camera_manager = camera_manager
        self.detection_pipeline = detection_pipeline
        self.color_classifier = color_classifier
        self.plate_reader = plate_reader
        self.tracker_algorithm = tracker_algorithm
        self.analysis_fps = analysis_fps
        self.reconnect_seconds = reconnect_seconds
        self.open_timeout_ms = open_timeout_ms
        self.read_timeout_ms = read_timeout_ms
        self.event_cooldown_seconds = event_cooldown_seconds
        self.event_callback = event_callback
        self._workers: dict[str, _Worker] = {}
        self._latest_frames: dict[str, bytes] = {}
        self._messages: dict[str, str] = {}
        self._lock = threading.RLock()
        self._inference_lock = threading.Lock()
        self._closed = False

    def refresh(self) -> tuple[CameraViewState, ...]:
        """Start workers for new enabled cameras and stop removed or disabled ones."""
        if self._closed:
            raise RuntimeError("CCTV monitor is stopped and cannot be refreshed.")
        cameras = self.camera_manager.list_cameras()
        configured = {camera.camera_id: camera for camera in cameras}
        desired = {camera.camera_id for camera in cameras if camera.enabled}

        with self._lock:
            to_stop = [camera_id for camera_id in self._workers if camera_id not in desired]
        for camera_id in to_stop:
            self._stop_worker(camera_id)

        with self._lock:
            for camera_id in desired:
                worker = self._workers.get(camera_id)
                if worker is None or not worker.thread.is_alive():
                    stop_event = threading.Event()
                    thread = threading.Thread(
                        target=self._run_camera,
                        args=(camera_id, stop_event),
                        name=f"camera-{camera_id[:8]}",
                        daemon=True,
                    )
                    self._workers[camera_id] = _Worker(stop_event, thread)
                    thread.start()

            views = []
            for camera in cameras:
                views.append(
                    CameraViewState(
                        camera=camera,
                        status=camera.status,
                        latest_frame_jpeg=self._latest_frames.get(camera.camera_id),
                        message=self._messages.get(camera.camera_id, ""),
                    )
                )
        return tuple(views)

    def current_views(self) -> tuple[CameraViewState, ...]:
        """Return current camera configurations and in-memory frames without starting workers."""
        cameras = self.camera_manager.list_cameras()
        with self._lock:
            return tuple(
                CameraViewState(
                    camera=camera,
                    status=camera.status,
                    latest_frame_jpeg=self._latest_frames.get(camera.camera_id),
                    message=self._messages.get(camera.camera_id, ""),
                )
                for camera in cameras
            )

    def _run_camera(self, camera_id: str, stop_event: threading.Event) -> None:
        tracker = VehicleTracker(
            self.detection_pipeline.vehicle_model,
            tracker=self.tracker_algorithm,
            device=self.detection_pipeline.device,
        )
        matcher = SmokeVehicleMatcher()
        frame_count = 0
        analyzed_count = 0
        classifier_unavailable = False
        last_event_time: dict[int, float] = {}

        while not stop_event.is_set():
            capture = None
            try:
                camera = self.camera_manager.get_camera(camera_id)
                if not camera.enabled:
                    self.camera_manager.set_status(camera_id, "disabled")
                    return
                self.camera_manager.set_status(camera_id, "connecting")
                stream_url = self.camera_manager._stream_url_for_worker(camera_id)
                capture = self._open_capture(stream_url)
                if not capture.isOpened():
                    self._set_runtime(camera_id, "stream_unavailable", "Stream is unavailable; retrying.")
                    stop_event.wait(self.reconnect_seconds)
                    continue

                fps = float(capture.get(cv2.CAP_PROP_FPS))
                if not np.isfinite(fps) or fps <= 0:
                    fps = 25.0
                stride = max(1, int(round(fps / self.analysis_fps)))
                connected_marked = False
                last_smoke = ()
                last_vehicles = ()
                while not stop_event.is_set():
                    ok, frame = capture.read()
                    if not ok:
                        self._set_runtime(camera_id, "disconnected", "Stream stopped returning frames; reconnecting.")
                        break
                    if not connected_marked:
                        self.camera_manager.set_status(camera_id, "connected")
                        connected_marked = True
                    if frame_count % stride == 0:
                        with self._inference_lock:
                            last_smoke = self.detection_pipeline.detect_smoke(frame)
                            last_vehicles = tracker.update(frame)
                            matches = matcher.update(analyzed_count, last_smoke, last_vehicles)
                            analyzed_count += 1
                            event_items = self._make_events(
                                camera,
                                frame_count,
                                frame,
                                matches,
                                last_vehicles,
                                classifier_unavailable,
                            )
                            if any(item.colour.status == "unavailable" for item in event_items):
                                classifier_unavailable = True
                        for event in event_items:
                            track_id = event.vehicle_track_id
                            now = time.monotonic()
                            if track_id is not None:
                                previous = last_event_time.get(track_id, float("-inf"))
                                if now - previous < self.event_cooldown_seconds:
                                    continue
                                last_event_time[track_id] = now
                            if self.event_callback is not None:
                                self.event_callback(event)
                    frame = self._draw_frame(frame, last_vehicles, last_smoke)

                    success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                    if success:
                        with self._lock:
                            self._latest_frames[camera_id] = encoded.tobytes()
                            self._messages[camera_id] = ""
                    frame_count += 1

                if stop_event.is_set():
                    return
                if not stop_event.is_set():
                    stop_event.wait(self.reconnect_seconds)
            except Exception as exc:
                # Do not log exception text because third-party stream errors can contain URLs.
                LOGGER.error(
                    "Camera processing failed: camera_id=%s error_type=%s",
                    camera_id,
                    type(exc).__name__,
                )
                self._set_runtime(
                    camera_id,
                    "processing_error",
                    f"Camera processing failed ({type(exc).__name__}); retrying.",
                )
                stop_event.wait(self.reconnect_seconds)
            finally:
                if capture is not None:
                    capture.release()

        self._set_runtime(camera_id, "disconnected", "Monitoring stopped.")

    def _make_events(
        self,
        camera: CameraConfig,
        frame_index: int,
        frame: np.ndarray,
        matches,
        vehicles: tuple[TrackedVehicle, ...],
        unavailable: bool,
    ):
        events: list[CCTVDetectionEvent] = []
        height, width = frame.shape[:2]
        for match in matches:
            if unavailable:
                colour = ColorResult(
                    None,
                    "unavailable",
                    0.0,
                    "pixel_share_heuristic",
                    None,
                    {},
                    "Classifier unavailable earlier for this camera stream.",
                )
            else:
                x1, y1, x2, y2 = match.smoke.xyxy
                crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                try:
                    colour = self.color_classifier.classify(crop)
                except Exception:
                    colour = ColorResult(
                        None,
                        "unavailable",
                        0.0,
                        "pixel_share_heuristic",
                        None,
                        {},
                        "Colour classifier could not run; no colour was assigned.",
                    )
            associated_vehicle = next(
                (vehicle for vehicle in vehicles if vehicle.track_id == match.vehicle_track_id),
                None,
            )
            try:
                plate = self.plate_reader.locate(
                    frame,
                    associated_vehicle.xyxy if associated_vehicle is not None else None,
                )
            except Exception as exc:
                LOGGER.error("Plate detection failed: error_type=%s", type(exc).__name__)
                plate = PlateResult("unavailable", None, None, "Plate detector could not run for this event.")
            events.append(
                CCTVDetectionEvent(
                    camera_id=camera.camera_id,
                    camera_name=camera.display_name,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    frame_index=frame_index,
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
        return events

    @staticmethod
    def _draw_frame(frame: np.ndarray, vehicles: tuple[TrackedVehicle, ...], smoke) -> np.ndarray:
        output = frame.copy()
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle.xyxy
            cv2.rectangle(output, (x1, y1), (x2, y2), (220, 150, 40), 2)
            cv2.putText(
                output,
                f"{vehicle.label} | ID {vehicle.track_id}",
                (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 150, 40),
                2,
                cv2.LINE_AA,
            )
        for detection in smoke:
            x1, y1, x2, y2 = detection.xyxy
            cv2.rectangle(output, (x1, y1), (x2, y2), (40, 60, 230), 2)
            cv2.putText(
                output,
                f"Smoke {detection.confidence:.2f}",
                (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (40, 60, 230),
                2,
                cv2.LINE_AA,
            )
        return output

    def _set_runtime(self, camera_id: str, status: str, message: str) -> None:
        try:
            self.camera_manager.set_status(camera_id, status)
        except KeyError:
            pass
        with self._lock:
            self._messages[camera_id] = message
            if status in {"disconnected", "disabled", "stream_unavailable"}:
                self._latest_frames.pop(camera_id, None)

    def _open_capture(self, stream_url: str):
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.open_timeout_ms,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.read_timeout_ms,
        ]
        return cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG, params)

    def _stop_worker(self, camera_id: str) -> None:
        with self._lock:
            worker = self._workers.pop(camera_id, None)
        if worker is None:
            return
        worker.stop_event.set()
        if worker.thread.is_alive():
            timeout = max(6.0, (self.open_timeout_ms + self.read_timeout_ms) / 1000 + 1.0)
            worker.thread.join(timeout=timeout)
        with self._lock:
            self._latest_frames.pop(camera_id, None)
            self._messages.pop(camera_id, None)

    def stop(self) -> None:
        """Stop all camera threads and release in-memory frames."""
        self._closed = True
        with self._lock:
            camera_ids = list(self._workers)
        for camera_id in camera_ids:
            self._stop_worker(camera_id)

    def __enter__(self) -> CCTVMonitor:
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
