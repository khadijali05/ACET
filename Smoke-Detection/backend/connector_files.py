"""Connect the project's Python services into one dashboard-ready object."""

from __future__ import annotations

from pathlib import Path

from .authentication import AdministratorAuthenticator
from .camera_manager import CameraManager
from .cctv_monitor import CCTVMonitor
from .color_classifier import SmokeColorClassifier
from .detection_pipeline import DetectionPipeline, DetectionSettings
from .email_alert import EmailAlertService
from .event_store import EventStore
from .plate_reader import PlateReader
from .pre_checks import PreChecks
from .report_generator import ReportGenerator
from .settings import ApplicationSettings


class ApplicationServices:
    """Create and connect shared services without loading AI models at import time.

    Databases are kept in the project's ``data`` folder. Call ``start_monitor``
    only when live CCTV monitoring is needed; that method loads the existing
    model checkpoints and starts workers for configured cameras.
    """

    def __init__(
        self,
        project_root: str | Path,
        *,
        runtime_settings: ApplicationSettings | None = None,
        detection_settings: DetectionSettings | None = None,
        device: str | int | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.data_dir = self.project_root / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_settings = runtime_settings or ApplicationSettings.from_environment()
        effective_device = device if device is not None else self.runtime_settings.device

        self.camera_manager = CameraManager(self.data_dir / "cameras.sqlite3")
        self.event_store = EventStore(self.data_dir / "events.sqlite3")
        self.email_alerts = EmailAlertService(self.data_dir / "email_alerts.sqlite3")
        self.report_generator = ReportGenerator(self.event_store)
        self.detection_pipeline = DetectionPipeline(
            self.project_root,
            settings=detection_settings or self.runtime_settings.detection_settings(),
            device=effective_device,
        )
        classifier_device = f"cuda:{effective_device}" if isinstance(effective_device, int) else effective_device
        self.color_classifier = SmokeColorClassifier(self.project_root, device=classifier_device)
        self.plate_reader = PlateReader(self.project_root, device=effective_device)
        self.monitor: CCTVMonitor | None = None

    @staticmethod
    def authenticate_admin(username: str, password: str) -> bool:
        """Check dashboard credentials configured through environment variables."""
        authenticator = AdministratorAuthenticator.from_environment()
        return authenticator.authenticate(username, password).authenticated

    def start_monitor(self, **monitor_options: object) -> CCTVMonitor:
        """Load the existing checkpoints and connect live events to SQLite."""
        if self.monitor is not None:
            return self.monitor
        report = PreChecks(self.project_root, self.runtime_settings, self.camera_manager).run()
        failures = [check.message for check in report.checks if check.status == "error"]
        if failures:
            raise RuntimeError("Live-monitoring readiness checks failed: " + " ".join(failures))
        default_monitor_options = {
            "tracker_algorithm": self.runtime_settings.tracker_algorithm,
            "analysis_fps": self.runtime_settings.analysis_fps,
            "reconnect_seconds": self.runtime_settings.reconnect_seconds,
            "open_timeout_ms": self.runtime_settings.open_timeout_ms,
            "read_timeout_ms": self.runtime_settings.read_timeout_ms,
            "event_cooldown_seconds": self.runtime_settings.event_cooldown_seconds,
        }
        default_monitor_options.update(monitor_options)
        self.detection_pipeline.load()
        self.color_classifier.load()
        self.plate_reader.load()
        self.monitor = CCTVMonitor(
            self.camera_manager,
            self.detection_pipeline,
            self.color_classifier,
            self.plate_reader,
            event_callback=self._save_and_alert,
            **default_monitor_options,
        )
        return self.monitor

    def _save_and_alert(self, event: object) -> None:
        """Save a real event first, then independently try its configured alert."""
        event_id = self.event_store.save_event(event)  # type: ignore[arg-type]
        self.email_alerts.send_event(event, event_id)  # type: ignore[arg-type]

    def close(self) -> None:
        """Stop live CCTV workers and release their in-memory frames."""
        if self.monitor is not None:
            self.monitor.stop()
            self.monitor = None

    def __enter__(self) -> ApplicationServices:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
