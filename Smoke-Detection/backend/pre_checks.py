"""Readiness checks that run before live CCTV monitoring starts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

from .camera_manager import CameraManager
from .settings import ApplicationSettings


@dataclass(frozen=True)
class PreCheck:
    """One clear readiness result for the dashboard."""

    component: str
    status: str
    message: str


@dataclass(frozen=True)
class PreCheckReport:
    """All readiness results and whether live monitoring may start."""

    checks: tuple[PreCheck, ...]

    @property
    def ready(self) -> bool:
        return all(check.status != "error" for check in self.checks)


class PreChecks:
    """Check prerequisites without loading models or opening CCTV streams."""

    def __init__(
        self,
        project_root: str | Path,
        settings: ApplicationSettings,
        camera_manager: CameraManager,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.settings = settings
        self.camera_manager = camera_manager

    def run(self) -> PreCheckReport:
        """Return GPU, model, and protected-camera readiness results."""
        checks = [self._gpu_check(), *self._model_checks(), self._camera_key_check()]
        return PreCheckReport(tuple(checks))

    def _gpu_check(self) -> PreCheck:
        if not torch.cuda.is_available():
            return PreCheck(
                "GPU",
                "error",
                "CUDA is unavailable. Check the NVIDIA driver and CUDA-enabled PyTorch installation.",
            )
        device_index = self._device_index()
        if device_index is None or device_index >= torch.cuda.device_count():
            return PreCheck(
                "GPU",
                "error",
                f"Configured device {self.settings.device} is unavailable. "
                f"Detected GPU count: {torch.cuda.device_count()}.",
            )
        return PreCheck(
            "GPU",
            "ready",
            f"{self.settings.device} is ready: {torch.cuda.get_device_name(device_index)}.",
        )

    def _model_checks(self) -> tuple[PreCheck, ...]:
        models = (
            ("Smoke detector", "smoke_detector.pt"),
            ("Vehicle detector", "vehicle_detector.pt"),
            ("Plate detector", "plate_detector.pt"),
        )
        checks: list[PreCheck] = []
        for name, filename in models:
            present = (self.project_root / "models" / filename).is_file()
            checks.append(
                PreCheck(
                    name,
                    "ready" if present else "error",
                    f"{filename} is available." if present else f"{filename} is missing from the models folder.",
                )
            )
        return tuple(checks)

    def _camera_key_check(self) -> PreCheck:
        enabled_cameras = self.camera_manager.list_cameras(enabled_only=True)
        if not enabled_cameras:
            return PreCheck("Cameras", "warning", "No enabled cameras are configured yet.")
        if os.environ.get("SMOKE_DETECTION_CAMERA_KEY"):
            return PreCheck("Camera credentials", "ready", "Camera credential encryption key is configured.")
        return PreCheck(
            "Camera credentials",
            "warning",
            "No camera credential key is configured. Cameras without saved credentials can still run.",
        )

    def _device_index(self) -> int | None:
        device = self.settings.device.strip().lower()
        if device == "cuda":
            return 0
        if device.startswith("cuda:"):
            try:
                index = int(device.partition(":")[2])
            except ValueError:
                return None
            return index if index >= 0 else None
        return None
