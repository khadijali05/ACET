"""Heuristic association of smoke detections with tracked vehicles."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import exp, sqrt
from typing import Literal, Sequence

from .detection_pipeline import Detection
from .tracking_id import TrackedVehicle


AssociationStatus = Literal["associated", "uncertain", "unassociated"]


@dataclass(frozen=True)
class AssociationSettings:
    lookback_frames: int = 15
    spatial_margin_x: int = 150
    spatial_margin_y: int = 100
    minimum_association_score: float = 0.40
    minimum_candidate_score: float = 0.22
    minimum_score_margin: float = 0.10
    history_decay_frames: float = 8.0

    def __post_init__(self) -> None:
        if self.lookback_frames < 0:
            raise ValueError("lookback_frames cannot be negative.")
        for name in (
            "minimum_association_score",
            "minimum_candidate_score",
            "minimum_score_margin",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.history_decay_frames <= 0:
            raise ValueError("history_decay_frames must be greater than zero.")


@dataclass(frozen=True)
class SmokeVehicleMatch:
    smoke: Detection
    status: AssociationStatus
    vehicle_track_id: int | None
    candidate_track_id: int | None
    association_score: float
    score_margin: float
    evidence_age_frames: int | None


class SmokeVehicleMatcher:
    """Match smoke boxes to current or recent tracked vehicles using geometry."""

    def __init__(self, settings: AssociationSettings | None = None) -> None:
        self.settings = settings or AssociationSettings()
        self._history: deque[tuple[int, tuple[TrackedVehicle, ...]]] = deque(
            maxlen=self.settings.lookback_frames + 1
        )
        self._last_frame_index: int | None = None

    def update(
        self,
        frame_index: int,
        smoke_detections: Sequence[Detection],
        tracked_vehicles: Sequence[TrackedVehicle],
    ) -> tuple[SmokeVehicleMatch, ...]:
        """Store current tracks and return an association result for each smoke box."""
        if self._last_frame_index is not None and frame_index <= self._last_frame_index:
            raise ValueError("frame_index must increase for each analyzed frame.")
        self._last_frame_index = frame_index
        current_tracks = tuple(tracked_vehicles)
        self._history.append((frame_index, current_tracks))

        return tuple(self._match_one(frame_index, smoke) for smoke in smoke_detections)

    def reset(self) -> None:
        """Clear temporal evidence when a video or camera session is restarted."""
        self._history.clear()
        self._last_frame_index = None

    def _match_one(self, frame_index: int, smoke: Detection) -> SmokeVehicleMatch:
        candidates: dict[int, tuple[float, int]] = {}
        sx1, sy1, sx2, sy2 = smoke.xyxy
        if sx2 <= sx1 or sy2 <= sy1:
            return SmokeVehicleMatch(smoke, "unassociated", None, None, 0.0, 0.0, None)

        for evidence_frame, vehicles in reversed(self._history):
            age = frame_index - evidence_frame
            if age > self.settings.lookback_frames:
                continue
            age_weight = exp(-age / self.settings.history_decay_frames)
            for vehicle in vehicles:
                spatial_score = self._spatial_score(smoke.xyxy, vehicle.xyxy)
                score = spatial_score * age_weight
                previous = candidates.get(vehicle.track_id)
                if previous is None or score > previous[0]:
                    candidates[vehicle.track_id] = (score, age)

        ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)
        if not ranked:
            return SmokeVehicleMatch(smoke, "unassociated", None, None, 0.0, 0.0, None)

        best_id, (best_score, evidence_age) = ranked[0]
        second_score = ranked[1][1][0] if len(ranked) > 1 else 0.0
        margin = max(0.0, best_score - second_score)

        if best_score >= self.settings.minimum_association_score and margin >= self.settings.minimum_score_margin:
            return SmokeVehicleMatch(smoke, "associated", best_id, best_id, best_score, margin, evidence_age)
        if best_score >= self.settings.minimum_candidate_score:
            return SmokeVehicleMatch(smoke, "uncertain", None, best_id, best_score, margin, evidence_age)
        return SmokeVehicleMatch(smoke, "unassociated", None, best_id, best_score, margin, evidence_age)

    def _spatial_score(
        self,
        smoke_box: tuple[int, int, int, int],
        vehicle_box: tuple[int, int, int, int],
    ) -> float:
        sx1, sy1, sx2, sy2 = smoke_box
        vx1, vy1, vx2, vy2 = vehicle_box
        expanded = (
            vx1 - self.settings.spatial_margin_x,
            vy1 - self.settings.spatial_margin_y,
            vx2 + self.settings.spatial_margin_x,
            vy2 + self.settings.spatial_margin_y,
        )
        ex1, ey1, ex2, ey2 = expanded
        smoke_area = max(0, sx2 - sx1) * max(0, sy2 - sy1)
        if smoke_area == 0:
            return 0.0

        intersection_width = max(0, min(sx2, ex2) - max(sx1, ex1))
        intersection_height = max(0, min(sy2, ey2) - max(sy1, ey1))
        intersection_over_smoke = (intersection_width * intersection_height) / smoke_area

        dx = max(0, vx1 - sx2, sx1 - vx2)
        dy = max(0, vy1 - sy2, sy1 - vy2)
        distance = sqrt(dx * dx + dy * dy)
        vehicle_area = max(1, vx2 - vx1) * max(1, vy2 - vy1)
        distance_scale = max(25.0, 0.25 * sqrt(vehicle_area))
        proximity = exp(-distance / distance_scale)
        return 0.70 * intersection_over_smoke + 0.30 * proximity
