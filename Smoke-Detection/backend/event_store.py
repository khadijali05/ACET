"""Persistent storage, filtering, and Excel-compatible export for CCTV events."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cctv_monitor import CCTVDetectionEvent


@dataclass(frozen=True)
class EventPage:
    records: tuple[dict[str, Any], ...]
    total_records: int
    page: int
    page_size: int


class EventStore:
    """Store actual CCTV detection events; video-test events are not accepted here."""

    EXPORT_COLUMNS = (
        "event_id",
        "timestamp_utc",
        "camera_id",
        "camera_name",
        "frame_index",
        "vehicle_track_id",
        "association_status",
        "association_score",
        "smoke_classification",
        "classification_status",
        "classification_heuristic_score",
        "smoke_detector_score",
        "smoke_x1",
        "smoke_y1",
        "smoke_x2",
        "smoke_y2",
        "license_plate",
        "plate_location_status",
        "plate_detector_score",
        "plate_x1",
        "plate_y1",
        "plate_x2",
        "plate_y2",
        "snapshot_ref",
    )

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS detection_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp_utc TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    camera_name TEXT NOT NULL,
                    frame_index INTEGER NOT NULL,
                    vehicle_track_id INTEGER,
                    candidate_vehicle_track_id INTEGER,
                    association_status TEXT NOT NULL,
                    association_score REAL NOT NULL,
                    smoke_classification TEXT,
                    classification_status TEXT NOT NULL,
                    classification_heuristic_score REAL NOT NULL,
                    classification_explanation TEXT NOT NULL,
                    smoke_detector_score REAL NOT NULL,
                    smoke_box_json TEXT NOT NULL,
                    pixel_shares_json TEXT NOT NULL,
                    florence_caption TEXT,
                    license_plate TEXT,
                    plate_status TEXT NOT NULL DEFAULT 'not_checked',
                    plate_detector_score REAL,
                    plate_box_json TEXT,
                    processing_status TEXT NOT NULL,
                    snapshot_ref TEXT
                )"""
            )
            existing_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(detection_events)").fetchall()
            }
            migrations = {
                "plate_status": "ALTER TABLE detection_events ADD COLUMN plate_status TEXT NOT NULL DEFAULT 'not_checked'",
                "plate_detector_score": "ALTER TABLE detection_events ADD COLUMN plate_detector_score REAL",
                "plate_box_json": "ALTER TABLE detection_events ADD COLUMN plate_box_json TEXT",
            }
            for column, statement in migrations.items():
                if column not in existing_columns:
                    connection.execute(statement)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_detection_events_time ON detection_events(timestamp_utc DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_detection_events_camera ON detection_events(camera_id, timestamp_utc DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_detection_events_class ON detection_events(smoke_classification, timestamp_utc DESC)"
            )

    def save_event(self, event: CCTVDetectionEvent) -> str:
        """Persist one event received from the live CCTV monitor."""
        if not isinstance(event, CCTVDetectionEvent):
            raise TypeError("EventStore accepts CCTVDetectionEvent records only.")
        event_id = str(uuid.uuid4())
        colour = event.colour
        processing_status = "processed" if colour.status != "unavailable" else "classification_unavailable"
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO detection_events (
                    event_id, timestamp_utc, camera_id, camera_name, frame_index,
                    vehicle_track_id, candidate_vehicle_track_id, association_status,
                    association_score, smoke_classification, classification_status,
                    classification_heuristic_score, classification_explanation,
                    smoke_detector_score, smoke_box_json, pixel_shares_json,
                    florence_caption, license_plate, plate_status, plate_detector_score,
                    plate_box_json, processing_status, snapshot_ref
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event_id,
                    event.timestamp_utc,
                    event.camera_id,
                    event.camera_name,
                    event.frame_index,
                    event.vehicle_track_id,
                    event.candidate_vehicle_track_id,
                    event.association_status,
                    event.association_score,
                    colour.label,
                    colour.status,
                    colour.heuristic_score,
                    colour.explanation,
                    event.smoke_detector_score,
                    json.dumps(event.smoke_box),
                    json.dumps(colour.pixel_shares, sort_keys=True),
                    colour.florence_caption,
                    event.plate.text,
                    event.plate.status,
                    event.plate.confidence,
                    json.dumps(event.plate.xyxy) if event.plate.xyxy is not None else None,
                    processing_status,
                    None,
                ),
            )
        return event_id

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM detection_events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return self._to_dict(row) if row else None

    def list_events(
        self,
        *,
        camera_id: str | None = None,
        start_utc: str | None = None,
        end_utc: str | None = None,
        smoke_classification: str | None = None,
        association_status: str | None = None,
        search: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> EventPage:
        if page < 1:
            raise ValueError("page must be at least 1.")
        if not 1 <= page_size <= 500:
            raise ValueError("page_size must be between 1 and 500.")
        where: list[str] = []
        parameters: list[Any] = []
        if camera_id:
            where.append("camera_id = ?")
            parameters.append(camera_id)
        if start_utc:
            where.append("timestamp_utc >= ?")
            parameters.append(start_utc)
        if end_utc:
            where.append("timestamp_utc <= ?")
            parameters.append(end_utc)
        if smoke_classification:
            where.append("smoke_classification = ?")
            parameters.append(smoke_classification)
        if association_status:
            where.append("association_status = ?")
            parameters.append(association_status)
        if search:
            term = f"%{search.strip()}%"
            where.append(
                "(camera_name LIKE ? OR camera_id LIKE ? OR CAST(vehicle_track_id AS TEXT) LIKE ? OR smoke_classification LIKE ?)"
            )
            parameters.extend((term, term, term, term))
        clause = f" WHERE {' AND '.join(where)}" if where else ""
        offset = (page - 1) * page_size
        with self._connect() as connection:
            total = connection.execute(
                f"SELECT COUNT(*) FROM detection_events{clause}", parameters
            ).fetchone()[0]
            rows = connection.execute(
                f"SELECT * FROM detection_events{clause} ORDER BY timestamp_utc DESC, event_id DESC LIMIT ? OFFSET ?",
                [*parameters, page_size, offset],
            ).fetchall()
        return EventPage(tuple(self._to_dict(row) for row in rows), int(total), page, page_size)

    def export_csv(self, **filters: Any) -> bytes:
        """Return filtered records as UTF-8 CSV bytes that Excel can open."""
        filters = dict(filters)
        filters.pop("page", None)
        filters.pop("page_size", None)
        page = self.list_events(page=1, page_size=500, **filters)
        records = list(page.records)
        while len(records) < page.total_records:
            next_page = self.list_events(
                page=len(records) // 500 + 1,
                page_size=500,
                **filters,
            )
            if not next_page.records:
                break
            records.extend(next_page.records)

        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=self.EXPORT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            box = record["smoke_box"]
            plate_box = record["plate_box"]
            export_row = dict(record)
            export_row.update(
                {
                    "smoke_x1": box[0],
                    "smoke_y1": box[1],
                    "smoke_x2": box[2],
                    "smoke_y2": box[3],
                    "plate_location_status": record["plate_status"],
                    "plate_detector_score": record["plate_detector_score"],
                    "plate_x1": plate_box[0] if plate_box else None,
                    "plate_y1": plate_box[1] if plate_box else None,
                    "plate_x2": plate_box[2] if plate_box else None,
                    "plate_y2": plate_box[3] if plate_box else None,
                }
            )
            writer.writerow(export_row)
        return output.getvalue().encode("utf-8-sig")

    @staticmethod
    def _to_dict(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["smoke_box"] = json.loads(result.pop("smoke_box_json"))
        result["pixel_shares"] = json.loads(result.pop("pixel_shares_json"))
        plate_box_json = result.pop("plate_box_json", None)
        result["plate_box"] = json.loads(plate_box_json) if plate_box_json else None
        return result
