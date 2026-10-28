"""Build clear summaries from saved CCTV detection events."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .event_store import EventStore


@dataclass(frozen=True)
class EventReport:
    """A report summary for actual events saved within the requested filters."""

    total_events: int
    smoke_type_counts: dict[str, int]
    camera_counts: dict[str, int]
    association_counts: dict[str, int]
    classification_unavailable: int
    first_event_utc: str | None
    last_event_utc: str | None


class ReportGenerator:
    """Summarize events from an EventStore without creating placeholder results."""

    def __init__(self, event_store: EventStore) -> None:
        self.event_store = event_store

    def generate(
        self,
        *,
        start_utc: str | None = None,
        end_utc: str | None = None,
        camera_id: str | None = None,
    ) -> EventReport:
        """Count real events in a time window, optionally limited to one camera.

        Timestamps should use a consistent, sortable UTC ISO-8601 format, such as
        ``2026-10-28T12:30:00+00:00``. Both bounds include their endpoint.
        """
        if start_utc and end_utc and start_utc > end_utc:
            raise ValueError("start_utc must be earlier than or equal to end_utc.")

        page_number = 1
        page_size = 500
        total = 0
        smoke_types: Counter[str] = Counter()
        cameras: Counter[str] = Counter()
        associations: Counter[str] = Counter()
        unavailable = 0
        first_event: str | None = None
        last_event: str | None = None

        while True:
            page = self.event_store.list_events(
                camera_id=camera_id,
                start_utc=start_utc,
                end_utc=end_utc,
                page=page_number,
                page_size=page_size,
            )
            if not page.records:
                break

            for event in page.records:
                total += 1
                timestamp = event["timestamp_utc"]
                first_event = timestamp if first_event is None or timestamp < first_event else first_event
                last_event = timestamp if last_event is None or timestamp > last_event else last_event
                smoke_types[event["smoke_classification"] or "unclassified"] += 1
                cameras[event["camera_name"]] += 1
                associations[event["association_status"]] += 1
                if event["classification_status"] == "unavailable":
                    unavailable += 1

            if total >= page.total_records:
                break
            page_number += 1

        return EventReport(
            total_events=total,
            smoke_type_counts=dict(sorted(smoke_types.items())),
            camera_counts=dict(sorted(cameras.items())),
            association_counts=dict(sorted(associations.items())),
            classification_unavailable=unavailable,
            first_event_utc=first_event,
            last_event_utc=last_event,
        )
