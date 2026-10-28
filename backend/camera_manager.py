"""Persistent camera configuration and safe stream connection checks."""

from __future__ import annotations

import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

import cv2
from cryptography.fernet import Fernet, InvalidToken


CAMERA_STATUSES = {
    "configured",
    "connecting",
    "connected",
    "disconnected",
    "authentication_failed",
    "stream_unavailable",
    "processing_error",
    "disabled",
}


class CameraConfigurationError(ValueError):
    """Raised for invalid camera configuration or unavailable credential protection."""


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    display_name: str
    stream_url: str
    enabled: bool
    status: str
    created_at: str
    updated_at: str
    last_checked_at: str | None


@dataclass(frozen=True)
class ConnectionCheck:
    camera_id: str
    status: str
    message: str
    elapsed_seconds: float


class CameraManager:
    """Store camera definitions and test their current stream availability."""

    def __init__(
        self,
        database_path: str | Path,
        credential_key: str | bytes | None = None,
        open_timeout_ms: int = 5000,
        read_timeout_ms: int = 5000,
    ) -> None:
        self.database_path = Path(database_path).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.open_timeout_ms = open_timeout_ms
        self.read_timeout_ms = read_timeout_ms
        configured_key = credential_key or os.environ.get("SMOKE_DETECTION_CAMERA_KEY")
        self._fernet: Fernet | None = None
        if configured_key:
            try:
                key_bytes = configured_key.encode() if isinstance(configured_key, str) else configured_key
                self._fernet = Fernet(key_bytes)
            except Exception as exc:
                raise CameraConfigurationError(
                    "SMOKE_DETECTION_CAMERA_KEY must be a valid Fernet key."
                ) from exc
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS cameras (
                    camera_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    stream_url TEXT NOT NULL,
                    username_token TEXT,
                    password_token TEXT,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'configured',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_checked_at TEXT
                )"""
            )

    @staticmethod
    def _validate_url(stream_url: str) -> str:
        value = stream_url.strip()
        parts = urlsplit(value)
        if parts.scheme.lower() not in {"rtsp", "rtsps", "http", "https"} or not parts.hostname:
            raise CameraConfigurationError(
                "Use a valid RTSP, RTSPS, HTTP, or HTTPS camera stream address."
            )
        if parts.username is not None or parts.password is not None:
            raise CameraConfigurationError(
                "Enter camera username and password in the credential fields, not inside the stream address."
            )
        sensitive_query_names = {"password", "passwd", "token", "access_token", "api_key", "auth"}
        query_names = {item.partition("=")[0].lower() for item in parts.query.split("&") if item}
        if sensitive_query_names & query_names:
            raise CameraConfigurationError(
                "Do not put credentials or access tokens in the stream address. Use the protected credential fields."
            )
        return value

    def _encrypt(self, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        if self._fernet is None:
            raise CameraConfigurationError(
                "Camera credentials are not saved until SMOKE_DETECTION_CAMERA_KEY is configured."
            )
        return self._fernet.encrypt(value.encode("utf-8")).decode("ascii")

    def _decrypt(self, token: str | None) -> str | None:
        if not token:
            return None
        if self._fernet is None:
            raise CameraConfigurationError(
                "Camera credentials are encrypted. Configure SMOKE_DETECTION_CAMERA_KEY to use this camera."
            )
        try:
            return self._fernet.decrypt(token.encode("ascii")).decode("utf-8")
        except InvalidToken as exc:
            raise CameraConfigurationError(
                "Camera credentials could not be decrypted. Check the configured credential key."
            ) from exc

    def add_camera(
        self,
        display_name: str,
        stream_url: str,
        *,
        username: str | None = None,
        password: str | None = None,
        enabled: bool = True,
    ) -> CameraConfig:
        name = display_name.strip()
        if not name:
            raise CameraConfigurationError("Camera display name cannot be empty.")
        clean_url = self._validate_url(stream_url)
        username_token = self._encrypt(username)
        password_token = self._encrypt(password)
        now = self._now()
        camera_id = str(uuid.uuid4())
        status = "configured" if enabled else "disabled"
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO cameras
                   (camera_id, display_name, stream_url, username_token, password_token,
                    enabled, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (camera_id, name, clean_url, username_token, password_token, int(enabled), status, now, now),
            )
        return self.get_camera(camera_id)

    def update_camera(
        self,
        camera_id: str,
        *,
        display_name: str,
        stream_url: str,
        username: str | None = None,
        password: str | None = None,
        replace_credentials: bool = False,
    ) -> CameraConfig:
        name = display_name.strip()
        if not name:
            raise CameraConfigurationError("Camera display name cannot be empty.")
        clean_url = self._validate_url(stream_url)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT enabled, username_token, password_token FROM cameras WHERE camera_id = ?",
                (camera_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Camera {camera_id} was not found.")
            username_token = self._encrypt(username) if replace_credentials else row["username_token"]
            password_token = self._encrypt(password) if replace_credentials else row["password_token"]
            now = self._now()
            status = "configured" if row["enabled"] else "disabled"
            connection.execute(
                """UPDATE cameras SET display_name = ?, stream_url = ?, username_token = ?,
                   password_token = ?, status = ?, updated_at = ? WHERE camera_id = ?""",
                (name, clean_url, username_token, password_token, status, now, camera_id),
            )
        return self.get_camera(camera_id)

    def set_enabled(self, camera_id: str, enabled: bool) -> CameraConfig:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE cameras SET enabled = ?, status = ?, updated_at = ? WHERE camera_id = ?",
                (int(enabled), "configured" if enabled else "disabled", self._now(), camera_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Camera {camera_id} was not found.")
        return self.get_camera(camera_id)

    def remove_camera(self, camera_id: str) -> None:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM cameras WHERE camera_id = ?", (camera_id,))
            if cursor.rowcount == 0:
                raise KeyError(f"Camera {camera_id} was not found.")

    def list_cameras(self, *, enabled_only: bool = False) -> tuple[CameraConfig, ...]:
        with self._connect() as connection:
            query = "SELECT * FROM cameras"
            if enabled_only:
                query += " WHERE enabled = 1"
            query += " ORDER BY display_name COLLATE NOCASE"
            rows = connection.execute(query).fetchall()
        return tuple(self._public_config(row) for row in rows)

    def get_camera(self, camera_id: str) -> CameraConfig:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM cameras WHERE camera_id = ?", (camera_id,)).fetchone()
        if row is None:
            raise KeyError(f"Camera {camera_id} was not found.")
        return self._public_config(row)

    def _stream_url_for_worker(self, camera_id: str) -> str:
        """Build a credentialed stream URL for backend use; never return it to the UI."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT stream_url, username_token, password_token FROM cameras WHERE camera_id = ?",
                (camera_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Camera {camera_id} was not found.")
        return self._authenticated_url(
            row["stream_url"],
            self._decrypt(row["username_token"]),
            self._decrypt(row["password_token"]),
        )

    @staticmethod
    def _public_config(row: sqlite3.Row) -> CameraConfig:
        return CameraConfig(
            camera_id=row["camera_id"],
            display_name=row["display_name"],
            stream_url=row["stream_url"],
            enabled=bool(row["enabled"]),
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_checked_at=row["last_checked_at"],
        )

    def set_status(self, camera_id: str, status: str) -> CameraConfig:
        if status not in CAMERA_STATUSES:
            raise CameraConfigurationError(f"Unsupported camera status: {status}")
        with self._connect() as connection:
            row = connection.execute("SELECT enabled FROM cameras WHERE camera_id = ?", (camera_id,)).fetchone()
            if row is None:
                raise KeyError(f"Camera {camera_id} was not found.")
            final_status = "disabled" if not row["enabled"] else status
            connection.execute(
                "UPDATE cameras SET status = ?, updated_at = ? WHERE camera_id = ?",
                (final_status, self._now(), camera_id),
            )
        return self.get_camera(camera_id)

    def test_connection(self, camera_id: str) -> ConnectionCheck:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM cameras WHERE camera_id = ?", (camera_id,)).fetchone()
        if row is None:
            raise KeyError(f"Camera {camera_id} was not found.")
        if not row["enabled"]:
            return ConnectionCheck(camera_id, "disabled", "Monitoring is disabled for this camera.", 0.0)

        started = time.monotonic()
        username = self._decrypt(row["username_token"])
        password = self._decrypt(row["password_token"])
        stream_url = self._authenticated_url(row["stream_url"], username, password)
        self.set_status(camera_id, "connecting")
        cap = None
        try:
            params = [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.open_timeout_ms,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.read_timeout_ms,
            ]
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG, params)
            if not cap.isOpened():
                status = "stream_unavailable"
                message = "The stream could not be opened. Check the address, credentials, and camera availability."
            else:
                ok, _frame = cap.read()
                status = "connected" if ok else "stream_unavailable"
                message = "Camera stream is connected and returned a video frame." if ok else "The stream opened but did not return a video frame."
        except Exception as exc:
            error_text = str(exc).lower()
            status = "authentication_failed" if any(code in error_text for code in ("401", "403", "unauthorized", "forbidden")) else "stream_unavailable"
            message = "Camera authentication failed." if status == "authentication_failed" else "The stream connection test failed. Check the address and stream format."
        finally:
            if cap is not None:
                cap.release()

        elapsed = time.monotonic() - started
        with self._connect() as connection:
            connection.execute(
                "UPDATE cameras SET status = ?, updated_at = ?, last_checked_at = ? WHERE camera_id = ?",
                (status, self._now(), self._now(), camera_id),
            )
        return ConnectionCheck(camera_id, status, message, elapsed)

    @staticmethod
    def _authenticated_url(stream_url: str, username: str | None, password: str | None) -> str:
        if not username and not password:
            return stream_url
        parts = urlsplit(stream_url)
        user_info = quote(username or "", safe="")
        if password:
            user_info += ":" + quote(password, safe="")
        host = parts.hostname or ""
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        if parts.port:
            host += f":{parts.port}"
        return urlunsplit((parts.scheme, f"{user_info}@{host}", parts.path, parts.query, parts.fragment))

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
