"""Encrypted email-alert configuration and delivery for saved CCTV events."""

from __future__ import annotations

import os
import smtplib
import sqlite3
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from .cctv_monitor import CCTVDetectionEvent
from .finding_error import get_logger


LOGGER = get_logger("email_alert")


class EmailAlertConfigurationError(ValueError):
    """Raised when email alert settings are incomplete or invalid."""


@dataclass(frozen=True)
class EmailAlertSettings:
    enabled: bool
    recipient_addresses: tuple[str, ...]
    smtp_host: str
    smtp_port: int
    use_starttls: bool
    sender_address: str
    username: str
    password_configured: bool


class EmailAlertService:
    """Send a concise detection report after an event has been saved locally."""

    def __init__(self, database_path: str | Path, *, credential_key: str | None = None) -> None:
        self.database_path = Path(database_path).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        key = credential_key or os.environ.get("SMOKE_DETECTION_CAMERA_KEY")
        self._fernet: Fernet | None = None
        if key:
            try:
                self._fernet = Fernet(key.encode("ascii"))
            except Exception as exc:
                raise EmailAlertConfigurationError(
                    "SMOKE_DETECTION_CAMERA_KEY must be a valid Fernet key before email passwords can be saved."
                ) from exc
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS email_alert_settings (
                    settings_id INTEGER PRIMARY KEY CHECK (settings_id = 1),
                    enabled INTEGER NOT NULL DEFAULT 0,
                    recipients TEXT NOT NULL DEFAULT '',
                    smtp_host TEXT NOT NULL DEFAULT '',
                    smtp_port INTEGER NOT NULL DEFAULT 587,
                    use_starttls INTEGER NOT NULL DEFAULT 1,
                    sender_address TEXT NOT NULL DEFAULT '',
                    username TEXT NOT NULL DEFAULT '',
                    password_token TEXT
                )"""
            )
            connection.execute("INSERT OR IGNORE INTO email_alert_settings (settings_id) VALUES (1)")

    def get_settings(self) -> EmailAlertSettings:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM email_alert_settings WHERE settings_id = 1").fetchone()
        return EmailAlertSettings(
            enabled=bool(row["enabled"]),
            recipient_addresses=tuple(item for item in row["recipients"].split(",") if item),
            smtp_host=row["smtp_host"], smtp_port=int(row["smtp_port"]),
            use_starttls=bool(row["use_starttls"]), sender_address=row["sender_address"],
            username=row["username"], password_configured=bool(row["password_token"]),
        )

    def save_settings(self, *, enabled: bool, recipient_addresses: str, smtp_host: str,
                      smtp_port: int, use_starttls: bool, sender_address: str, username: str,
                      password: str | None, replace_password: bool) -> EmailAlertSettings:
        recipients = self._normalise_recipients(recipient_addresses)
        host, sender, login = smtp_host.strip(), sender_address.strip(), username.strip()
        if enabled:
            if not recipients or not host or not sender:
                raise EmailAlertConfigurationError("Enter a recipient, SMTP server, and sender email before enabling alerts.")
            if not 1 <= smtp_port <= 65535:
                raise EmailAlertConfigurationError("SMTP port must be between 1 and 65535.")
        with self._connect() as connection:
            current = connection.execute("SELECT password_token FROM email_alert_settings WHERE settings_id = 1").fetchone()
            password_token = self._encrypt(password or "") if replace_password else current["password_token"]
            if enabled and not password_token:
                raise EmailAlertConfigurationError("Enter the SMTP app password before enabling alerts.")
            connection.execute(
                """UPDATE email_alert_settings SET enabled = ?, recipients = ?, smtp_host = ?, smtp_port = ?,
                   use_starttls = ?, sender_address = ?, username = ?, password_token = ? WHERE settings_id = 1""",
                (int(enabled), ",".join(recipients), host, smtp_port, int(use_starttls), sender, login, password_token),
            )
        return self.get_settings()

    @staticmethod
    def _normalise_recipients(value: str) -> tuple[str, ...]:
        recipients = tuple(item.strip() for item in value.replace(";", ",").split(",") if item.strip())
        if any("@" not in item or item.startswith("@") or item.endswith("@") for item in recipients):
            raise EmailAlertConfigurationError("Enter valid email addresses separated by commas.")
        return recipients

    def _encrypt(self, value: str) -> str | None:
        if not value:
            return None
        if self._fernet is None:
            raise EmailAlertConfigurationError("Run setup_security.py before saving an SMTP app password.")
        return self._fernet.encrypt(value.encode("utf-8")).decode("ascii")

    def _password(self) -> str:
        with self._connect() as connection:
            token = connection.execute("SELECT password_token FROM email_alert_settings WHERE settings_id = 1").fetchone()["password_token"]
        if not token or self._fernet is None:
            raise EmailAlertConfigurationError("An SMTP app password has not been configured.")
        try:
            return self._fernet.decrypt(token.encode("ascii")).decode("utf-8")
        except InvalidToken as exc:
            raise EmailAlertConfigurationError("The SMTP app password could not be decrypted. Save it again after checking the security key.") from exc

    def send_test(self) -> None:
        self._send(self.get_settings(), "Smoke Detection email alert test", "This is a test email from the Smoke Detection Console.")

    def send_event(self, event: CCTVDetectionEvent, event_id: str) -> None:
        settings = self.get_settings()
        if not settings.enabled:
            return
        smoke_type = (event.colour.label or "Unclassified").replace("_", " ").title()
        plate_text = event.plate.text or "Unreadable"
        vehicle = str(event.vehicle_track_id) if event.vehicle_track_id is not None else "No confirmed vehicle"
        subject = f"Smoke detection alert | {event.camera_name}"
        body = "\n".join((
            "A CCTV smoke detection event was saved.", "", f"Event ID: {event_id}",
            f"Time (UTC): {event.timestamp_utc}", f"Camera: {event.camera_name}",
            f"Smoke type: {smoke_type}", f"Smoke detector confidence: {event.smoke_detector_score:.1%}",
            f"Vehicle association: {event.association_status.title()} ({event.association_score:.1%})",
            f"Vehicle track ID: {vehicle}", f"License plate: {plate_text}",
            f"Plate result: {event.plate.status.replace('_', ' ').title()}", "",
            "Open Detection events in the Smoke Detection Console to review this record.",
        ))
        try:
            self._send(settings, subject, body)
        except Exception as exc:
            LOGGER.error("Detection alert could not be sent: error_type=%s", type(exc).__name__)

    def _send(self, settings: EmailAlertSettings, subject: str, body: str) -> None:
        if not settings.recipient_addresses or not settings.smtp_host or not settings.sender_address:
            raise EmailAlertConfigurationError("Complete the email alert settings before sending an email.")
        message = EmailMessage()
        message["Subject"], message["From"] = subject, settings.sender_address
        message["To"] = ", ".join(settings.recipient_addresses)
        message.set_content(body)
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=20) as smtp:
            smtp.ehlo()
            if settings.use_starttls:
                smtp.starttls(); smtp.ehlo()
            if settings.username:
                smtp.login(settings.username, self._password())
            smtp.send_message(message)
