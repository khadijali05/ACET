"""Configure local administrator access and camera credential protection."""

from __future__ import annotations

import getpass
import os
import subprocess
import sys

from cryptography.fernet import Fernet

from backend.authentication import AdministratorAuthenticator


ENVIRONMENT_NAMES = (
    "SMOKE_DETECTION_ADMIN_USERNAME",
    "SMOKE_DETECTION_ADMIN_PASSWORD_HASH",
    "SMOKE_DETECTION_CAMERA_KEY",
)


def _set_user_environment(name: str, value: str) -> None:
    """Persist one non-plaintext setting for future Windows terminal sessions."""
    subprocess.run(["setx", name, value], check=True, capture_output=True, text=True)


def main() -> int:
    if os.name != "nt":
        print("This helper is for Windows because it uses Windows user environment settings.")
        return 1

    print("Smoke Detection local security setup")
    username = input("Administrator username: ").strip()
    if not username:
        print("Administrator username cannot be empty.")
        return 1

    password = getpass.getpass("Administrator password: ")
    confirmation = getpass.getpass("Confirm administrator password: ")
    if not password:
        print("Administrator password cannot be empty.")
        return 1
    if password != confirmation:
        print("Passwords did not match. No settings were changed.")
        return 1

    password_hash = AdministratorAuthenticator.hash_password(password)
    camera_key = Fernet.generate_key().decode("ascii")
    try:
        _set_user_environment("SMOKE_DETECTION_ADMIN_USERNAME", username)
        _set_user_environment("SMOKE_DETECTION_ADMIN_PASSWORD_HASH", password_hash)
        _set_user_environment("SMOKE_DETECTION_CAMERA_KEY", camera_key)
    except subprocess.CalledProcessError as exc:
        print(f"Windows could not save the settings: {exc}")
        return 1

    print("Security settings saved for your Windows user account.")
    print("Close this terminal, open a new PowerShell window, then start the dashboard.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
