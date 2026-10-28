"""Administrator credential verification without storing readable passwords."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass


class AuthenticationConfigurationError(RuntimeError):
    """Raised when the administrator credentials are not configured safely."""


@dataclass(frozen=True)
class AuthenticationResult:
    authenticated: bool
    message: str


class AdministratorAuthenticator:
    """Verify one configured administrator account using a salted scrypt hash."""

    HASH_SCHEME = "scrypt"
    SCRYPT_N = 2**14
    SCRYPT_R = 8
    SCRYPT_P = 1
    SALT_BYTES = 16
    KEY_BYTES = 32

    def __init__(self, username: str, password_hash: str) -> None:
        if not username.strip():
            raise AuthenticationConfigurationError("Administrator username is empty.")
        if not self._is_valid_hash(password_hash):
            raise AuthenticationConfigurationError(
                "Administrator password hash is missing or invalid. Configure SMOKE_DETECTION_ADMIN_PASSWORD_HASH."
            )
        self._username = username.strip()
        self._password_hash = password_hash

    @classmethod
    def from_environment(cls) -> AdministratorAuthenticator:
        """Read the administrator identity and password hash from environment variables."""
        username = os.environ.get("SMOKE_DETECTION_ADMIN_USERNAME", "").strip()
        password_hash = os.environ.get("SMOKE_DETECTION_ADMIN_PASSWORD_HASH", "").strip()
        if not username:
            raise AuthenticationConfigurationError(
                "Set SMOKE_DETECTION_ADMIN_USERNAME before starting the administrator dashboard."
            )
        if not password_hash:
            raise AuthenticationConfigurationError(
                "Set SMOKE_DETECTION_ADMIN_PASSWORD_HASH before starting the administrator dashboard."
            )
        return cls(username, password_hash)

    @classmethod
    def hash_password(cls, password: str) -> str:
        """Create a salted scrypt password hash for secure configuration."""
        if not password:
            raise ValueError("Password cannot be empty.")
        salt = secrets.token_bytes(cls.SALT_BYTES)
        digest = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=cls.SCRYPT_N,
            r=cls.SCRYPT_R,
            p=cls.SCRYPT_P,
            dklen=cls.KEY_BYTES,
        )
        return "$".join(
            (
                cls.HASH_SCHEME,
                str(cls.SCRYPT_N),
                str(cls.SCRYPT_R),
                str(cls.SCRYPT_P),
                base64.urlsafe_b64encode(salt).decode("ascii"),
                base64.urlsafe_b64encode(digest).decode("ascii"),
            )
        )

    @classmethod
    def _is_valid_hash(cls, encoded: str) -> bool:
        try:
            scheme, n_text, r_text, p_text, salt_text, digest_text = encoded.split("$")
            n, r, p = int(n_text), int(r_text), int(p_text)
            salt = base64.urlsafe_b64decode(salt_text.encode("ascii"))
            digest = base64.urlsafe_b64decode(digest_text.encode("ascii"))
            return (
                scheme == cls.HASH_SCHEME
                and n >= 2**14
                and n & (n - 1) == 0
                and r > 0
                and p > 0
                and len(salt) >= 16
                and len(digest) >= 32
            )
        except (ValueError, TypeError):
            return False

    def authenticate(self, username: str, password: str) -> AuthenticationResult:
        """Return the same generic failure message for an unknown user or bad password."""
        if not isinstance(username, str) or not isinstance(password, str):
            return AuthenticationResult(False, "Invalid username or password.")
        username_ok = hmac.compare_digest(username.strip().encode("utf-8"), self._username.encode("utf-8"))
        password_ok = self._verify_password(password)
        if username_ok and password_ok:
            return AuthenticationResult(True, "Login successful.")
        return AuthenticationResult(False, "Invalid username or password.")

    def _verify_password(self, password: str) -> bool:
        try:
            scheme, n_text, r_text, p_text, salt_text, digest_text = self._password_hash.split("$")
            if scheme != self.HASH_SCHEME:
                return False
            salt = base64.urlsafe_b64decode(salt_text.encode("ascii"))
            expected = base64.urlsafe_b64decode(digest_text.encode("ascii"))
            actual = hashlib.scrypt(
                password.encode("utf-8"),
                salt=salt,
                n=int(n_text),
                r=int(r_text),
                p=int(p_text),
                dklen=len(expected),
            )
            return hmac.compare_digest(actual, expected)
        except (ValueError, TypeError, MemoryError):
            return False
