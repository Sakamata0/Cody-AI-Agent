"""
Pydantic models for authentication requests, responses, and user settings.
"""

import re
from typing import Literal

from pydantic import BaseModel, field_validator


# --- Auth Request Models ---


class RegisterRequest(BaseModel):
    """Registration request with email and password."""

    email: str
    password: str

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """
        Validate password meets policy:
        - At least 8 characters
        - At least one uppercase letter
        - At least one lowercase letter
        - At least one digit
        - At least one special character
        """
        violations = validate_password_rules(v)
        if violations:
            raise ValueError(
                f"Password must contain: {', '.join(violations)}"
            )
        return v


class VerifyRequest(BaseModel):
    """Email verification request with 6-digit code."""

    email: str
    code: str


class LoginRequest(BaseModel):
    """Login request with email and password."""

    email: str
    password: str


class RefreshRequest(BaseModel):
    """Token refresh request."""

    refresh_token: str


# --- Auth Response Models ---


class UserInfo(BaseModel):
    """User identity information from Cognito."""

    user_id: str
    email: str


class TokenResponse(BaseModel):
    """Authentication token response."""

    access_token: str
    id_token: str
    refresh_token: str | None = None
    user: UserInfo


class MessageResponse(BaseModel):
    """Generic message response."""

    message: str


# --- User Settings Models ---


class UserSettings(BaseModel):
    """User preferences stored in S3."""

    display_name: str
    avatar_index: int | None = None
    theme: Literal["light", "dark"]
    language: Literal["en", "fr", "ar"]

    @field_validator("display_name")
    @classmethod
    def validate_display_name(cls, v: str) -> str:
        """Display name must be 1-50 characters, letters/spaces/hyphens/apostrophes only."""
        if not v or len(v) < 1:
            raise ValueError("Display name must be at least 1 character")
        if len(v) > 50:
            raise ValueError("Display name must be at most 50 characters")
        if re.search(r"[0-9]", v):
            raise ValueError("Display name must not contain numbers")
        if re.search(r"[^a-zA-ZÀ-ÿ\s\'\-]", v):
            raise ValueError("Display name can only contain letters, spaces, hyphens, and apostrophes")
        return v

    @field_validator("avatar_index")
    @classmethod
    def validate_avatar_index(cls, v: int | None) -> int | None:
        """Avatar index must be 0-11 or null."""
        if v is not None and (v < 0 or v > 11):
            raise ValueError("Avatar index must be between 0 and 11")
        return v


class UserSettingsUpdate(BaseModel):
    """Partial update for user settings. All fields optional."""

    display_name: str | None = None
    avatar_index: int | None = None
    theme: Literal["light", "dark"] | None = None
    language: Literal["en", "fr", "ar"] | None = None

    @field_validator("display_name")
    @classmethod
    def validate_display_name(cls, v: str | None) -> str | None:
        """Display name must be 1-50 characters if provided, letters/spaces/hyphens/apostrophes only."""
        if v is None:
            return v
        if len(v) < 1:
            raise ValueError("Display name must be at least 1 character")
        if len(v) > 50:
            raise ValueError("Display name must be at most 50 characters")
        if re.search(r"[0-9]", v):
            raise ValueError("Display name must not contain numbers")
        if re.search(r"[^a-zA-ZÀ-ÿ\s\'\-]", v):
            raise ValueError("Display name can only contain letters, spaces, hyphens, and apostrophes")
        return v

    @field_validator("avatar_index")
    @classmethod
    def validate_avatar_index(cls, v: int | None) -> int | None:
        """Avatar index must be 0-11 or null."""
        if v is not None and (v < 0 or v > 11):
            raise ValueError("Avatar index must be between 0 and 11")
        return v


# --- Password Validation Utility ---


def validate_password_rules(password: str) -> list[str]:
    """
    Check password against all rules and return a list of violations.

    Rules:
    - At least 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character

    Returns an empty list if the password is valid.
    """
    violations = []

    if len(password) < 8:
        violations.append("at least 8 characters")

    if not re.search(r"[A-Z]", password):
        violations.append("at least one uppercase letter")

    if not re.search(r"[a-z]", password):
        violations.append("at least one lowercase letter")

    if not re.search(r"\d", password):
        violations.append("at least one digit")

    if not re.search(r"[^A-Za-z0-9]", password):
        violations.append("at least one special character")

    return violations
