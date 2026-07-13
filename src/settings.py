"""
User Settings Router — manages per-user preferences stored in S3.

Endpoints:
- GET /  : Retrieve user settings (returns defaults if none exist)
- PUT /  : Validate and persist updated settings

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
"""

import re

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import ValidationError

from src.middleware import get_current_user, get_verifier, security
from src.models import UserSettings, UserSettingsUpdate
from src.storage import load_settings, save_settings

router = APIRouter()


def _extract_display_name_from_email(email: str) -> str:
    """Extract a clean display name from email prefix. Removes numbers and special chars."""
    prefix = email.split("@")[0] if "@" in email else email
    # Remove numbers, dots, underscores, hyphens — keep only letters
    clean = re.sub(r"[^a-zA-ZÀ-ÿ]", "", prefix)
    return clean[:50] if len(clean) >= 2 else "User"


def _default_settings(display_name: str = "User") -> UserSettings:
    """
    Build default settings when no persisted settings exist.
    """
    return UserSettings(
        display_name=display_name,
        avatar_index=None,
        theme="dark",
        language="en",
    )


@router.get("", response_model=UserSettings)
async def get_settings(
    user_id: str = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> UserSettings:
    """
    Retrieve user settings from S3.
    Returns default settings if no settings file exists for this user.
    """
    data = load_settings(user_id)
    if data is None:
        # Try to extract email from token for a better default name
        display_name = "User"
        if credentials:
            try:
                import jwt as pyjwt
                unverified = pyjwt.decode(
                    credentials.credentials,
                    options={"verify_signature": False},
                )
                username = unverified.get("username", "")
                if "@" in username:
                    display_name = _extract_display_name_from_email(username)
            except Exception:
                pass
        return _default_settings(display_name)
    return UserSettings(**data)


@router.put("", response_model=UserSettings)
async def update_settings(
    settings: UserSettingsUpdate,
    user_id: str = Depends(get_current_user),
) -> UserSettings:
    """
    Validate and persist user settings to S3.

    Merges provided fields with existing settings (or defaults).
    Returns 400 with field-level error messages if validation fails.
    """
    # Load existing settings or defaults
    existing_data = load_settings(user_id)
    if existing_data is None:
        current = _default_settings()
    else:
        current = UserSettings(**existing_data)

    # Merge: apply only the fields that were explicitly provided
    update_data = settings.model_dump(exclude_unset=True)
    merged = current.model_dump()
    merged.update(update_data)

    # Validate the merged result
    try:
        validated = UserSettings(**merged)
    except ValidationError as e:
        errors = {}
        for error in e.errors():
            field = error["loc"][-1] if error["loc"] else "unknown"
            errors[str(field)] = error["msg"]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Validation error", "errors": errors},
        )

    # Persist to S3 (or local fallback)
    save_settings(user_id, validated.model_dump())

    return validated
