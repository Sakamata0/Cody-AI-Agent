"""
User Settings Router — manages per-user preferences stored in S3.

Endpoints:
- GET /  : Retrieve user settings (returns defaults if none exist)
- PUT /  : Validate and persist updated settings

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import ValidationError

from src.middleware import get_current_user
from src.models import UserSettings, UserSettingsUpdate
from src.storage import load_settings, save_settings

router = APIRouter()


def _default_settings(user_id: str) -> UserSettings:
    """
    Build default settings when no persisted settings exist.

    Default values:
    - display_name: first 8 chars of user_id (since we only have sub, not email)
    - avatar_index: null (shows initials)
    - theme: "dark"
    - language: "en"
    """
    return UserSettings(
        display_name=user_id[:8],
        avatar_index=None,
        theme="dark",
        language="en",
    )


@router.get("", response_model=UserSettings)
async def get_settings(user_id: str = Depends(get_current_user)) -> UserSettings:
    """
    Retrieve user settings from S3.
    Returns default settings if no settings file exists for this user.
    """
    data = load_settings(user_id)
    if data is None:
        return _default_settings(user_id)
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
        current = _default_settings(user_id)
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
