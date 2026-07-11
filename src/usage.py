"""
Usage tracking — weekly message limits per user.

Storage: local JSON files at data/usage/{user_id}.json
(with S3 fallback when available)

Each user's usage file:
{
  "messages_used": 5,
  "week_start": "2026-07-06T00:00:00",
  "messages_limit": 50
}

The week resets every Monday at 00:00 UTC.
"""

import json
import os
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.middleware import get_current_user

router = APIRouter()

# Default weekly limit per user
DEFAULT_WEEKLY_LIMIT = 50

# Local storage for usage data
_USAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "usage")
os.makedirs(_USAGE_DIR, exist_ok=True)


def _usage_path(user_id: str) -> str:
    safe_id = "".join(c for c in user_id if c.isalnum() or c in "-_")
    return os.path.join(_USAGE_DIR, f"{safe_id}.json")


def _get_week_start() -> datetime:
    """Get the start of the current week (Monday 00:00 UTC)."""
    now = datetime.utcnow()
    monday = now - timedelta(days=now.weekday())
    return monday.replace(hour=0, minute=0, second=0, microsecond=0)


def _get_week_end() -> datetime:
    """Get the end of the current week (next Monday 00:00 UTC)."""
    return _get_week_start() + timedelta(days=7)


def _load_usage(user_id: str) -> dict:
    """Load usage data for a user. Resets if week has passed."""
    path = _usage_path(user_id)
    week_start = _get_week_start()

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.loads(f.read())

            # Check if we need to reset (new week)
            stored_week = datetime.fromisoformat(data.get("week_start", ""))
            if stored_week < week_start:
                # New week — reset
                data = {
                    "messages_used": 0,
                    "week_start": week_start.isoformat(),
                    "messages_limit": data.get("messages_limit", DEFAULT_WEEKLY_LIMIT),
                }
                _save_usage(user_id, data)
            return data
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    # No file or corrupted — create fresh
    data = {
        "messages_used": 0,
        "week_start": week_start.isoformat(),
        "messages_limit": DEFAULT_WEEKLY_LIMIT,
    }
    _save_usage(user_id, data)
    return data


def _save_usage(user_id: str, data: dict) -> None:
    """Persist usage data locally."""
    path = _usage_path(user_id)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, indent=2))


def increment_usage(user_id: str) -> bool:
    """
    Increment message count for user. Returns True if within limit,
    False if limit reached.
    """
    data = _load_usage(user_id)
    if data["messages_used"] >= data["messages_limit"]:
        return False
    data["messages_used"] += 1
    _save_usage(user_id, data)
    return True


def check_usage(user_id: str) -> dict:
    """Get current usage data for a user."""
    return _load_usage(user_id)


# --- API Response Model ---

class UsageResponse(BaseModel):
    messages_used: int
    messages_limit: int
    resets_at: str


# --- Endpoint ---

@router.get("", response_model=UsageResponse)
async def get_usage(user_id: str = Depends(get_current_user)) -> UsageResponse:
    """Get weekly usage stats for the authenticated user."""
    data = _load_usage(user_id)
    resets_at = _get_week_end().isoformat()
    return UsageResponse(
        messages_used=data["messages_used"],
        messages_limit=data["messages_limit"],
        resets_at=resets_at,
    )
