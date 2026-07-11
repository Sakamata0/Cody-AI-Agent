"""
Conversation storage — persists chat sessions to AWS S3.

Conversations are stored as JSON files:
  s3://s3-cody-bucket/conversations/{user_id}/{conversation_id}.json

User settings are stored as:
  s3://s3-cody-bucket/settings/{user_id}.json
"""

import json
import os
import re
import uuid
from datetime import datetime

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "s3-cody-bucket")
REGION = os.getenv("AWS_REGION", "eu-north-1")
PREFIX = "conversations"
SETTINGS_PREFIX = "settings"

s3 = boto3.client("s3", region_name=REGION)

# Pattern for valid conversation IDs: exactly 8 alphanumeric characters
_CONVERSATION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9]{8}$")


def validate_conversation_id(conversation_id: str) -> None:
    """
    Validate that a conversation_id is exactly 8 alphanumeric characters.
    Raises ValueError if invalid — prevents S3 path traversal attacks.
    """
    if not _CONVERSATION_ID_PATTERN.match(conversation_id):
        raise ValueError(
            "Invalid conversation_id: must be exactly 8 alphanumeric characters."
        )


def _key(user_id: str, conversation_id: str) -> str:
    """Build the user-scoped S3 object key."""
    return f"{PREFIX}/{user_id}/{conversation_id}.json"


# Local fallback directory for conversations when S3 is unavailable
_LOCAL_CONVERSATIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "conversations_local")
os.makedirs(_LOCAL_CONVERSATIONS_DIR, exist_ok=True)


def _local_conv_dir(user_id: str) -> str:
    """Build local directory path for user conversations."""
    safe_id = "".join(c for c in user_id if c.isalnum() or c in "-_")
    path = os.path.join(_LOCAL_CONVERSATIONS_DIR, safe_id)
    os.makedirs(path, exist_ok=True)
    return path


def list_conversations(user_id: str) -> list[dict]:
    """
    List all saved conversations for a specific user,
    sorted by last updated (newest first).
    """
    conversations = []

    # Try S3
    try:
        response = s3.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix=f"{PREFIX}/{user_id}/",
        )
        if "Contents" in response:
            for obj in response["Contents"]:
                try:
                    data = _get_object(obj["Key"])
                    if data:
                        conversations.append({
                            "id": data["id"],
                            "title": data.get("title", "Untitled"),
                            "updated_at": data.get("updated_at", ""),
                            "message_count": len(data.get("messages", [])),
                        })
                except (ClientError, json.JSONDecodeError, KeyError):
                    continue
    except ClientError:
        pass

    # Also check local fallback
    local_dir = _local_conv_dir(user_id)
    if os.path.exists(local_dir):
        for filename in os.listdir(local_dir):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(local_dir, filename), "r", encoding="utf-8") as f:
                        data = json.loads(f.read())
                    # Avoid duplicates
                    if not any(c["id"] == data["id"] for c in conversations):
                        conversations.append({
                            "id": data["id"],
                            "title": data.get("title", "Untitled"),
                            "updated_at": data.get("updated_at", ""),
                            "message_count": len(data.get("messages", [])),
                        })
                except (json.JSONDecodeError, OSError, KeyError):
                    continue

    conversations.sort(key=lambda x: x["updated_at"], reverse=True)
    return conversations


def load_conversation(user_id: str, conversation_id: str) -> dict | None:
    """Load a conversation by ID, scoped to user. Tries S3 then local."""
    validate_conversation_id(conversation_id)
    key = _key(user_id, conversation_id)
    result = _get_object(key)
    if result:
        return result

    # Fallback: local file
    local_path = os.path.join(_local_conv_dir(user_id), f"{conversation_id}.json")
    if os.path.exists(local_path):
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                return json.loads(f.read())
        except (json.JSONDecodeError, OSError):
            return None
    return None


def save_conversation(user_id: str, conversation_id: str, title: str, messages: list) -> None:
    """Save a conversation. Tries S3 first, falls back to local."""
    validate_conversation_id(conversation_id)
    key = _key(user_id, conversation_id)
    data = {
        "id": conversation_id,
        "title": title,
        "updated_at": datetime.now().isoformat(),
        "messages": messages,
    }
    body = json.dumps(data, ensure_ascii=False, indent=2)

    try:
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
    except (ClientError, Exception) as e:
        # S3 unavailable — save locally
        print(f"[STORAGE] S3 save failed ({e}), saving conversation locally.")
        local_path = os.path.join(_local_conv_dir(user_id), f"{conversation_id}.json")
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(body)


def delete_conversation(user_id: str, conversation_id: str) -> bool:
    """Delete a conversation. Returns True if deleted. Scoped to user."""
    validate_conversation_id(conversation_id)
    key = _key(user_id, conversation_id)
    deleted = False

    try:
        s3.delete_object(Bucket=BUCKET_NAME, Key=key)
        deleted = True
    except ClientError:
        pass

    # Also delete local copy if it exists
    local_path = os.path.join(_local_conv_dir(user_id), f"{conversation_id}.json")
    if os.path.exists(local_path):
        os.remove(local_path)
        deleted = True

    return deleted


def new_conversation_id() -> str:
    """Generate a new unique conversation ID (8 hex characters)."""
    return str(uuid.uuid4())[:8]


def generate_title(first_message: str) -> str:
    """Generate a conversation title using the LLM for a formal summary."""
    from src.llm import chat_model

    try:
        response = chat_model.invoke(
            f"Generate a short, formal title (max 5 words) for a conversation that starts with this message. "
            f"Return ONLY the title, nothing else. No quotes, no punctuation at the end.\n\n"
            f"Message: {first_message}"
        )
        title = response.content.strip().strip('"').strip("'")[:50]
        return title if title else first_message[:50]
    except Exception:
        title = first_message.strip()[:50]
        if len(first_message) > 50:
            title += "..."
        return title


def _get_object(key: str) -> dict | None:
    """Fetch and parse a JSON object from S3."""
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        body = response["Body"].read().decode("utf-8")
        return json.loads(body)
    except (ClientError, json.JSONDecodeError):
        return None


# --- User Settings Storage ---

# Local fallback directory for settings when S3 is unavailable
_LOCAL_SETTINGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "settings")
os.makedirs(_LOCAL_SETTINGS_DIR, exist_ok=True)


def _local_settings_path(user_id: str) -> str:
    """Build local file path for user settings."""
    # Sanitize user_id to prevent path traversal
    safe_id = "".join(c for c in user_id if c.isalnum() or c in "-_")
    return os.path.join(_LOCAL_SETTINGS_DIR, f"{safe_id}.json")


def load_settings(user_id: str) -> dict | None:
    """
    Load user settings. Tries S3 first, falls back to local file.
    Returns None if no settings file exists.
    """
    # Try S3 first
    key = f"{SETTINGS_PREFIX}/{user_id}.json"
    result = _get_object(key)
    if result is not None:
        return result

    # Fallback: local file
    local_path = _local_settings_path(user_id)
    if os.path.exists(local_path):
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                return json.loads(f.read())
        except (json.JSONDecodeError, OSError):
            return None

    return None


def save_settings(user_id: str, settings_dict: dict) -> None:
    """
    Save user settings. Tries S3 first, falls back to local file.
    """
    key = f"{SETTINGS_PREFIX}/{user_id}.json"
    body = json.dumps(settings_dict, ensure_ascii=False, indent=2)

    try:
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
    except (ClientError, Exception) as e:
        # S3 unavailable — save locally
        print(f"[STORAGE] S3 save failed ({e}), saving settings locally.")
        local_path = _local_settings_path(user_id)
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(body)
