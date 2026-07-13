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


def list_conversations(user_id: str) -> list[dict]:
    """
    List all saved conversations for a specific user,
    sorted by last updated (newest first).
    """
    conversations = []

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

    conversations.sort(key=lambda x: x["updated_at"], reverse=True)
    return conversations


def load_conversation(user_id: str, conversation_id: str) -> dict | None:
    """Load a conversation by ID, scoped to user."""
    validate_conversation_id(conversation_id)
    key = _key(user_id, conversation_id)
    return _get_object(key)


def save_conversation(user_id: str, conversation_id: str, title: str, messages: list) -> None:
    """Save a conversation to S3."""
    validate_conversation_id(conversation_id)
    key = _key(user_id, conversation_id)
    data = {
        "id": conversation_id,
        "title": title,
        "updated_at": datetime.now().isoformat(),
        "messages": messages,
    }
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=json.dumps(data, ensure_ascii=False, indent=2),
        ContentType="application/json",
    )


def delete_conversation(user_id: str, conversation_id: str) -> bool:
    """Delete a conversation. Returns True if deleted. Scoped to user."""
    validate_conversation_id(conversation_id)
    key = _key(user_id, conversation_id)
    try:
        s3.delete_object(Bucket=BUCKET_NAME, Key=key)
        return True
    except ClientError:
        return False


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


def load_settings(user_id: str) -> dict | None:
    """Load user settings from S3."""
    key = f"{SETTINGS_PREFIX}/{user_id}.json"
    return _get_object(key)


def save_settings(user_id: str, settings_dict: dict) -> None:
    """Save user settings to S3."""
    key = f"{SETTINGS_PREFIX}/{user_id}.json"
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=json.dumps(settings_dict, ensure_ascii=False, indent=2),
        ContentType="application/json",
    )
