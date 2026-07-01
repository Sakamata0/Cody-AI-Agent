"""
Conversation storage — persists chat sessions as JSON files.

Each conversation is saved as a JSON file in data/conversations/.
This can be swapped to S3 in production by changing save/load functions.
"""

import json
import os
import uuid
from datetime import datetime

CONVERSATIONS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "conversations"
)


def _ensure_dir():
    """Create the conversations directory if it doesn't exist."""
    os.makedirs(CONVERSATIONS_DIR, exist_ok=True)


def list_conversations() -> list[dict]:
    """
    List all saved conversations, sorted by last updated (newest first).

    Returns:
        List of dicts with: id, title, updated_at, message_count
    """
    _ensure_dir()
    conversations = []

    for filename in os.listdir(CONVERSATIONS_DIR):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(CONVERSATIONS_DIR, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            conversations.append({
                "id": data["id"],
                "title": data.get("title", "Untitled"),
                "updated_at": data.get("updated_at", ""),
                "message_count": len(data.get("messages", [])),
            })
        except (json.JSONDecodeError, KeyError):
            continue

    conversations.sort(key=lambda x: x["updated_at"], reverse=True)
    return conversations


def load_conversation(conversation_id: str) -> dict | None:
    """Load a conversation by ID."""
    _ensure_dir()
    filepath = os.path.join(CONVERSATIONS_DIR, f"{conversation_id}.json")
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_conversation(conversation_id: str, title: str, messages: list) -> None:
    """Save a conversation to disk."""
    _ensure_dir()
    filepath = os.path.join(CONVERSATIONS_DIR, f"{conversation_id}.json")
    data = {
        "id": conversation_id,
        "title": title,
        "updated_at": datetime.now().isoformat(),
        "messages": messages,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def delete_conversation(conversation_id: str) -> bool:
    """Delete a conversation file. Returns True if deleted."""
    filepath = os.path.join(CONVERSATIONS_DIR, f"{conversation_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False


def new_conversation_id() -> str:
    """Generate a new unique conversation ID."""
    return str(uuid.uuid4())[:8]


def generate_title(first_message: str) -> str:
    """Generate a conversation title from the first user message."""
    title = first_message.strip()[:50]
    if len(first_message) > 50:
        title += "..."
    return title
