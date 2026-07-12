"""
Migration endpoint — moves legacy conversations to per-user storage.

One-time migration: moves conversations/{id}.json to conversations/{admin_user_id}/{id}.json.
Idempotent — returns success if already migrated.
"""

import re

import boto3
from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends

from src.middleware import get_current_user
from src.models import MessageResponse

router = APIRouter()

BUCKET_NAME = "s3-cody-bucket"
PREFIX = "conversations"

s3 = boto3.client("s3", region_name="eu-north-1")


@router.post("/migrate-conversations", response_model=MessageResponse)
async def migrate_conversations(
    admin_user_id: str = Depends(get_current_user),
) -> MessageResponse:
    """
    One-time migration: moves conversations/{id}.json to conversations/{admin_user_id}/{id}.json.
    Idempotent — returns success if already migrated.

    Only migrates files at the root of conversations/ (i.e., conversations/xxx.json),
    not files already in a subdirectory (conversations/user_id/xxx.json).
    """
    # Pattern for root-level conversation files: conversations/{id}.json
    # These are files NOT in a subdirectory under conversations/
    root_file_pattern = re.compile(r"^conversations/([^/]+\.json)$")

    try:
        # List all objects under conversations/ prefix
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=f"{PREFIX}/")

        files_to_migrate = []
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                match = root_file_pattern.match(key)
                if match:
                    files_to_migrate.append(key)

    except ClientError:
        files_to_migrate = []

    if not files_to_migrate:
        return MessageResponse(message="No migration necessary")

    migrated_count = 0
    for source_key in files_to_migrate:
        # Extract filename (e.g., "068c014c.json")
        filename = source_key.split("/")[-1]
        destination_key = f"{PREFIX}/{admin_user_id}/{filename}"

        try:
            # Copy the object to the new location
            s3.copy_object(
                Bucket=BUCKET_NAME,
                CopySource={"Bucket": BUCKET_NAME, "Key": source_key},
                Key=destination_key,
            )
            # Delete the original
            s3.delete_object(Bucket=BUCKET_NAME, Key=source_key)
            migrated_count += 1
        except ClientError:
            # Skip files that fail to migrate
            continue

    if migrated_count == 0:
        return MessageResponse(message="No migration necessary")

    return MessageResponse(
        message=f"Successfully migrated {migrated_count} conversation(s)"
    )
