"""
LangChain ChatModel configured for AWS Bedrock.

This module provides a ready-to-use ChatBedrockConverse instance that serves
as the foundation for all LLM interactions in the Cody AI Agent project.
Includes rate limiting and retry with exponential backoff for throttling errors.
"""

import os

from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from botocore.config import Config

# Load environment variables from .env file at the project root.
load_dotenv()

# Configuration with sensible defaults matching our AWS setup.
REGION = os.getenv("AWS_REGION", "eu-north-1")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "eu.anthropic.claude-haiku-4-5-20251001-v1:0")

# Boto3 retry configuration — handles throttling at the SDK level.
# Uses exponential backoff with up to 5 retries on 429 errors.
bedrock_config = Config(
    retries={
        "max_attempts": 5,
        "mode": "adaptive",  # Adaptive mode automatically handles throttling with backoff.
    }
)

# Default chat model instance with retry config.
chat_model = ChatBedrockConverse(
    model=MODEL_ID,
    region_name=REGION,
    temperature=0.2,
    max_tokens=1024,
    config=bedrock_config,
)
