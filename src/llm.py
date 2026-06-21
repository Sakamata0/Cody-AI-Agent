"""
LangChain ChatModel configured for AWS Bedrock.

This module provides a ready-to-use ChatBedrockConverse instance that serves
as the foundation for all LLM interactions in the Cody AI Agent project.
"""

import os

from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse

# Load environment variables from .env file at the project root.
load_dotenv()

# Configuration with sensible defaults matching our AWS setup.
REGION = os.getenv("AWS_REGION", "eu-north-1")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "eu.anthropic.claude-haiku-4-5-20251001-v1:0")

# Default chat model instance.
chat_model = ChatBedrockConverse(
    model=MODEL_ID,
    region_name=REGION,
    temperature=0.2,
    max_tokens=1024,
)
