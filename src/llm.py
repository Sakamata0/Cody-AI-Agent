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


def get_chat_model(
    model_id: str = MODEL_ID,
    region: str = REGION,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> ChatBedrockConverse:
    """
    Create and return a ChatBedrockConverse instance.

    Args:
        model_id: The Bedrock model identifier.
        region: AWS region where Bedrock is available.
        temperature: Sampling temperature (0 = deterministic, 1 = creative).
        max_tokens: Maximum tokens in the model response.

    Returns:
        A configured ChatBedrockConverse instance ready for use.
    """
    return ChatBedrockConverse(
        model=model_id,
        region_name=region,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# Default instance — import this directly for simple use cases.
chat_model = get_chat_model()
