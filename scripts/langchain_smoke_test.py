"""
Smoke test for the LangChain + Bedrock integration.

Validates that ChatBedrockConverse is properly configured and can
generate responses through the LangChain abstraction layer.
"""

import sys
import os
import time

# Add project root to path so we can import from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain_core.messages import HumanMessage, SystemMessage
from src.llm import chat_model, MODEL_ID, REGION


def main():
    print("\n=== LangChain + Bedrock Smoke Test ===")
    print(f"Model : {MODEL_ID}")
    print(f"Region: {REGION}")
    print()

    # Test 1: Simple invocation
    print("[Test 1] Simple message...")
    start = time.time()
    response = chat_model.invoke("Say hello in one sentence. You are Cody, an AI agent.")
    latency = (time.time() - start) * 1000
    print(f"Response: {response.content}")
    print(f"Tokens  : {response.usage_metadata}")
    print(f"Latency : {latency:.0f} ms")
    print()

    # Test 2: Using message objects (system + human)
    print("[Test 2] System + Human messages...")
    messages = [
        SystemMessage(content="You are Cody, a helpful AI assistant built on AWS Bedrock."),
        HumanMessage(content="What can you help me with? Answer in one sentence."),
    ]
    start = time.time()
    response = chat_model.invoke(messages)
    latency = (time.time() - start) * 1000
    print(f"Response: {response.content}")
    print(f"Tokens  : {response.usage_metadata}")
    print(f"Latency : {latency:.0f} ms")
    print()

    print("All tests passed. LangChain base is functional.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        raise SystemExit(1)
