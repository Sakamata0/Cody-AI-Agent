"""
Test script for REST API tools (weather + exchange rate).

Demonstrates:
1. Direct tool invocations
2. An agent deciding which API tool to use based on the question
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

from src.llm import chat_model
from src.tools.api import api_tools
from src.tools.api.weather import weather_tool
from src.tools.api.exchange import exchange_rate_tool


def test_weather_standalone():
    """Test the weather tool directly."""
    print("[Test 1] Weather tool - direct invocation...")
    start = time.time()
    result = weather_tool.invoke("Tunis")
    latency = (time.time() - start) * 1000
    print(f"Result:\n{result}")
    print(f"Latency: {latency:.0f} ms")
    print()


def test_exchange_standalone():
    """Test the exchange rate tool directly."""
    print("[Test 2] Exchange rate tool - direct invocation...")
    start = time.time()
    result = exchange_rate_tool.invoke("100 USD to EUR")
    latency = (time.time() - start) * 1000
    print(f"Result:\n{result}")
    print(f"Latency: {latency:.0f} ms")
    print()


def test_agent_with_api_tools():
    """Test the agent choosing the right API tool."""
    print("[Test 3] Agent deciding which tool to use...")

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are Cody, an autonomous AI agent with access to real-time data tools. "
            "Use the weather_tool for weather questions and exchange_rate_tool for "
            "currency conversion. Be concise in your answers."
        )),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(chat_model, api_tools, prompt)
    executor = AgentExecutor(agent=agent, tools=api_tools, verbose=True)

    questions = [
        "What's the weather like in Paris right now?",
        "How much is 250 EUR in Japanese Yen?",
        "Is it cold in New York today?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        start = time.time()
        result = executor.invoke({"input": question, "chat_history": []})
        latency = (time.time() - start) * 1000
        print(f"Answer: {result['output']}")
        print(f"Latency: {latency:.0f} ms")
        print("-" * 60)


def main():
    print("\n=== REST API Tools Test ===\n")
    test_weather_standalone()
    test_exchange_standalone()
    test_agent_with_api_tools()
    print("\nAll tests passed. API tools are functional.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        raise SystemExit(1)
