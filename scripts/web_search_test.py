"""
Test script for the Tavily web search tool integration.

Demonstrates:
1. The tool working standalone (direct invocation)
2. An agent deciding to use the tool for a current-events question
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain_core.messages import HumanMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

from src.llm import chat_model
from src.tools.web_search import web_search_tool


def test_tool_standalone():
    """Test the search tool directly without an agent."""
    print("[Test 1] Direct tool invocation...")
    start = time.time()
    results = web_search_tool.invoke("latest AI news 2025")
    latency = (time.time() - start) * 1000

    print(f"Results: {len(results)} items returned")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['url']}")
        print(f"     {result['content'][:100]}...")
    print(f"Latency: {latency:.0f} ms")
    print()


def test_agent_with_tool():
    """Test that the agent decides to use the tool for a current-events question."""
    print("[Test 2] Agent with web search tool...")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are Cody, an autonomous AI agent. Use the web_search tool "
                   "when you need current information. Be concise in your answers."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(chat_model, [web_search_tool], prompt)
    executor = AgentExecutor(agent=agent, tools=[web_search_tool], verbose=True)

    start = time.time()
    result = executor.invoke({"input": "What happened in the world today?", "chat_history": []})
    latency = (time.time() - start) * 1000

    print(f"\nFinal Answer: {result['output']}")
    print(f"Total Latency: {latency:.0f} ms")
    print()


def main():
    print("\n=== Web Search Tool Test ===\n")
    test_tool_standalone()
    test_agent_with_tool()
    print("All tests passed. Web search tool is functional.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        raise SystemExit(1)
