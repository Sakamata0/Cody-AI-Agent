"""
Test script for the infinite loop bug fix.

Demonstrates:
1. Agent stops at max_iterations and produces a final answer
2. Agent handles impossible/ambiguous queries gracefully
3. Duplicate call detection works
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.agent import run_agent, create_agent, StepLoggingHandler


def test_max_iterations():
    """
    Ask a question that could lead to many tool calls.
    Verify the agent stops within the limit and gives a partial answer.
    """
    print("[Test 1] Max iterations — complex question that could loop...")
    result = run_agent(
        "Find the exact population of every country in Africa and their GDP per capita."
    )
    # The agent should answer with what it can find, not loop forever.
    assert result["output"], "Agent should produce a final answer"
    assert result["latency_ms"] < 60000, "Should complete within 60s timeout"
    print("Agent stopped and produced an answer within limits.")
    print()


def test_impossible_tool_query():
    """
    Ask something that will cause tool errors/empty results.
    The agent should give up gracefully instead of retrying forever.
    """
    print("[Test 2] Impossible query — agent should give up gracefully...")
    result = run_agent(
        "Query the database for the table 'unicorns' and get all unicorn names."
    )
    assert result["output"], "Agent should produce a final answer"
    print("Agent handled impossible query gracefully.")
    print()


def test_duplicate_detection():
    """Verify the duplicate call tracking works."""
    print("[Test 3] Duplicate call detection...")
    logger = StepLoggingHandler()

    # Simulate duplicate calls.
    class FakeAction:
        tool = "weather_tool"
        tool_input = {"city": "Paris"}

    logger.on_agent_action(FakeAction(), run_id="test")
    logger.on_agent_action(FakeAction(), run_id="test")

    # Check that the same call was tracked twice.
    assert len(logger.tool_calls) == 2
    assert logger.tool_calls[0] == logger.tool_calls[1]
    print("Duplicate detection working — warning printed above.")
    print()


def main():
    print("\n" + "=" * 60)
    print("       CODY AI AGENT — Infinite Loop Fix Test")
    print("=" * 60 + "\n")

    test_duplicate_detection()
    test_max_iterations()
    test_impossible_tool_query()

    print("All loop-fix tests passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        raise SystemExit(1)
