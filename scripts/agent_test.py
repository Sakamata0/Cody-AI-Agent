"""
Test script for the core Cody agent with multi-step reasoning.

Demonstrates:
1. Single-tool usage (agent picks the right tool)
2. Multi-step reasoning (agent chains multiple tools)
3. Step logging (each Thought/Action/Observation is tracked)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.agent import run_agent


def main():
    print("\n" + "=" * 60)
    print("         CODY AI AGENT — Multi-Step Reasoning Test")
    print("=" * 60)

    # Test 1: Single tool — should use weather_tool
    run_agent("What's the weather in London right now?")

    # Test 2: Single tool — should use sql_query
    run_agent("How many employees do we have in the Sales department?")

    # Test 3: Multi-step — should use web_search + exchange_rate_tool
    run_agent(
        "Search the web for the current price of Bitcoin in USD, "
        "then convert that amount to EUR."
    )

    # Test 4: Multi-step — should use sql_query + sql_query (multiple queries)
    run_agent(
        "Which department has the highest average salary, "
        "and how many projects does that department have?"
    )

    print("\n\nAll tests completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        raise SystemExit(1)
