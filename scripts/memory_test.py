"""
Test script for conversational memory.

Demonstrates:
1. The agent remembers context from previous messages
2. Anaphora resolution ("it", "that city", "its price", etc.)
3. Multi-turn conversation flow
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.agent import ChatSession


def main():
    print("\n" + "=" * 60)
    print("       CODY AI AGENT — Conversational Memory Test")
    print("=" * 60)

    session = ChatSession()

    # Turn 1: Establish context about a city.
    session.chat("What's the weather in Tokyo?")

    # Turn 2: Anaphora — "there" should refer to Tokyo.
    session.chat("Is it usually this hot there in June?")

    # Turn 3: Anaphora — "that city" should still refer to Tokyo.
    session.chat("How many employees do we have working in that city?")

    # Turn 4: New topic but referencing earlier context.
    session.chat("Now check the weather in Berlin.")

    # Turn 5: Anaphora — "both cities" should refer to Tokyo and Berlin.
    session.chat("Compare the temperatures of both cities.")

    print("\n\nMemory test completed.")
    print(f"Total messages in memory: {len(session.history)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        raise SystemExit(1)
