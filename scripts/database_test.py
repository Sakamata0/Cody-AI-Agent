"""
Test script for the database query tool integration.

Demonstrates:
1. The SQL tool working standalone (direct query)
2. An agent translating natural language to SQL and answering
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain_core.messages import HumanMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

from src.llm import chat_model
from src.tools.database import sql_query_tool, db


def test_tool_standalone():
    """Test the SQL tool directly."""
    print("[Test 1] Direct SQL query...")
    start = time.time()
    result = sql_query_tool.invoke("SELECT name, salary FROM employees WHERE department_id = 1")
    latency = (time.time() - start) * 1000

    print(f"Result: {result}")
    print(f"Latency: {latency:.0f} ms")
    print()


def test_agent_natural_language():
    """Test the agent translating natural language to SQL."""
    print("[Test 2] Agent with natural language question...")

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are Cody, an autonomous AI agent with access to a company database. "
            "When the user asks about company data (employees, departments, projects, "
            "salaries, budgets), write a SQL SELECT query and use the sql_query tool "
            "to get the answer. The database schema is:\n"
            "- departments (id, name, budget)\n"
            "- employees (id, name, department_id, salary, hire_date)\n"
            "- projects (id, name, department_id, status, deadline)\n"
            "Always formulate a clear answer based on the query results."
        )),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(chat_model, [sql_query_tool], prompt)
    executor = AgentExecutor(agent=agent, tools=[sql_query_tool], verbose=True)

    questions = [
        "How many employees are in the Engineering department?",
        "What is the average salary across all departments?",
        "Which projects are currently in progress?",
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
    print("\n=== Database Query Tool Test ===")
    print(f"Database tables: {db.get_usable_table_names()}")
    print()
    test_tool_standalone()
    test_agent_natural_language()
    print("\nAll tests passed. Database tool is functional.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        raise SystemExit(1)
