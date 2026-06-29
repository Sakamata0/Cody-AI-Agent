"""
Cody AI Agent — Core ReAct Agent with multi-tool orchestration and memory.

This module defines the central agent that combines all available tools
and uses a ReAct (Reasoning and Acting) approach to solve complex,
multi-step problems. Includes conversational memory for context retention.
"""

import sys
import os
import time
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler

load_dotenv()

# Import the LLM and all tools.
from src.llm import chat_model
from src.tools.web_search import web_search_tool
from src.tools.database import sql_query_tool
from src.tools.api import api_tools

# All tools available to the agent.
ALL_TOOLS = [web_search_tool, sql_query_tool] + api_tools

# System prompt that defines Cody's behavior and reasoning approach.
SYSTEM_PROMPT = """You are Cody, an autonomous AI agent built on AWS Bedrock.

You have access to the following tools:
- web_search: Search the web for current information and recent news.
- sql_query: Query the company database (departments, employees, projects).
- weather_tool: Get current weather for any city.
- exchange_rate_tool: Convert currencies or get exchange rates.

REASONING APPROACH:
1. Analyze the user's question and identify what information is needed.
2. If the question requires multiple steps, plan them before acting.
3. Use tools one at a time, observe the result, then decide the next step.
4. Chain multiple tools when needed (e.g., search web THEN convert currency).
5. Always formulate a clear, concise final answer based on all observations.

RULES:
- Only use tools when external data is needed. Answer from knowledge when possible.
- If a tool returns an error, try a different approach or inform the user.
- Be concise but thorough in your final answers.
- Show your reasoning when solving multi-step problems.
- Use conversation history to resolve references like "it", "that", "its price", etc.
- NEVER call the same tool with the same parameters more than once.
- If after 3 tool calls you still don't have a satisfactory answer, provide your best answer with what you have. Do NOT keep retrying.
- If a tool keeps failing, inform the user about the issue and give a partial answer.
"""


class StepLoggingHandler(BaseCallbackHandler):
    """Callback handler that logs each reasoning step and detects loops."""

    def __init__(self):
        self.steps = []
        self.step_count = 0
        self.tool_calls = []  # Track (tool, input) to detect duplicates.

    def on_agent_action(self, action: Any, **kwargs) -> None:
        """Called when the agent decides to use a tool."""
        self.step_count += 1

        # Detect duplicate tool calls (same tool + same input).
        call_signature = (action.tool, str(action.tool_input))
        if call_signature in self.tool_calls:
            print(f"  [Step {self.step_count}] DUPLICATE CALL DETECTED: {action.tool}")
        self.tool_calls.append(call_signature)

        step = {
            "step": self.step_count,
            "type": "action",
            "tool": action.tool,
            "input": action.tool_input,
        }
        self.steps.append(step)
        print(f"  [Step {self.step_count}] Action: {action.tool}({action.tool_input})")

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when a tool returns a result."""
        output_str = str(output)
        display_output = output_str[:200] + "..." if len(output_str) > 200 else output_str
        step = {
            "step": self.step_count,
            "type": "observation",
            "output": output_str,
        }
        self.steps.append(step)
        print(f"  [Step {self.step_count}] Observation: {display_output}")

    def on_agent_finish(self, finish: Any, **kwargs) -> None:
        """Called when the agent produces a final answer."""
        self.step_count += 1
        step = {
            "step": self.step_count,
            "type": "final_answer",
            "output": finish.return_values.get("output", ""),
        }
        self.steps.append(step)


class ChatSession:
    """
    Manages a conversation session with memory.

    Stores the full history of human/AI messages so the agent can
    resolve anaphores and maintain context across turns.
    """

    def __init__(self):
        self.history = []  # List of HumanMessage / AIMessage objects.
        self.executor = self._create_executor()

    def _create_executor(self) -> AgentExecutor:
        """Create the agent executor."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        agent = create_tool_calling_agent(chat_model, ALL_TOOLS, prompt)

        return AgentExecutor(
            agent=agent,
            tools=ALL_TOOLS,
            verbose=False,
            max_iterations=10,
            max_execution_time=60,  # Hard timeout: 60 seconds max.
            handle_parsing_errors=True,
            early_stopping_method="generate",  # Force LLM to produce a final answer when stopped.
        )

    def chat(self, message: str) -> dict:
        """
        Send a message and get a response, maintaining conversation history.

        Args:
            message: The user's message.

        Returns:
            dict with keys: output, steps, latency_ms
        """
        logger = StepLoggingHandler()

        print(f"\n{'='*60}")
        print(f"User: {message}")
        print(f"{'='*60}")
        print("Reasoning steps:")

        start = time.time()
        result = self.executor.invoke(
            {"input": message, "chat_history": self.history},
            config={"callbacks": [logger]},
        )
        latency = (time.time() - start) * 1000

        # Store the exchange in memory.
        self.history.append(HumanMessage(content=message))
        self.history.append(AIMessage(content=result["output"]))

        print(f"\nCody: {result['output']}")
        print(f"Total Steps: {logger.step_count} | Latency: {latency:.0f} ms")
        print(f"Memory: {len(self.history)} messages stored")
        print(f"{'='*60}")

        return {
            "output": result["output"],
            "steps": logger.steps,
            "latency_ms": latency,
        }

    def clear(self):
        """Clear conversation history."""
        self.history = []
        print("[Memory cleared]")


def create_agent() -> AgentExecutor:
    """Create and return the Cody agent executor (stateless, no memory)."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(chat_model, ALL_TOOLS, prompt)

    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=False,
        max_iterations=10,
        max_execution_time=60,  # Hard timeout: 60 seconds max.
        handle_parsing_errors=True,
        early_stopping_method="generate",  # Force LLM to produce a final answer when stopped.
    )


def run_agent(query: str, chat_history: list = None) -> dict:
    """
    Run the agent on a single query (stateless). For multi-turn
    conversations with memory, use ChatSession instead.

    Returns:
        dict with keys: output, steps, latency_ms
    """
    if chat_history is None:
        chat_history = []

    executor = create_agent()
    logger = StepLoggingHandler()

    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    print("Reasoning steps:")

    start = time.time()
    result = executor.invoke(
        {"input": query, "chat_history": chat_history},
        config={"callbacks": [logger]},
    )
    latency = (time.time() - start) * 1000

    print(f"\nFinal Answer: {result['output']}")
    print(f"Total Steps: {logger.step_count} | Latency: {latency:.0f} ms")
    print(f"{'='*60}")

    return {
        "output": result["output"],
        "steps": logger.steps,
        "latency_ms": latency,
    }
