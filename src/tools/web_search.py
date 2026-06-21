"""
Web Search Tool using Tavily API.

Provides the agent with the ability to search the web for recent information.
Tavily returns clean, LLM-optimized summaries of search results.
"""

import os

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# Tavily returns up to max_results search results, each with a summary snippet.
web_search_tool = TavilySearchResults(
    max_results=5,
    search_depth="basic",
    name="web_search",
    description=(
        "Search the web for current information. Use this tool when you need "
        "recent news, real-time data, or information that may not be in your "
        "training data. Input should be a search query string."
    ),
)
