"""
REST API Tools registry.

Import all API tools here so the agent can access them from a single list.
To add a new API tool: create a new file in this folder and import it below.
"""

from src.tools.api.weather import weather_tool
from src.tools.api.exchange import exchange_rate_tool

# All API tools available to the agent.
api_tools = [weather_tool, exchange_rate_tool]
