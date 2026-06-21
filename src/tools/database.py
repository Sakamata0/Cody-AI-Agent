"""
Database Query Tool using LangChain's SQL utilities.

Allows the agent to query a SQLite database in read-only mode,
translating natural language questions into SQL queries.
"""

import os

from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_core.tools import tool

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "company.db")

# Connect to the SQLite database in read-only mode.
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")


@tool
def database_query(question: str) -> str:
    """
    Query the company database using natural language.
    Use this tool when the user asks about employees, departments,
    projects, salaries, budgets, or any internal company data.
    Input should be a SQL query string (SELECT only).
    """
    # Only allow SELECT statements for read-only access.
    if not question.strip().upper().startswith("SELECT"):
        return "Error: Only SELECT queries are allowed. The database is read-only."

    try:
        result = db.run(question)
        return result if result else "No results found."
    except Exception as e:
        return f"Query error: {str(e)}"


# Tool that the agent can use to run SQL directly.
sql_query_tool = QuerySQLDatabaseTool(db=db, name="sql_query", description=(
    "Execute a SQL SELECT query against the company database and return results. "
    "The database contains tables: departments (id, name, budget), "
    "employees (id, name, department_id, salary, hire_date), "
    "projects (id, name, department_id, status, deadline). "
    "Input must be a valid SQL SELECT query."
))
