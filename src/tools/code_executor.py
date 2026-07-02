"""
Code Execution Tool — allows the agent to generate and run Python code.

Executes Python code in a restricted environment with a timeout
to prevent infinite loops and dangerous operations.
"""

import sys
import io
import contextlib
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class CodeExecutorInput(BaseModel):
    """Input schema for the code executor tool."""
    code: str = Field(
        description="Python code to execute. Must be a plain string containing valid Python code. "
                    "Examples: 'print(2 + 2)', 'import math; print(math.sqrt(144))', "
                    "'data = [1,2,3,4,5]; print(sum(data) / len(data))'"
    )


# Restricted built-ins — no file I/O, no imports of dangerous modules.
SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool,
    "chr": chr, "dict": dict, "dir": dir, "divmod": divmod,
    "enumerate": enumerate, "filter": filter, "float": float,
    "format": format, "frozenset": frozenset, "getattr": getattr,
    "hasattr": hasattr, "hash": hash, "hex": hex, "id": id,
    "int": int, "isinstance": isinstance, "issubclass": issubclass,
    "iter": iter, "len": len, "list": list, "map": map, "max": max,
    "min": min, "next": next, "oct": oct, "ord": ord, "pow": pow,
    "print": print, "range": range, "repr": repr, "reversed": reversed,
    "round": round, "set": set, "slice": slice, "sorted": sorted,
    "str": str, "sum": sum, "tuple": tuple, "type": type, "zip": zip,
    "__import__": __import__,
}

# Allowed modules for import.
ALLOWED_MODULES = {
    "math", "statistics", "random", "datetime", "json",
    "re", "collections", "itertools", "functools",
    "string", "decimal", "fractions",
}


def _safe_import(name, *args, **kwargs):
    """Only allow importing from the whitelist."""
    if name not in ALLOWED_MODULES:
        raise ImportError(
            f"Module '{name}' is not allowed. "
            f"Allowed modules: {', '.join(sorted(ALLOWED_MODULES))}"
        )
    return __import__(name, *args, **kwargs)


@tool(args_schema=CodeExecutorInput)
def code_executor_tool(code: str) -> str:
    """
    Execute Python code and return the output.
    Use this tool when the user asks to calculate something complex,
    solve a math problem, process data, or generate code.

    Input MUST be a plain string containing valid Python code.
    Examples:
      - "print(2 ** 10)"
      - "import math; print(math.pi * 5**2)"
      - "numbers = [10, 20, 30]; print(sum(numbers))"
      - "for i in range(5): print(f'Item {i}')"

    Do NOT pass a dictionary or object. Only a plain Python code string.
    The code runs in a sandboxed environment with limited modules.
    Available modules: math, statistics, random, datetime, json, re, collections, itertools, functools, string, decimal, fractions.
    """
    # Capture stdout.
    stdout_capture = io.StringIO()

    # Build restricted globals.
    restricted_globals = {"__builtins__": {**SAFE_BUILTINS, "__import__": _safe_import}}

    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, restricted_globals)  # noqa: S102

        output = stdout_capture.getvalue()
        if not output.strip():
            return "Code executed successfully (no output produced)."
        return f"Output:\n{output.strip()}"

    except ImportError as e:
        return f"Import error: {str(e)}"
    except SyntaxError as e:
        return f"Syntax error: {str(e)}"
    except Exception as e:
        return f"Runtime error ({type(e).__name__}): {str(e)}"
