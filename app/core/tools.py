from datetime import datetime
from pydantic import create_model
import inspect
import json
from inspect import Parameter
from typing import Any, Callable, List, Dict
import asyncio
import importlib
import pkgutil
import app.tools as tools_package
import sys
import os
from pathlib import Path


# Add the new tool management classes/functions
def schema(f: Callable) -> Dict[str, Any]:
    """Create a function schema compatible with OpenAI's tool format"""
    # Get the original function if it's decorated
    original_func = getattr(f, "__wrapped__", f)

    # Create parameter dict, defaulting to string type if annotation is empty
    kw = {}
    for name, param in inspect.signature(original_func).parameters.items():
        # Skip 'kernel' parameter as it's injected
        if name == "kernel":
            continue

        # If no type annotation, default to str
        annotation = param.annotation if param.annotation != Parameter.empty else str
        default = ... if param.default == Parameter.empty else param.default
        kw[name] = (annotation, default)

    s = create_model(f"Input for `{original_func.__name__}`", **kw).model_json_schema()
    # Remove the title and definitions which aren't needed in the function schema
    s.pop("title", None)
    s.pop("definitions", None)

    # Special case for python function to ensure correct schema
    if original_func.__name__ == "python":
        return {
            "type": "function",
            "name": "python",
            "description": original_func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute",
                    }
                },
                "required": ["code"],
            },
        }

    return {
        "type": "function",
        "name": original_func.__name__,
        "description": original_func.__doc__ or "",
        "parameters": s,
    }


def call_func(
    name: str, arguments: str, available_functions: Dict[str, Callable]
) -> Any:
    """Call a function by name with JSON-formatted arguments"""
    if name not in available_functions:
        raise ValueError(f"Function {name} not found in available functions")
    f = available_functions[name]
    return f(**json.loads(arguments))


def tool(f: Callable) -> Callable:
    """Decorator to mark a function as a tool"""
    f._is_tool = True
    return f


def bind_kernel(f: Callable, kernel) -> Callable:
    """Bind a kernel to a function that needs it"""

    def wrapped(*args, **kwargs):
        if "kernel" not in kwargs:
            kwargs["kernel"] = kernel
        return f(*args, **kwargs)

    # Preserve the original function's metadata
    wrapped._is_tool = getattr(f, "_is_tool", False)
    wrapped.__name__ = f.__name__
    wrapped.__doc__ = f.__doc__
    return wrapped


class ToolManager:
    def __init__(self):
        self.tools: List[Dict[str, Any]] = []
        self.available_functions: Dict[str, Callable] = {}
        self.tool_choice = "none"
        # Add history tracking
        self.tool_history: List[Dict[str, Any]] = []
        self.jupyter_kernel = None

    def get_available_tools(self) -> List[Callable]:
        """Return list of all tool functions marked with @tool decorator"""
        tools = []

        # Get the tools directory path
        tools_dir = Path(__file__).parent.parent / "tools"

        # Import all Python files in the tools directory
        for file in tools_dir.glob("*.py"):
            if file.name != "__init__.py":
                module_name = f"app.tools.{file.stem}"
                if module_name not in sys.modules:
                    try:
                        importlib.import_module(module_name)
                    except ImportError as e:
                        print(f"Error importing {module_name}: {e}")

        # Look through all loaded modules in the tools package
        for module_name, module in sys.modules.items():
            if module_name.startswith("app.tools.") and not module_name.endswith(
                "__init__"
            ):
                if module:  # Check if module is not None
                    # Get all functions marked with @tool decorator
                    tools.extend(
                        [
                            obj
                            for _, obj in inspect.getmembers(module)
                            if inspect.isfunction(obj) and hasattr(obj, "_is_tool")
                        ]
                    )

        print("Found tools:", [tool.__name__ for tool in tools])
        return tools

    def register_tools(self, functions: List[Callable]) -> None:
        """Register functions as tools for the AI to use"""
        # Bind kernel to functions that need it
        bound_functions = [
            (
                bind_kernel(f, self.jupyter_kernel)
                if "kernel" in inspect.signature(f).parameters
                else f
            )
            for f in functions
        ]
        self.tools = [schema(f) for f in bound_functions]
        self.tool_choice = "auto" if self.tools else "none"
        self.available_functions = {f.__name__: f for f in bound_functions}

    async def execute_tool(self, tool_name: str, args: dict) -> dict:
        """Execute a tool by name with the given arguments"""
        start_time = datetime.now()
        try:
            if tool_name not in self.available_functions:
                error_result = {"error": f"Tool {tool_name} not found"}
                self.tool_history.append(
                    {
                        "tool": tool_name,
                        "arguments": args,
                        "result": error_result,
                        "timestamp": start_time,
                        "duration": (datetime.now() - start_time).total_seconds(),
                        "success": False,
                    }
                )
                return error_result

            func = self.available_functions[tool_name]
            # For the Python tool, handle the args differently
            if tool_name == "python" and isinstance(args, dict) and "code" not in args:
                # If args is a string, use it directly as code
                if isinstance(args.get("args"), str):
                    args = {"code": args["args"]}
                # If args is a dict but doesn't have 'code', convert the entire args to code
                else:
                    args = {"code": str(args["args"])}

            # Execute the function
            result = (
                await func(**args)
                if asyncio.iscoroutinefunction(func)
                else func(**args)
            )

            # Record successful execution
            self.tool_history.append(
                {
                    "tool": tool_name,
                    "arguments": args,
                    "result": result,
                    "timestamp": start_time,
                    "duration": (datetime.now() - start_time).total_seconds(),
                    "success": True,
                }
            )
            return result

        except Exception as e:
            error_result = {"error": str(e)}
            # Record failed execution
            self.tool_history.append(
                {
                    "tool": tool_name,
                    "arguments": args,
                    "result": error_result,
                    "timestamp": start_time,
                    "duration": (datetime.now() - start_time).total_seconds(),
                    "success": False,
                }
            )
            return error_result
