import requests
from requests import HTTPError
from datetime import datetime
from datetime import timedelta
import geocoder
from pydantic import create_model
import inspect
import json
from inspect import Parameter
from typing import Any, Callable, List, Dict
import asyncio


# Add the new tool management classes/functions
def schema(f: Callable) -> Dict[str, Any]:
    """Create a function schema compatible with OpenAI's tool format"""
    kw = {
        n: (o.annotation, ... if o.default == Parameter.empty else o.default)
        for n, o in inspect.signature(f).parameters.items()
    }
    s = create_model(f"Input for `{f.__name__}`", **kw).model_json_schema()
    # Remove the title and definitions which aren't needed in the function schema
    s.pop("title", None)
    s.pop("definitions", None)
    return {
        "type": "function",
        "name": f.__name__,
        "description": f.__doc__ or "",
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


class ToolManager:
    def __init__(self):
        self.tools: List[Dict[str, Any]] = []
        self.available_functions: Dict[str, Callable] = {}
        self.tool_choice = "none"

    def get_available_tools(self) -> List[Callable]:
        """Return list of all available tool functions"""
        # Return all the tool functions defined in this module
        return [
            sum,
            geocode,
            weather,
            get_todays_date,
            get_tomorrows_date,
            get_current_time,
            user_location,
        ]

    def register_tools(self, functions: List[Callable]) -> None:
        """Register functions as tools for the AI to use"""
        self.tools = [schema(f) for f in functions]
        self.tool_choice = "auto" if self.tools else "none"
        self.available_functions = {f.__name__: f for f in functions}

    async def execute_tool(self, tool_name: str, args: dict) -> dict:
        """Execute a tool by name with the given arguments"""
        try:
            # Get the function directly from available_functions
            if tool_name not in self.available_functions:
                return {"error": f"Tool {tool_name} not found"}

            func = self.available_functions[tool_name]

            # Execute the function with the provided arguments
            result = (
                await func(**args)
                if asyncio.iscoroutinefunction(func)
                else func(**args)
            )
            return result  # Return the result directly, don't wrap it

        except Exception as e:
            return {"error": str(e)}


# Your existing tool functions
def sum(a: int, b: int = 1):
    "Adds a + b"
    return a + b


def geocode(city_name: str):
    "Geocodes a city name into latitude and longitude data"
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=10&language=en&format=json"

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")
    else:
        return response.json()


def weather(latitude: float, longitude: float):
    "Returns the weather conditions for a given latitude and longitude"
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,is_day,precipitation,rain,showers,snowfall&timezone=America%2FChicago"

    try:
        response = requests.get(url)
        response.raise_for_status()
    except HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")
    else:
        return response.json()


def get_todays_date():
    "Gets today's date"
    return datetime.now().strftime("%B %d, %Y")


def get_tomorrows_date():
    "Gets tomorrow's date"
    now = datetime.now()
    tomorrow = now + timedelta(days=1)
    return tomorrow.strftime("%B %d, %Y")


def get_current_time():
    "Gets the current time"
    return datetime.now().strftime("%m/%d/%Y %I:%M %p")


def user_location():
    "Guesses the users current location in the format of <city>, <state> based on IP address. But should always ask the user when scheduling."
    g = geocoder.ip("me")
    city = g.city
    state = g.state
    return f"{city}, {state}"
