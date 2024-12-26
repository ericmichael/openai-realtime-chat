from datetime import datetime
from typing import Callable, Dict, Any
from functools import wraps
import geocoder


class MagicVariableManager:
    def __init__(self):
        self.magic_variables: Dict[str, Callable] = {}

    def register_variable(self, name: str, func: Callable) -> None:
        """Register a magic variable function"""
        self.magic_variables[name] = func

    def process_instructions(self, instructions: str) -> str:
        """Process instructions and replace magic variables with their values"""
        processed = instructions
        for var_name, var_func in self.magic_variables.items():
            placeholder = "{" + var_name + "}"
            if placeholder in processed:
                try:
                    value = var_func()
                    processed = processed.replace(placeholder, str(value))
                except Exception as e:
                    print(f"Error processing magic variable {var_name}: {e}")
                    processed = processed.replace(
                        placeholder, f"<{var_name} unavailable>"
                    )
        return processed

    def get_documentation(self) -> str:
        """Generate markdown documentation for all registered magic variables"""
        docs = ["Available magic variables (will be replaced with real values):"]

        for var_name, func in self.magic_variables.items():
            description = func.__doc__ or "No description available"
            # Clean up the docstring
            description = description.strip()
            # Add example if the function can be safely called
            try:
                example = func()
                docs.append(f"- `{{{var_name}}}` - {description} (e.g., {example})")
            except:
                docs.append(f"- `{{{var_name}}}` - {description}")

        return "\n".join(docs)


# Create a global magic variable manager
magic_manager = MagicVariableManager()


def magic_variable(name: str) -> Callable:
    """Decorator to mark a function as a magic variable"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        magic_manager.register_variable(name, func)
        return wrapper

    return decorator


# Define magic variables using the decorator
@magic_variable("todays_date")
def get_todays_date() -> str:
    """Returns today's date in Month DD, YYYY format"""
    return datetime.now().strftime("%B %d, %Y")


@magic_variable("current_time")
def get_current_time() -> str:
    """Returns the current time in HH:MM AM/PM format"""
    return datetime.now().strftime("%I:%M %p")


@magic_variable("user_location")
def get_user_location() -> str:
    """Returns the user's location based on IP address"""
    g = geocoder.ip("me")
    return f"{g.city}, {g.state}"
