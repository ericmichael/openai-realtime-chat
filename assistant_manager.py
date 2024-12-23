import json
from typing import Dict, Any, List

AVAILABLE_VOICES = [
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "sage",
    "shimmer",
    "verse",
]

DEFAULT_INSTRUCTIONS = {
    "General Assistant": "You are a helpful assistant.",
    "Spanish Language Teacher": "You are a Spanish (MX) language teacher. Help users practice speaking and correct their grammar and pronunciation.",
    "Technical Expert": "You are a technical expert. Provide detailed technical explanations and help debug problems.",
}


class AssistantManager:
    def __init__(self):
        self.available_voices = AVAILABLE_VOICES
        self.assistants: Dict[str, Dict[str, Any]] = {
            "General Assistant": {
                "instructions": DEFAULT_INSTRUCTIONS["General Assistant"],
                "voice": "alloy",
                "tools": [],
            },
            "Spanish Language Teacher": {
                "instructions": DEFAULT_INSTRUCTIONS["Spanish Language Teacher"],
                "voice": "alloy",
                "tools": [],
            },
            "Technical Expert": {
                "instructions": DEFAULT_INSTRUCTIONS["Technical Expert"],
                "voice": "alloy",
                "tools": ["get_weather", "get_time"],
            },
        }
        self.load_assistants()

    def load_assistants(self) -> None:
        """Load assistants from the JSON file, merging with existing defaults."""
        try:
            with open("assistants.json", "r") as f:
                saved_assistants = json.load(f)
                self.assistants.update(saved_assistants)
        except FileNotFoundError:
            self.save_assistants()

    def save_assistants(self) -> None:
        """Save current assistants to the JSON file."""
        with open("assistants.json", "w") as f:
            json.dump(self.assistants, f, indent=2)

    def add_assistant(self, name: str, data: Dict[str, Any]) -> List[str]:
        """
        Add a new assistant with the given name and data.

        Args:
            name: Name of the new assistant
            data: Dictionary containing assistant configuration

        Returns:
            List of all assistant names
        """
        self.assistants[name] = data
        self.save_assistants()
        return list(self.assistants.keys())

    def delete_assistant(self, name: str) -> List[str]:
        """
        Delete an assistant by name, if it's not a default assistant.

        Args:
            name: Name of the assistant to delete

        Returns:
            List of remaining assistant names
        """
        if name in self.assistants and name not in DEFAULT_INSTRUCTIONS:
            del self.assistants[name]
            self.save_assistants()
        return list(self.assistants.keys())

    def edit_assistant(self, name: str, data: Dict[str, Any]) -> List[str]:
        """
        Edit an existing assistant's configuration.

        Args:
            name: Name of the assistant to edit
            data: New configuration data

        Returns:
            List of all assistant names
        """
        if name in self.assistants:
            self.assistants[name] = data
            self.save_assistants()
        return list(self.assistants.keys())

    def get_assistant(self, name: str) -> Dict[str, Any]:
        """
        Get an assistant's configuration by name.

        Args:
            name: Name of the assistant

        Returns:
            Assistant configuration dictionary
        """
        return self.assistants.get(
            name,
            {
                "instructions": "You are a helpful assistant.",
                "voice": "alloy",
                "tools": [],
            },
        )

    def get_all_assistants(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all assistant configurations.

        Returns:
            Dictionary of all assistants and their configurations
        """
        return self.assistants.copy()

    def is_default_assistant(self, name: str) -> bool:
        """
        Check if an assistant is a default assistant.

        Args:
            name: Name of the assistant to check

        Returns:
            True if the assistant is a default assistant, False otherwise
        """
        return name in DEFAULT_INSTRUCTIONS
