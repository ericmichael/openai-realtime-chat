from services.assistant_service import AssistantService

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
        self.service = AssistantService()

        # Initialize default assistants if they don't exist
        self._initialize_defaults()

    def _initialize_defaults(self):
        """Initialize default assistants in the database if they don't exist"""
        for name, instructions in DEFAULT_INSTRUCTIONS.items():
            assistant = self.service.get_assistant_by_name(name)
            if not assistant:
                self.service.create_assistant(
                    name=name,
                    instructions=instructions,
                    voice="alloy",
                    tools=(
                        []
                        if name != "Technical Expert"
                        else ["get_weather", "get_time"]
                    ),
                )

    def get_all_assistants(self):
        """Get all assistant configurations."""
        return self.service.get_all_assistants()

    def get_assistant(self, name):
        """Get an assistant's configuration by name."""
        assistant = self.service.get_assistant_by_name(name)
        if assistant:
            return assistant.to_dict()
        return {
            "instructions": "You are a helpful assistant.",
            "voice": "alloy",
            "tools": [],
        }

    def add_assistant(self, name, data):
        """Add a new assistant with the given name and data."""
        self.service.create_assistant(
            name=name,
            instructions=data["instructions"],
            voice=data["voice"],
            tools=data.get("tools", []),
        )
        return list(self.get_all_assistants().keys())

    def delete_assistant(self, name):
        """Delete an assistant by name, if it's not a default assistant."""
        if name not in DEFAULT_INSTRUCTIONS:
            self.service.delete_assistant(name)
        return list(self.get_all_assistants().keys())

    def edit_assistant(self, name, data):
        """Edit an existing assistant's configuration."""
        self.service.update_assistant(
            name=name,
            instructions=data["instructions"],
            voice=data["voice"],
            tools=data.get("tools", []),
        )
        return list(self.get_all_assistants().keys())

    def is_default_assistant(self, name):
        """Check if an assistant is a default assistant."""
        return name in DEFAULT_INSTRUCTIONS
