import os
import sys
import json
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from services.assistant_service import AssistantService


def migrate_json_to_db():
    with open("assistants.json", "r") as f:
        assistants = json.load(f)

    for name, data in assistants.items():
        AssistantService.create_assistant(
            name=name,
            instructions=data["instructions"],
            voice=data["voice"],
            tools=data.get("tools", []),
        )


if __name__ == "__main__":
    migrate_json_to_db()
