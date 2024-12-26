import json
from app.services.assistant import AssistantService
from app.config import Config


def migrate_json_to_db():
    json_path = Config.BASE_DIR / "assistants.json"
    print(f"Migrating assistants from {json_path}...")

    try:
        with open(json_path, "r") as f:
            assistants = json.load(f)

        for name, data in assistants.items():
            AssistantService.create_assistant(
                name=name,
                instructions=data["instructions"],
                voice=data["voice"],
                tools=data.get("tools", []),
            )
        print("Assistant migration completed successfully!")

    except FileNotFoundError:
        print(f"Error: assistants.json not found at {json_path}")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in assistants.json")
    except Exception as e:
        print(f"Error during migration: {e}")


if __name__ == "__main__":
    migrate_json_to_db()
