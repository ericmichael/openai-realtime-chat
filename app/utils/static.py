from app.config import Config


def load_static_file(filename: str) -> str:
    """Load a static file from the static directory."""
    file_path = Config.get_static_file_path(filename)
    with open(file_path, "r") as file:
        return file.read()
