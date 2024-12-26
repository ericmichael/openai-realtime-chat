from pathlib import Path
import os
import sys
from dotenv import load_dotenv

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

load_dotenv()


class Config:
    # Base paths
    BASE_DIR = PROJECT_ROOT
    STATIC_DIR = BASE_DIR / "static"

    # Database settings
    POSTGRES_USER = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST")
    POSTGRES_DB = os.getenv("POSTGRES_DB")

    # OpenAI settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Application settings
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

    @classmethod
    def get_database_url(cls) -> str:
        return f"postgresql://{cls.POSTGRES_USER}:{cls.POSTGRES_PASSWORD}@{cls.POSTGRES_HOST}/{cls.POSTGRES_DB}"

    @classmethod
    def get_static_file_path(cls, filename: str) -> Path:
        return cls.STATIC_DIR / filename
