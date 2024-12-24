import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from sqlalchemy import create_engine, inspect
from sqlalchemy_utils import database_exists, create_database
from models.base import Base, engine
from models.assistant import Assistant
from dotenv import load_dotenv

load_dotenv()


def init_db():
    # Create database if it doesn't exist
    if not database_exists(engine.url):
        create_database(engine.url)
        print(f"Created database {engine.url}")

    # # Create vector extension if it doesn't exist
    # with engine.connect() as conn:
    #     conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    #     conn.commit()

    # Create all tables
    Base.metadata.create_all(engine)
    print("Created all database tables")

    # Verify table creation
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"Available tables: {tables}")


if __name__ == "__main__":
    init_db()
