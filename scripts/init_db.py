from app.config import Config
from app.models.base import Base, engine
from sqlalchemy_utils import database_exists, create_database


def init_db():
    # Create database if it doesn't exist
    if not database_exists(engine.url):
        create_database(engine.url)
        print(f"Created database {Config.POSTGRES_DB}")

    # Create all tables
    Base.metadata.create_all(engine)
    print("Created all database tables")


if __name__ == "__main__":
    init_db()
