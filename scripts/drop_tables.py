import os
import sys
from pathlib import Path
from sqlalchemy.sql import text

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.base import Base, engine


def drop_all_tables():
    print("Dropping all tables...")
    # Drop tables in order of dependencies
    try:
        # First drop vector_embeddings (it depends on other tables)
        Base.metadata.tables["vector_embeddings"].drop(engine, checkfirst=True)

        # Then drop edges (it depends on nodes)
        Base.metadata.tables["edges"].drop(engine, checkfirst=True)

        # Then drop the main tables
        Base.metadata.tables["documents"].drop(engine, checkfirst=True)
        Base.metadata.tables["nodes"].drop(engine, checkfirst=True)

        print("All tables dropped successfully!")
    except Exception as e:
        print(f"Error dropping tables: {e}")
        # Alternative approach: drop all with CASCADE
        print("Attempting to drop all tables with CASCADE...")
        with engine.connect() as conn:
            conn.execute(
                text(
                    "DROP TABLE IF EXISTS vector_embeddings, edges, documents, nodes CASCADE"
                )
            )
            conn.commit()
        print("Tables dropped with CASCADE.")


if __name__ == "__main__":
    drop_all_tables()
