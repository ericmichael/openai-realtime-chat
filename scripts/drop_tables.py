from sqlalchemy.sql import text
from app.models.base import Base, engine
from app.config import Config


def drop_all_tables():
    print(f"Dropping all tables in database {Config.POSTGRES_DB}...")
    try:
        # Drop tables in order of dependencies
        Base.metadata.tables["vector_embeddings"].drop(engine, checkfirst=True)
        Base.metadata.tables["edges"].drop(engine, checkfirst=True)
        Base.metadata.tables["documents"].drop(engine, checkfirst=True)
        Base.metadata.tables["nodes"].drop(engine, checkfirst=True)
        Base.metadata.tables["assistants"].drop(engine, checkfirst=True)

        print("All tables dropped successfully!")
    except Exception as e:
        print(f"Error dropping tables: {e}")
        # Alternative approach: drop all with CASCADE
        print("Attempting to drop all tables with CASCADE...")
        with engine.connect() as conn:
            conn.execute(
                text(
                    """
                    DROP TABLE IF EXISTS 
                        vector_embeddings, 
                        edges, 
                        documents, 
                        nodes,
                        assistants 
                    CASCADE
                    """
                )
            )
            conn.commit()
        print("Tables dropped with CASCADE.")


if __name__ == "__main__":
    drop_all_tables()
