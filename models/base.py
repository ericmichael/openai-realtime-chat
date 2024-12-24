from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

Base = declarative_base()
engine = create_engine(
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}/{os.getenv('POSTGRES_DB')}"
)

Session = sessionmaker(bind=engine)

# Create the vector extension if it doesn't exist
with engine.connect() as connection:
    connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    connection.commit()

# Register vector type after creating extension
with engine.connect() as connection:
    connection.execute(text("SELECT '[1,2,3]'::vector"))
    connection.commit()
