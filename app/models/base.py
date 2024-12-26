from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
from app.config import Config

# Load environment variables from .env file
load_dotenv()

Base = declarative_base()

# Create engine using config
engine = create_engine(Config.get_database_url())
Session = sessionmaker(bind=engine)

# Create vector extension
with engine.connect() as connection:
    connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    connection.commit()

# Register vector type after creating extension
with engine.connect() as connection:
    connection.execute(text("SELECT '[1,2,3]'::vector"))
    connection.commit()
