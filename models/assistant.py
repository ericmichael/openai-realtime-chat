from datetime import datetime
from sqlalchemy import Column, Integer, String, JSON, DateTime
from .base import Base


class Assistant(Base):
    __tablename__ = "assistants"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    instructions = Column(String, nullable=False)
    voice = Column(String(50), nullable=False)
    tools = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @classmethod
    def find_by_name(cls, session, name):
        return session.query(cls).filter_by(name=name).first()

    @classmethod
    def all(cls, session):
        return session.query(cls).all()

    def to_dict(self):
        return {
            "instructions": self.instructions,
            "voice": self.voice,
            "tools": self.tools or [],
        }

    @classmethod
    def create(cls, session, **kwargs):
        instance = cls(**kwargs)
        session.add(instance)
        session.commit()
        return instance

    def update(self, session, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        session.commit()
        return self

    @classmethod
    def find_or_create_by(cls, session, **kwargs):
        instance = session.query(cls).filter_by(**kwargs).first()
        if not instance:
            instance = cls.create(session, **kwargs)
        return instance
