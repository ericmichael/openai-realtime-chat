from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime
from sqlalchemy.orm import relationship
from .base import Base
from .concerns.vectorizable import vectorizable


@vectorizable
class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    published = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Add polymorphic identity
    __mapper_args__ = {"polymorphic_identity": "Document"}

    # Define the relationship here instead of as a backref
    vector_embeddings = relationship(
        "VectorEmbedding",
        primaryjoin="and_(Document.id==VectorEmbedding.vectorizable_id, "
        'VectorEmbedding.vectorizable_type=="Document")',
        back_populates="document",
        cascade="all, delete-orphan",
    )

    @classmethod
    def find_by_title(cls, session, title):
        return session.query(cls).filter_by(title=title).first()

    @classmethod
    def all(cls, session):
        return session.query(cls).all()

    def to_dict(self):
        return {
            "title": self.title,
            "content": self.content,
            "published": self.published,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def create(cls, session, **kwargs):
        print(f"Creating document with kwargs: {kwargs}")  # Debug
        instance = cls(**kwargs)
        session.add(instance)
        session.commit()
        print(f"Document created with ID: {instance.id}")  # Debug
        instance.sync_embedding()
        print(
            f"Vector embeddings synced. Count: {len(instance.vector_embeddings)}"
        )  # Debug
        return instance

    def update(self, session, **kwargs):
        print(f"Updating document {self.id} with kwargs: {kwargs}")  # Debug
        for key, value in kwargs.items():
            setattr(self, key, value)
        session.commit()
        print(f"Document updated")  # Debug
        self.sync_embedding()
        print(
            f"Vector embeddings synced. Count: {len(self.vector_embeddings)}"
        )  # Debug
        return self

    @classmethod
    def find_or_create_by(cls, session, **kwargs):
        instance = session.query(cls).filter_by(**kwargs).first()
        if not instance:
            instance = cls.create(session, **kwargs)
        return instance


Document.vectorizes("content", model="text-embedding-3-small")
