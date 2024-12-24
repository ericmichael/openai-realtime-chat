from datetime import datetime
from typing import Dict, Any, Optional, Set, Type, Union
import os
from pathlib import Path
from jinja2 import Template
from sqlalchemy import event, func
from sqlalchemy.orm import relationship, Session
from functools import wraps
from openai import OpenAI
from ..vector_embedding import VectorEmbedding


class VectorizableRegistry:
    _models: Set[Type] = set()

    @classmethod
    def register(cls, model: Type) -> None:
        cls._models.add(model)

    @classmethod
    def get_models(cls) -> Set[Type]:
        return cls._models


class VectorizableConfiguration:
    """Configuration mixin for vectorizable models"""

    vector_configurations: Dict[str, Dict] = {}
    registered_templates: Dict[str, str] = {}

    @classmethod
    def vectorizes(
        cls,
        method_name: str,
        *,
        model: str = "text-embedding-3-small",
        auto_sync: bool = True,
        chunking: Optional[Dict] = None,
        template: Optional[str] = None,
    ) -> None:
        """Configure vector embedding for a field"""
        normalized_chunking = cls._normalize_chunking_config(chunking)

        cls.vector_configurations[method_name] = {
            "model": model,
            "auto_sync": auto_sync,
            "chunking": normalized_chunking,
            "template": template,
        }

    @classmethod
    def register_template(cls, path: str, content: str) -> None:
        cls.registered_templates[path] = content

    @classmethod
    def clear_registered_templates(cls) -> None:
        cls.registered_templates = {}

    @staticmethod
    def _normalize_chunking_config(chunking: Optional[Dict]) -> Optional[Dict]:
        if not chunking:
            return None

        return {
            "strategy": chunking.get("strategy", "recursive"),
            "max_tokens": chunking.get("max_tokens", 100),
            "overlap": chunking.get("overlap", 10),
            "metadata": chunking.get("metadata", False),
        }

    def render_template(self, template_path: str, item: Any) -> str:
        """Render a template with the given item"""
        if not isinstance(template_path, str):
            raise ValueError("Template path must be a string")

        if self.registered_templates.get(template_path):
            template_content = self.registered_templates[template_path]
        else:
            template_file = Path("app/views") / f"{template_path}.j2"
            if not template_file.exists():
                raise ValueError(f"Template not found at path: {template_path}.j2")
            template_content = template_file.read_text()

        template = Template(template_content)
        return template.render(item=item)


class VectorizableSync:
    """Sync mixin for vectorizable models"""

    def sync_embedding(self, field_name: Optional[str] = None) -> None:
        """Sync embeddings for one or all fields"""
        # Get the session from the current object
        session = Session.object_session(self)
        if not session:
            raise ValueError("No session found for object")

        if field_name:
            self._sync_single_embedding(field_name)
        else:
            self._sync_all_fields()

        # Commit the changes
        session.commit()

    def queue_embedding_sync(self, field_name: Optional[str] = None) -> None:
        """Queue embedding sync for async processing"""
        if self.should_sync_embedding(field_name):
            # Here you would integrate with your async task system (Celery, etc.)
            pass

    def should_sync_embedding(self, field_name: Optional[str] = None) -> bool:
        """Check if embedding should be synced"""
        if not hasattr(self, "id"):
            return False

        if field_name:
            config = self.vector_configurations.get(field_name)
            return bool(
                config
                and config["auto_sync"]
                and not any(
                    ve.field_name == field_name for ve in self.vector_embeddings
                )
            )
        return any(
            config["auto_sync"]
            and not any(ve.field_name == field for ve in self.vector_embeddings)
            for field, config in self.vector_configurations.items()
        )

    def _sync_single_embedding(self, field_name: str) -> None:
        """Sync embedding for a single field"""
        print(f"Syncing embedding for field: {field_name}")  # Debug
        config = self.vector_configurations.get(field_name)
        if not config:
            print(f"No vector configuration found for {field_name}")  # Debug
            raise ValueError(f"No vector configuration found for {field_name}")

        content = getattr(self, field_name)
        if not content:
            print(f"No content found for field {field_name}")  # Debug
            return

        # Get the session
        session = Session.object_session(self)

        # Delete existing embeddings using the session query
        session.query(VectorEmbedding).filter_by(
            vectorizable_type=self.__class__.__name__,
            vectorizable_id=self.id,
            field_name=field_name,
        ).delete()

        # Clear the relationship
        self.vector_embeddings = [
            ve for ve in self.vector_embeddings if ve.field_name != field_name
        ]

        if config["template"]:
            content = self.render_template(config["template"], self)
            print(f"Rendered template content length: {len(content)}")  # Debug

        if config["chunking"]:
            print("Using chunked embedding")  # Debug
            self._sync_chunked_embedding(content, field_name, config)
        else:
            print("Using single vector embedding")  # Debug
            self._sync_single_vector(content, field_name, config["model"])

    def _sync_chunked_embedding(
        self, content: str, field_name: str, config: Dict
    ) -> None:
        """Sync chunked embeddings"""
        from text_chunker import TextChunker  # Import your chunking implementation

        chunks = TextChunker.chunk(
            content,
            strategy=config["chunking"]["strategy"],
            max_tokens=config["chunking"]["max_tokens"],
            overlap=config["chunking"]["overlap"],
            metadata=config["chunking"]["metadata"],
        )

        for index, chunk in enumerate(chunks):
            chunk_content = chunk["content"] if isinstance(chunk, dict) else chunk
            chunk_metadata = (
                chunk["metadata"] if isinstance(chunk, dict) else {"chunk_index": index}
            )

            vector = self._generate_embedding(chunk_content, config["model"])

            self.vector_embeddings.append(
                VectorEmbedding(
                    vectorizable_type=self.__class__.__name__,
                    vectorizable_id=self.id,
                    field_name=field_name,
                    vector=vector,
                    content=chunk_content,
                    embedding_metadata=chunk_metadata,
                    chunk_index=index,
                    total_chunks=len(chunks),
                )
            )

    def _sync_all_fields(self) -> None:
        """Sync embeddings for all configured fields"""
        for field_name in self.vector_configurations:
            self._sync_single_embedding(field_name)

    def _sync_single_vector(self, content: str, field_name: str, model: str) -> None:
        """Create a single vector embedding"""
        vector = self._generate_embedding(content, model)

        self.vector_embeddings.append(
            VectorEmbedding(
                vectorizable_type=self.__class__.__name__,
                vectorizable_id=self.id,
                field_name=field_name,
                vector=vector,
                content=content,
                embedding_metadata={},
                chunk_index=0,
                total_chunks=1,
            )
        )

    def _generate_embedding(self, content: str, model: str) -> list[float]:
        """Generate vector embedding using OpenAI's API

        Args:
            content: The text to embed
            model: The model name (e.g., "text-embedding-3-small")

        Returns:
            list[float]: The embedding vector
        """
        client = OpenAI()  # Uses OPENAI_API_KEY environment variable by default

        response = client.embeddings.create(
            input=content,
            model=model,
            encoding_format="float",  # Returns raw float values
        )

        return response.data[0].embedding


class VectorizableSearch:
    """Search mixin for vectorizable models"""

    @classmethod
    def embedding_search(
        cls,
        query: str,
        field_name: str,
        *,
        metric: str = "cosine",
        limit: Optional[int] = None,
        threshold: Optional[float] = None,
        combine_chunks: bool = True,
        session=None,
    ):
        """Search for records using vector similarity to the query"""
        if field_name not in cls.vector_configurations:
            raise ValueError(f"Invalid field_name: {field_name}")

        if not session:
            raise ValueError("Session is required")

        # Generate embedding for query
        client = OpenAI()
        response = client.embeddings.create(
            input=query,
            model=cls.vector_configurations[field_name]["model"],
            encoding_format="float",
        )
        embedding = response.data[0].embedding

        # Build base query
        base_query = session.query(cls)

        # Join with vector_embeddings
        base_query = base_query.join(
            VectorEmbedding,
            (VectorEmbedding.vectorizable_id == cls.id)
            & (VectorEmbedding.vectorizable_type == cls.__name__)
            & (VectorEmbedding.field_name == field_name),
        )

        # Add similarity score calculation
        similarity_score = VectorEmbedding.similarity_score_sql(embedding, metric)

        if combine_chunks:
            # Group by the main record and select minimum distance
            base_query = base_query.with_entities(
                cls, func.min(similarity_score).label("score")
            ).group_by(cls.id)

            if threshold:
                base_query = base_query.having(func.min(similarity_score) <= threshold)

            base_query = base_query.order_by(func.min(similarity_score).desc())
        else:
            # Treat each chunk as separate result
            base_query = base_query.with_entities(cls, similarity_score.label("score"))

            if threshold:
                base_query = base_query.filter(similarity_score <= threshold)

            base_query = base_query.order_by(similarity_score.desc())

        if limit:
            base_query = base_query.limit(limit)

        return base_query

    @classmethod
    def find_similar(
        cls,
        record_id: int,
        field_name: str,
        *,
        metric: str = "cosine",
        limit: Optional[int] = None,
        threshold: Optional[float] = None,
        combine_chunks: bool = True,
        session=None,
    ):
        """Find similar records to a given record"""
        if not session:
            raise ValueError("Session is required")

        # Get the embedding for the reference record
        embedding_record = (
            session.query(VectorEmbedding)
            .filter_by(
                vectorizable_type=cls.__name__,
                vectorizable_id=record_id,
                field_name=field_name,
            )
            .first()
        )

        if not embedding_record:
            raise ValueError(f"No embedding found for record {record_id}")

        # Build base query excluding the reference record
        base_query = session.query(cls).filter(cls.id != record_id)

        # Join with vector_embeddings
        base_query = base_query.join(
            VectorEmbedding,
            (VectorEmbedding.vectorizable_id == cls.id)
            & (VectorEmbedding.vectorizable_type == cls.__name__)
            & (VectorEmbedding.field_name == field_name),
        )

        # Add similarity score calculation
        similarity_score = VectorEmbedding.similarity_score_sql(
            embedding_record.vector, metric
        )

        if combine_chunks:
            base_query = base_query.with_entities(
                cls, func.min(similarity_score).label("score")
            ).group_by(cls.id)

            if threshold:
                base_query = base_query.having(func.min(similarity_score) <= threshold)

            base_query = base_query.order_by(func.min(similarity_score))
        else:
            base_query = base_query.with_entities(cls, similarity_score.label("score"))

            if threshold:
                base_query = base_query.filter(similarity_score <= threshold)

            base_query = base_query.order_by(similarity_score)

        if limit:
            base_query = base_query.limit(limit)

        return base_query


def vectorizable(cls):
    """Class decorator to make a model vectorizable"""
    # Register the model
    VectorizableRegistry.register(cls)

    # Add vector_embeddings relationship
    cls.vector_embeddings = relationship(
        "VectorEmbedding", backref="vectorizable", cascade="all, delete-orphan"
    )

    # Mix in the vectorizable functionality
    cls.__bases__ = (
        VectorizableConfiguration,
        VectorizableSync,
        VectorizableSearch,
    ) + cls.__bases__

    # Set up SQLAlchemy events
    @event.listens_for(cls, "before_update")
    def nullify_changed_embeddings(mapper, connection, target):
        for field_name in target.vector_configurations:
            if hasattr(target, f"{field_name}_changed"):
                target.vector_embeddings.filter_by(field_name=field_name).delete()

    # Register after_commit event on the Session instead of the model
    @event.listens_for(Session, "after_commit")
    def queue_sync_after_commit(session):
        for obj in session.new | session.dirty:
            if isinstance(obj, cls):
                obj.queue_embedding_sync()

    return cls
