from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    JSON,
    DateTime,
    Text,
    CheckConstraint,
    ForeignKey,
    Index,
    select,
    func,
    literal,
    asc,
    desc,
    text,
    type_coerce,
    Float,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from .base import Base
from openai import OpenAI
from typing import Optional


class VectorEmbedding(Base):
    __tablename__ = "vector_embeddings"

    id = Column(Integer, primary_key=True)
    vectorizable_type = Column(String(255), nullable=False)
    vectorizable_id = Column(Integer, nullable=False)
    field_name = Column(String(255), nullable=False)
    vector = Column(Vector(1536), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    embedding_metadata = Column(JSONB, default={}, nullable=False)
    chunk_index = Column(Integer)
    total_chunks = Column(Integer)
    content = Column(Text)

    # Define relationships without backrefs
    document = relationship(
        "Document",
        foreign_keys=[vectorizable_id],
        primaryjoin="and_(VectorEmbedding.vectorizable_id==Document.id, "
        "VectorEmbedding.vectorizable_type=='Document')",
    )

    node = relationship(
        "Node",
        foreign_keys=[vectorizable_id],
        primaryjoin="and_(VectorEmbedding.vectorizable_id==Node.id, "
        "VectorEmbedding.vectorizable_type=='Node')",
    )

    __table_args__ = (
        CheckConstraint(
            "chunk_index < total_chunks", name="check_chunk_index_within_bounds"
        ),
        CheckConstraint("chunk_index >= 0", name="check_chunk_index_positive"),
        # Add indexes
        Index(
            "index_vector_embeddings_on_metadata",
            embedding_metadata,
            postgresql_using="gin",
        ),
        Index(
            "index_vector_embeddings_on_vector",
            vector,
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"vector": "vector_l2_ops"},
        ),
        Index(
            "index_vector_embeddings_unique_chunks",
            vectorizable_type,
            vectorizable_id,
            field_name,
            chunk_index,
            unique=True,
        ),
        Index(
            "index_vector_embeddings_on_vectorizable",
            vectorizable_type,
            vectorizable_id,
        ),
    )

    @classmethod
    def similarity_score_sql(cls, vector, metric):
        """Generate SQL for calculating similarity scores using SQLAlchemy constructs"""
        # Debug input
        print(f"\n=== Similarity Score Calculation ===")
        print(f"Metric: {metric}")
        print(f"Vector length: {len(vector)}")

        # Use literal() to create a properly typed vector parameter
        vector_param = literal(vector, type_=Vector)

        # Calculate base similarity using vector operator
        distance_op = cls._distance_operator(metric)
        base_score = cls.vector.op(distance_op)(vector_param)

        # Debug the operator being used
        print(f"Distance operator: {distance_op}")

        # Normalize/transform the score based on metric
        if metric == "cosine":
            # Convert cosine distance to similarity
            similarity_score = 1 - func.cast(base_score, Float)
        elif metric == "l2":
            # Convert L2 distance to similarity
            similarity_score = 1 / (1 + func.cast(base_score, Float))
        elif metric == "inner":
            similarity_score = func.cast(base_score, Float)
        else:
            raise ValueError(f"Unsupported distance metric: {metric}")

        return similarity_score.label("similarity_score")

    @classmethod
    def threshold_condition(cls, vector, metric, threshold):
        """Generate threshold condition using SQLAlchemy constructs"""
        # Ensure vector is in the correct format
        if isinstance(vector, str):
            # Remove any 'vector' type cast and brackets if present
            vector = vector.replace("::vector", "").strip("[]").split(",")
            vector = [float(x) for x in vector]

        # Use the vector directly with literal()
        vector_param = literal(vector, type_=Vector)

        operator = "<=" if metric == "l2" else ">="
        vector_distance = cls.vector.op(cls._distance_operator(metric))(vector_param)

        return vector_distance.op(operator)(threshold)

    @classmethod
    def order_clause(cls, metric):
        """Generate order clause for vector similarity"""
        return desc("similarity_score")

    @staticmethod
    def _distance_operator(metric):
        """Get the appropriate PostgreSQL operator for the distance metric"""
        operators = {
            "l2": "<->",  # Euclidean distance
            "inner": "<#>",  # Negative inner product
            "cosine": "<=>",  # Cosine distance
        }
        if metric not in operators:
            raise ValueError(f"Unsupported distance metric: {metric}")
        return operators[metric]

    @property
    def chunked(self):
        """Check if the embedding is chunked"""
        return self.total_chunks is not None and self.total_chunks > 1

    @classmethod
    def find_by_field(cls, session, field_name):
        """Find embeddings by field name"""
        return session.query(cls).filter_by(field_name=field_name).all()

    @classmethod
    def ordered_chunks(cls, session):
        """Get embeddings ordered by chunk index"""
        return session.query(cls).order_by(cls.chunk_index)

    @property
    def vectorizable(self):
        """Get the parent record based on vectorizable_type"""
        if self.vectorizable_type == "Document":
            return self.document
        # Add more conditions here for other vectorizable types
        return None

    @classmethod
    def embedding_search(
        cls,
        query: str,
        *,
        field_name: str = None,
        metric: str = "cosine",
        limit: Optional[int] = None,
        threshold: Optional[float] = None,
        combine_chunks: bool = True,
        session=None,
    ):
        """Search across all vector embeddings.

        Args:
            query: The search query text
            field_name: Optional field name to filter by
            metric: Similarity metric ('cosine', 'l2', or 'inner')
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold
            combine_chunks: Whether to combine chunks from the same document
            session: SQLAlchemy session
        """
        if not session:
            raise ValueError("Session is required")

        # Generate embedding for query
        client = OpenAI()
        response = client.embeddings.create(
            input=query,
            model="text-embedding-3-small",
            encoding_format="float",
        )
        embedding = response.data[0].embedding

        # Calculate similarity score
        similarity_score = cls.similarity_score_sql(embedding, metric)

        # Build base query
        if combine_chunks:
            # Group by the document and select best matching chunk
            base_query = session.query(
                cls.vectorizable_type,
                cls.vectorizable_id,
                cls.field_name,
                func.min(similarity_score).label("score"),
                func.jsonb_agg(
                    func.jsonb_build_object(
                        "content",
                        cls.content,
                        "chunk_index",
                        cls.chunk_index,
                        "total_chunks",
                        cls.total_chunks,
                        "metadata",
                        cls.embedding_metadata,
                    )
                ).label("chunks"),
            )

            if field_name:
                base_query = base_query.filter(cls.field_name == field_name)

            base_query = base_query.group_by(
                cls.vectorizable_type, cls.vectorizable_id, cls.field_name
            )

            if threshold:
                base_query = base_query.having(func.min(similarity_score) <= threshold)

            base_query = base_query.order_by(func.min(similarity_score))
        else:
            # Return individual chunks as separate results
            base_query = session.query(
                cls.vectorizable_type,
                cls.vectorizable_id,
                cls.field_name,
                cls.content,
                cls.chunk_index,
                cls.total_chunks,
                cls.embedding_metadata,
                similarity_score.label("score"),
            )

            if field_name:
                base_query = base_query.filter(cls.field_name == field_name)

            if threshold:
                base_query = base_query.filter(similarity_score <= threshold)

            base_query = base_query.order_by(similarity_score)

        if limit:
            base_query = base_query.limit(limit)

        return base_query.all()
