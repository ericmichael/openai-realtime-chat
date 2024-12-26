from sqlalchemy import Column, Integer, String, Text, Table, ForeignKey
from sqlalchemy.orm import relationship
from .base import Base
from .concerns.vectorizable import vectorizable


# Define the Edge model class
class Edge(Base):
    __tablename__ = "edges"

    source_id = Column(Integer, ForeignKey("nodes.id"), primary_key=True)
    target_id = Column(Integer, ForeignKey("nodes.id"), primary_key=True)
    relationship_type = Column(String(255), primary_key=True)

    def __init__(self, source_id, target_id, relationship_type):
        self.source_id = source_id
        self.target_id = target_id
        self.relationship_type = relationship_type


@vectorizable
class Node(Base):
    __tablename__ = "nodes"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    node_type = Column(
        String(255), nullable=False
    )  # This will serve as our type discriminator
    description = Column(Text)

    # Relationships
    outgoing_edges = relationship(
        "Node",
        secondary="edges",
        primaryjoin=id == Edge.source_id,
        secondaryjoin=id == Edge.target_id,
        backref="incoming_edges",
    )

    def __init__(self, name, node_type, description=None):
        self.name = name
        self.node_type = node_type
        self.description = description

    def add_edge(self, target_node, relationship_type):
        """Add a directed edge to another node"""
        edge = Edge(
            source_id=self.id,
            target_id=target_node.id,
            relationship_type=relationship_type,
        )
        return edge

    @classmethod
    def semantic_search(cls, query, session, limit=5):
        """Search nodes by semantic similarity"""
        return cls.embedding_search(
            query=query, field_name="description", limit=limit, session=session
        )


# Configure vectorization for the description field
Node.vectorizes(
    "description",
    model="text-embedding-3-small",
    template="node_description.j2",  # We'll create this template
)
