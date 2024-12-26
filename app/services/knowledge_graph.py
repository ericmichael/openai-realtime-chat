from app.models.base import Session
from app.models.node import Node, Edge
from typing import List, Dict, Any, Tuple, Set
from sqlalchemy import and_


class KnowledgeGraphService:
    @staticmethod
    def get_all_nodes():
        with Session() as session:
            nodes = session.query(Node).all()
            return nodes

    @staticmethod
    def get_node_by_id(node_id: int):
        with Session() as session:
            return session.query(Node).filter_by(id=node_id).first()

    @staticmethod
    def create_node(name: str, node_type: str, description: str = None):
        with Session() as session:
            try:
                node = Node(name=name, node_type=node_type, description=description)
                session.add(node)
                session.commit()
                node.sync_embedding()
                return node
            except Exception as e:
                print(f"Error creating node: {e}")
                session.rollback()
                raise

    @staticmethod
    def update_node(node_id: int, name: str, node_type: str, description: str = None):
        with Session() as session:
            try:
                node = session.query(Node).filter_by(id=node_id).first()
                if node:
                    node.name = name
                    node.node_type = node_type
                    node.description = description
                    session.commit()
                    node.sync_embedding()
                    return node
                return None
            except Exception as e:
                print(f"Error updating node: {e}")
                session.rollback()
                raise

    @staticmethod
    def delete_node(node_id: int):
        with Session() as session:
            try:
                node = session.query(Node).filter_by(id=node_id).first()
                if node:
                    session.delete(node)
                    session.commit()
                    return True
                return False
            except Exception as e:
                print(f"Error deleting node: {e}")
                session.rollback()
                raise

    @staticmethod
    def create_edge(source_id: int, target_id: int, relationship_type: str):
        with Session() as session:
            try:
                source = session.query(Node).filter_by(id=source_id).first()
                target = session.query(Node).filter_by(id=target_id).first()
                if source and target:
                    edge = source.add_edge(target, relationship_type)
                    session.add(edge)
                    session.commit()
                    return edge
                return None
            except Exception as e:
                print(f"Error creating edge: {e}")
                session.rollback()
                raise

    @staticmethod
    def semantic_search(
        query: str, session: Session, limit: int = 3
    ) -> List[Tuple[Node, float]]:
        """
        Perform semantic search on nodes using the query.
        Returns list of (node, score) tuples.
        """
        return Node.semantic_search(query, session=session, limit=limit)

    @staticmethod
    def explore_relationships(
        initial_nodes: List[Node], max_hops: int, session: Session
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Explore relationships starting from initial nodes up to max_hops distance.
        Returns tuple of (initial_results, relationships).
        """
        # Format initial results
        initial_data = [
            {
                "Node": node.name,
                "Type": node.node_type,
                "Description": node.description,
                "Relevance": f"{score:.2f}",
            }
            for node, score in initial_nodes
        ]

        # Track all relationships we've seen
        relationships_data = []
        seen_edges: Set[Tuple] = set()

        # Explore relationships up to max_hops
        nodes_to_explore = [(node, 0) for node, _ in initial_nodes]

        while nodes_to_explore:
            current_node, depth = nodes_to_explore.pop(0)

            if depth >= max_hops:
                continue

            # Get outgoing relationships
            outgoing = (
                session.query(Edge, Node)
                .join(Node, Edge.target_id == Node.id)
                .filter(Edge.source_id == current_node.id)
                .all()
            )

            # Get incoming relationships
            incoming = (
                session.query(Edge, Node)
                .join(Node, Edge.source_id == Node.id)
                .filter(Edge.target_id == current_node.id)
                .all()
            )

            # Process relationships
            for edge, related_node in outgoing:
                edge_key = (current_node.id, related_node.id, edge.relationship_type)
                if edge_key not in seen_edges:
                    relationships_data.append(
                        {
                            "Source": current_node.name,
                            "Relationship": edge.relationship_type,
                            "Target": related_node.name,
                        }
                    )
                    seen_edges.add(edge_key)
                    nodes_to_explore.append((related_node, depth + 1))

            for edge, related_node in incoming:
                edge_key = (related_node.id, current_node.id, edge.relationship_type)
                if edge_key not in seen_edges:
                    relationships_data.append(
                        {
                            "Source": related_node.name,
                            "Relationship": edge.relationship_type,
                            "Target": current_node.name,
                        }
                    )
                    seen_edges.add(edge_key)
                    nodes_to_explore.append((related_node, depth + 1))

        return initial_data, relationships_data

    @staticmethod
    def search_knowledge_graph(
        query: str, max_hops: int, session: Session
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Search the knowledge graph using semantic search and relationship exploration.
        Returns tuple of (initial_results, relationships).
        """
        # First, find relevant nodes using semantic search
        results = KnowledgeGraphService.semantic_search(query, session=session, limit=3)

        if not results:
            return [], []

        # Explore relationships from the found nodes
        return KnowledgeGraphService.explore_relationships(results, max_hops, session)
