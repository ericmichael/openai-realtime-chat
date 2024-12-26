from app.models.base import Session, Base, engine
from app.models.node import Node
from app.config import Config


def seed_database():
    print(f"Seeding database {Config.POSTGRES_DB}...")

    # Create all tables
    Base.metadata.create_all(engine)

    # Create session
    session = Session()

    try:
        # Create nodes
        nodes = {
            "tolkien": Node(
                name="J.R.R. Tolkien",
                node_type="Person",
                description="English writer, poet, philologist, and academic. Creator of Middle-earth and author of The Lord of the Rings series.",
            ),
            "lotr": Node(
                name="The Lord of the Rings",
                node_type="Book Series",
                description="An epic high-fantasy novel series that tells the story of a hobbit's quest to destroy a powerful ring.",
            ),
            "fellowship": Node(
                name="The Fellowship of the Ring",
                node_type="Book",
                description="The first volume of The Lord of the Rings, following Frodo's journey from the Shire with the Fellowship.",
            ),
            "return_king": Node(
                name="The Return of the King",
                node_type="Book",
                description="The third and final volume of The Lord of the Rings, concluding the story of the War of the Ring.",
            ),
            "frodo": Node(
                name="Frodo Baggins",
                node_type="Character",
                description="A hobbit of the Shire who inherits the One Ring from his cousin Bilbo Baggins and undertakes the quest to destroy it.",
            ),
            "middle_earth": Node(
                name="Middle-earth",
                node_type="Location",
                description="The fictional setting of Tolkien's fantasy works, including The Lord of the Rings, The Hobbit, and The Silmarillion.",
            ),
        }

        # Add all nodes to session
        for node in nodes.values():
            session.add(node)

        # Flush to get IDs and commit to save nodes
        session.commit()

        if Config.DEBUG:
            print(f"Vector configurations: {nodes['tolkien'].vector_configurations}")

        # Sync embeddings for all nodes
        for node in nodes.values():
            node.sync_embedding()
            print(f"Created embedding for node: {node.name}")

        # Create relationships
        relationships = [
            (nodes["tolkien"], nodes["lotr"], "wrote"),
            (nodes["lotr"], nodes["frodo"], "includes character"),
            (nodes["fellowship"], nodes["lotr"], "is the first volume of"),
            (nodes["return_king"], nodes["lotr"], "is the third volume of"),
            (nodes["frodo"], nodes["middle_earth"], "lives in"),
            (nodes["middle_earth"], nodes["lotr"], "is the setting of"),
        ]

        # Add relationships and commit again
        for source, target, rel_type in relationships:
            edge = source.add_edge(target, rel_type)
            session.add(edge)

        session.commit()

        print("Database seeded successfully!")

        if Config.DEBUG:
            # Example semantic search
            print("\nTesting semantic search:")
            results = Node.semantic_search(
                "Who created the world where hobbits live?", session=session, limit=3
            )

            for node, score in results:
                print(f"\nScore: {score}")
                print(f"Node: {node.name}")
                print(f"Type: {node.node_type}")

    except Exception as e:
        print(f"Error seeding database: {e}")
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    seed_database()
