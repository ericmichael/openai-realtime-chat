from app.core.tools import tool
from app.models.base import Session
from app.services.knowledge_graph import KnowledgeGraphService


@tool
def search_knowledge_graph(query: str, max_hops: int = 2):
    """
    Search the knowledge graph using semantic search and explore relationships.

    Args:
        query: The search text to find relevant nodes
        max_hops: Maximum number of relationship hops to explore (default: 2)
    """

    with Session() as session:
        try:
            initial_results, relationships = (
                KnowledgeGraphService.search_knowledge_graph(
                    query=query, max_hops=max_hops, session=session
                )
            )

            # Format the results in a more structured way
            formatted_results = {
                "query": query,
                "total_nodes": len(initial_results),
                "total_relationships": len(relationships),
                "nodes": [
                    {
                        "name": result["Node"],
                        "type": result["Type"],
                        "description": result["Description"],
                        "relevance_score": float(result["Relevance"]),
                    }
                    for result in initial_results
                ],
                "relationships": [
                    {
                        "source": rel["Source"],
                        "relationship": rel["Relationship"],
                        "target": rel["Target"],
                    }
                    for rel in relationships
                ],
            }

            return formatted_results

        except Exception as e:
            print(f"Knowledge graph search error: {str(e)}")
            return {"error": str(e)}
