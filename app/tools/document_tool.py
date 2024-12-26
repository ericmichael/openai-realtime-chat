from app.core.tools import tool
from app.models.base import Session
from app.models.document import Document


@tool
def search_documents(
    query: str, limit: int = 5, metric: str = "cosine", threshold: float = None
):
    """
    Search documents using vector similarity. Returns documents most similar to the query text.

    Args:
        query: The search text
        limit: Maximum number of results to return (default: 5)
        metric: Similarity metric to use ('cosine', 'l2', or 'inner')
        threshold: Optional similarity threshold (0-1)
    """

    with Session() as session:
        try:
            results = Document.embedding_search(
                query=query,
                field_name="content",
                metric=metric,
                limit=limit,
                threshold=threshold,
                combine_chunks=True,
                session=session,
            )

            # Format results for return
            formatted_results = []
            for document, score in results:
                formatted_results.append(
                    {
                        "id": document.id,
                        "title": document.title,
                        "content": document.content,
                        "similarity_score": float(score),
                        "published": document.published,
                        "created_at": document.created_at.isoformat(),
                        "updated_at": document.updated_at.isoformat(),
                    }
                )

            return {
                "query": query,
                "total_results": len(formatted_results),
                "results": formatted_results,
            }

        except Exception as e:
            print(f"Search error: {str(e)}")
            return {"error": str(e)}
