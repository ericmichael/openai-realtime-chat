import gradio as gr
import pandas as pd
from app.models.base import Session
from app.models.vector_embedding import VectorEmbedding


def format_embeddings(for_update=False):
    """Format all embeddings for display in a DataFrame"""
    with Session() as session:
        embeddings = session.query(VectorEmbedding).all()

    if not embeddings:
        return (
            gr.update(value=pd.DataFrame(), visible=False)
            if for_update
            else pd.DataFrame()
        )

    # Format the embeddings into a list of dictionaries
    formatted_data = [
        {
            "Document": f"{e.vectorizable_type} #{e.vectorizable_id}",
            "Field": e.field_name,
            "Chunk": f"{e.chunk_index + 1}/{e.total_chunks}",
            "Content": (e.content[:100] + "..." if len(e.content) > 100 else e.content),
            "Created": e.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        }
        for e in embeddings
    ]

    # Convert to DataFrame
    df = pd.DataFrame(formatted_data)
    return gr.update(value=df, visible=True) if for_update else df


def create_vector_embeddings_interface():
    """Create the Vector Embeddings interface tab"""
    with gr.Tabs():
        # Search Tab
        with gr.Tab("Search"):
            gr.Markdown("Search through vector embeddings")

            with gr.Row():
                with gr.Column(scale=1):
                    search_query = gr.Textbox(
                        label="Search Query",
                        placeholder="Enter text to search...",
                        lines=3,
                    )
                    search_field = gr.Dropdown(
                        choices=["content"],  # Add more fields as needed
                        label="Search Field",
                        value="content",
                    )
                    search_limit = gr.Slider(
                        minimum=1, maximum=50, value=10, step=1, label="Result Limit"
                    )
                    search_button = gr.Button("Search Embeddings", variant="primary")

                with gr.Column(scale=2):
                    search_results = gr.Dataframe(
                        headers=["Score", "Document Title", "Content Preview"],
                        label="Search Results",
                    )

        # View All Tab
        with gr.Tab("View All"):
            gr.Markdown("All vector embeddings in the system")
            embeddings_display = gr.DataFrame(
                headers=["Document", "Field", "Chunk", "Content", "Created"],
                label="All Embeddings",
                value=format_embeddings(),  # This now returns a DataFrame directly
            )
            refresh_embeddings = gr.Button("Refresh Embeddings")

    def perform_vector_search(query, field_name, limit):
        """Perform vector similarity search and return results"""
        if not query:
            return gr.update(value=pd.DataFrame())

        print(f"\n=== Vector Search ===")
        print(f"Query: {query}")
        print(f"Field: {field_name}")
        print(f"Limit: {limit}")

        with Session() as session:
            results = VectorEmbedding.embedding_search(
                query=query,
                field_name=field_name if field_name != "All Fields" else None,
                limit=int(limit),
                session=session,
            )

            if not results:
                print("No results found")
                return gr.update(value=[], visible=True)

            # Debug the raw results
            print(f"\nFound {len(results)} results:")
            formatted_results = []

            for result in results:
                # Extract values from the SQLAlchemy Row object
                vectorizable_type = result[0]
                vectorizable_id = result[1]
                field = result[2]
                score = result[3]
                chunks = result[4] if len(result) > 4 else None

                print(f"\nDocument: {vectorizable_type} #{vectorizable_id}")
                print(f"Field: {field}")
                print(f"Score: {score:.3f}")
                if chunks:
                    print(f"Chunks: {chunks}")
                print("-" * 50)

                # Format for display
                formatted_results.append(
                    {
                        "Score": f"{score:.3f}",
                        "Document Title": f"{vectorizable_type} #{vectorizable_id}",
                        "Field": field,
                        "Content Preview": (
                            chunks[0]["content"][:100] + "..." if chunks else "N/A"
                        ),
                    }
                )

            df = pd.DataFrame(formatted_results)
            return gr.update(value=df, visible=True)

    # Wire up the search functionality
    search_button.click(
        fn=perform_vector_search,
        inputs=[search_query, search_field, search_limit],
        outputs=[search_results],
    )

    # Wire up the refresh functionality
    refresh_embeddings.click(
        fn=lambda: format_embeddings(for_update=True), outputs=[embeddings_display]
    )
