import gradio as gr
import pandas as pd
from app.models.base import Session
from app.services.knowledge_graph import KnowledgeGraphService


def create_knowledge_graph_search_interface(
    knowledge_graph_service: KnowledgeGraphService,
):
    """Create the Knowledge Graph Search interface tab

    Args:
        knowledge_graph_service: The Knowledge Graph service instance to use for searches
    """
    gr.Markdown("Search and explore the knowledge graph using natural language queries")

    with gr.Row():
        with gr.Column(scale=1):
            search_input = gr.Textbox(
                label="Search Query",
                placeholder="e.g., 'Who wrote The Lord of the Rings?' or 'What books are in Middle-earth?'",
                lines=3,
            )
            max_hops = gr.Slider(
                minimum=1,
                maximum=5,
                value=2,
                step=1,
                label="Maximum Path Length",
                info="Maximum number of relationship hops to explore",
            )
            search_btn = gr.Button("Search Graph", variant="primary")

        with gr.Column(scale=2):
            initial_results = gr.DataFrame(
                headers=["Node", "Type", "Description", "Relevance"],
                label="Direct Matches",
            )
            relationships = gr.DataFrame(
                headers=["Source", "Relationship", "Target"],
                label="Related Connections",
            )

    def search_knowledge_graph(query: str, max_hops: int):
        """
        Search the knowledge graph using natural language queries
        """
        with Session() as session:
            initial_results, relationships = (
                knowledge_graph_service.search_knowledge_graph(query, max_hops, session)
            )

            if not initial_results:
                return (
                    gr.update(value=[]),
                    gr.update(value=[]),
                )

            return (
                gr.update(value=pd.DataFrame(initial_results)),
                gr.update(value=pd.DataFrame(relationships)),
            )

    # Wire up the search functionality
    search_btn.click(
        fn=search_knowledge_graph,
        inputs=[search_input, max_hops],
        outputs=[initial_results, relationships],
    )
