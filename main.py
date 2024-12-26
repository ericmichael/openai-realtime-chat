# Third-party imports
import gradio as gr

# Config imports
from app.config import Config

# Core service imports
from app.core.assistant_manager import AssistantManager
from app.core.websocket import WebSocketManager

# Service imports
from app.services.document import DocumentService
from app.services.knowledge_graph import KnowledgeGraphService

# Interface imports
from app.interfaces import (
    create_assistant_management_interface,
    create_document_interface,
    create_knowledge_graph_management_interface,
    create_knowledge_graph_search_interface,
    create_vector_embeddings_interface,
    create_voice_chat_interface,
)

# Utility imports
from app.utils.static import load_static_file
from app.themes import TokyoNightTheme, CyberPunkTheme

ws_manager = WebSocketManager()
assistant_manager = AssistantManager()
document_service = DocumentService()
knowledge_graph_service = KnowledgeGraphService()


with gr.Blocks(
    theme=CyberPunkTheme(),
    head=f"""
    <script>{load_static_file('shortcuts.js')}</script>
    <style>{load_static_file('styles.css')}</style>
    """,
) as demo:
    gr.Markdown(
        """
        <div style='text-align: center; padding: 1rem;'>
            <h1 style='color: rgb(34, 211, 238);'>🤖 4341: AI-Powered Applications</h1>
            <p style='font-size: 1.1em; color: rgb(156, 163, 175);'>AI Playground 🧪</p>
        </div>
    """
    )

    # Main Tab Group
    with gr.Tab("💬 Main"):
        # Voice Chat Tab
        with gr.Tab("🎙️ Voice Chat"):
            assistant_template = create_voice_chat_interface(
                ws_manager, assistant_manager
            )

    # Backend Tab Group
    with gr.Tab("⚙️ System"):
        # Vector Embeddings Tab
        with gr.Tab("🔍 Vector Search"):
            create_vector_embeddings_interface()

        # Document Management Tab
        with gr.Tab("📄 Documents"):
            create_document_interface()

        # Assistant Management Tab
        with gr.Tab("🤖 Assistants"):
            create_assistant_management_interface(
                assistant_manager, ws_manager, assistant_template
            )

        # Knowledge Graph Tab Group
        with gr.Tab("🧠 Knowledge Graph"):
            # Knowledge Graph Management Tab
            with gr.Tab("📝 Manage"):
                create_knowledge_graph_management_interface(knowledge_graph_service)

            # Knowledge Graph Search Tab
            with gr.Tab("🔎 Search"):
                create_knowledge_graph_search_interface(knowledge_graph_service)

if __name__ == "__main__":
    demo.launch()
