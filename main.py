import gradio as gr
from app.config import Config
from app.core.assistant_manager import AssistantManager
from app.core.websocket import WebSocketManager
from app.services.document import DocumentService
from app.services.knowledge_graph import KnowledgeGraphService
from app.interfaces.document_interface import create_document_interface
from app.interfaces.tool_history_interface import create_tool_history_interface
from app.interfaces.knowledge_graph_search_interface import (
    create_knowledge_graph_search_interface,
)
from app.interfaces.assistant_management_interface import (
    create_assistant_management_interface,
)
from app.interfaces.vector_embeddings_interface import (
    create_vector_embeddings_interface,
)
from app.interfaces.debug_interface import create_debug_interface
from app.interfaces.knowledge_graph_management_interface import (
    create_knowledge_graph_management_interface,
)
from app.interfaces.voice_chat_interface import create_voice_chat_interface
from app.utils.static import load_static_file

ws_manager = WebSocketManager()
assistant_manager = AssistantManager()
document_service = DocumentService()
knowledge_graph_service = KnowledgeGraphService()


# Updated Gradio Interface
with gr.Blocks(
    head=f"""
    <script>{load_static_file('shortcuts.js')}</script>
    <style>{load_static_file('styles.css')}</style>
    """
) as demo:
    gr.Markdown("<h1 style='text-align: center;'>OpenAI Realtime API</h1>")
    with gr.Row(elem_classes="status-container"):
        status_indicator = gr.Markdown("🔴 Disconnected", elem_id="status")
        kernel_status = gr.Markdown("🔴 No Kernel", elem_id="kernel-status")

    # Replace the VoiceChat section with:
    assistant_template = create_voice_chat_interface(ws_manager, assistant_manager)

    # Replace the Debug section with:
    create_debug_interface(ws_manager)

    # Replace the Tool History section with:
    create_tool_history_interface(ws_manager)

    # Replace the Vector Embeddings section with:
    create_vector_embeddings_interface()

    create_document_interface()

    # Replace the Manage Assistants section with:
    create_assistant_management_interface(
        assistant_manager, ws_manager, assistant_template
    )

    # Replace the Manage Knowledge Graph section with:
    create_knowledge_graph_management_interface(knowledge_graph_service)

    # Replace the Search Knowledge Graph section with:
    create_knowledge_graph_search_interface(knowledge_graph_service)


if __name__ == "__main__":
    demo.launch()
