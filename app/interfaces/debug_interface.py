import gradio as gr
from app.core.websocket import WebSocketManager


def create_debug_interface(ws_manager: WebSocketManager):
    """Create the Debug interface tab

    Args:
        ws_manager: The WebSocket manager instance to use for accessing logs
    """
    with gr.Tab("Debug"):
        gr.Markdown("View WebSocket events and messages for debugging")

        debug_output = gr.TextArea(
            label="Event Logs",
            interactive=False,
            lines=20,
            value="No events logged yet.",
        )
        refresh_btn = gr.Button("Refresh Logs")

        def update_logs():
            return ws_manager.get_logs()

        refresh_btn.click(fn=update_logs, outputs=[debug_output])
