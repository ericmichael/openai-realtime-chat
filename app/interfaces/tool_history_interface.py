import gradio as gr
import json
from app.core.websocket import WebSocketManager


def create_tool_history_interface(ws_manager: WebSocketManager):
    """Create the Tool History interface tab

    Args:
        ws_manager: The WebSocket manager instance to use for tool history
    """
    with gr.Tab("Tool History"):
        gr.Markdown("View history of tool calls and their results")

        def format_tool_history():
            history = ws_manager.tool_manager.tool_history
            if not history:
                return "No tool calls recorded yet."

            markdown = ""
            for entry in history:
                success_icon = "✅" if entry["success"] else "❌"
                markdown += f"### {success_icon} {entry['tool']}\n"
                markdown += (
                    f"**Time:** {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                markdown += f"**Duration:** {entry['duration']:.2f}s\n\n"
                markdown += "**Arguments:**\n```json\n"
                markdown += json.dumps(entry["arguments"], indent=2) + "\n```\n\n"
                markdown += "**Result:**\n```json\n"
                markdown += json.dumps(entry["result"], indent=2) + "\n```\n\n"
                markdown += "---\n\n"
            return markdown

        tool_history = gr.Markdown("No tool calls recorded yet.")
        refresh_history = gr.Button("Refresh History")
        refresh_history.click(fn=format_tool_history, outputs=[tool_history])
