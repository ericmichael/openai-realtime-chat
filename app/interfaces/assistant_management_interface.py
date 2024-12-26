import gradio as gr
from app.core.assistant_manager import AssistantManager
from app.core.websocket import WebSocketManager
from app.utils.magic_variables import magic_manager


def create_assistant_management_interface(
    assistant_manager: AssistantManager,
    ws_manager: WebSocketManager,
    assistant_template: gr.Dropdown,
):
    """Create the Assistant Management interface tab

    Args:
        assistant_manager: The Assistant Manager instance to use
        ws_manager: The WebSocket Manager instance for tool access
        assistant_template: The assistant template dropdown from the main interface
    """
    gr.Markdown("Create and manage AI assistants with custom voices and tools")

    with gr.Row():
        with gr.Column(scale=1):
            assistant_list = gr.Dropdown(
                choices=["None"] + list(assistant_manager.get_all_assistants().keys()),
                label="Select Assistant",
                value="None",
            )

            new_assistant_btn = gr.Button("Create New Assistant", variant="primary")
            delete_btn = gr.Button("Delete Selected Assistant", variant="stop")

        with gr.Column(scale=2):
            assistant_name = gr.Textbox(
                label="Assistant Name",
                placeholder="Enter assistant name...",
                interactive=True,
            )
            assistant_voice = gr.Dropdown(
                choices=assistant_manager.available_voices,
                label="Voice",
                value="alloy",
            )

            # Replace static markdown with dynamic documentation
            gr.Markdown(magic_manager.get_documentation())

            assistant_instructions = gr.Textbox(
                label="Instructions",
                placeholder="Enter the instructions for this assistant...",
                lines=5,
                interactive=True,
            )
            assistant_tools = gr.CheckboxGroup(
                choices=[
                    f.__name__ for f in ws_manager.tool_manager.get_available_tools()
                ],
                label="Tools",
                info="Select tools to enable for this assistant",
            )
            save_btn = gr.Button("Save Changes", variant="primary")

    def create_new_assistant():
        """Reset the assistant form for creating a new assistant"""
        return (
            "",  # assistant_name
            "alloy",  # assistant_voice
            "",  # assistant_instructions
            [],  # assistant_tools
            gr.update(interactive=False),  # delete_btn
        )

    def load_assistant_details(assistant_name):
        if assistant_name == "None":
            return "", "alloy", "", [], gr.update(interactive=False)

        assistant = assistant_manager.get_assistant(assistant_name)
        return (
            assistant_name,
            assistant.get("voice", "alloy"),
            assistant.get("instructions", ""),
            assistant.get("tools", []),
            gr.update(
                interactive=not assistant_manager.is_default_assistant(assistant_name)
            ),
        )

    def save_assistant_changes(name, voice, instructions, tools):
        data = {
            "voice": voice,
            "instructions": instructions,
            "tools": tools,
        }

        if name in assistant_manager.get_all_assistants():
            assistant_manager.edit_assistant(name, data)
        else:
            assistant_manager.add_assistant(name, data)

        choices = ["None"] + list(assistant_manager.get_all_assistants().keys())
        return (
            gr.update(choices=choices, value=name),
            gr.update(choices=list(assistant_manager.get_all_assistants().keys())),
            "Assistant saved successfully!",
        )

    def delete_current_assistant(name):
        if assistant_manager.is_default_assistant(name):
            return (
                gr.update(
                    choices=["None"]
                    + list(assistant_manager.get_all_assistants().keys())
                ),
                "None",
                "alloy",
                "",
                [],
                gr.update(interactive=False),
                gr.update(choices=list(assistant_manager.get_all_assistants().keys())),
                "Cannot delete default assistants!",
            )

        assistant_manager.delete_assistant(name)
        choices = ["None"] + list(assistant_manager.get_all_assistants().keys())
        return (
            gr.update(choices=choices),
            "None",
            "alloy",
            "",
            [],
            gr.update(interactive=False),
            gr.update(choices=list(assistant_manager.get_all_assistants().keys())),
            "Assistant deleted successfully!",
        )

    # Wire up all the event handlers
    assistant_list.change(
        fn=load_assistant_details,
        inputs=[assistant_list],
        outputs=[
            assistant_name,
            assistant_voice,
            assistant_instructions,
            assistant_tools,
            delete_btn,
        ],
    )

    new_assistant_btn.click(
        fn=create_new_assistant,
        outputs=[
            assistant_name,
            assistant_voice,
            assistant_instructions,
            assistant_tools,
            delete_btn,
        ],
    )

    save_btn.click(
        fn=save_assistant_changes,
        inputs=[
            assistant_name,
            assistant_voice,
            assistant_instructions,
            assistant_tools,
        ],
        outputs=[
            assistant_list,
            assistant_template,
            gr.Textbox(visible=False),
        ],
    )

    delete_btn.click(
        fn=delete_current_assistant,
        inputs=[assistant_name],
        outputs=[
            assistant_list,
            assistant_name,
            assistant_voice,
            assistant_instructions,
            assistant_tools,
            delete_btn,
            assistant_template,
            gr.Textbox(visible=False),
        ],
    )
