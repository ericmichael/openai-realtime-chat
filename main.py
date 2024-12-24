import io
import json
import base64
from pydub import AudioSegment
import soundfile as sf
import gradio as gr
from dotenv import load_dotenv
import librosa
import numpy as np
from magic_variables import magic_manager
from assistant_manager import AssistantManager, DEFAULT_INSTRUCTIONS
from websocket_manager import WebSocketManager
from services.document_service import DocumentService
from models.base import Session
from models.vector_embedding import VectorEmbedding
import pandas as pd

load_dotenv()

ws_manager = WebSocketManager()
assistant_manager = AssistantManager()
document_service = DocumentService()


def format_embeddings():
    """Format all embeddings for display in a DataFrame"""
    with Session() as session:
        embeddings = session.query(VectorEmbedding).all()

    if not embeddings:
        return gr.update(value=pd.DataFrame(), visible=False)

    # Format the embeddings into a list of dictionaries
    formatted_data = [
        {
            "Document": f"{e.vectorizable_type} #{e.vectorizable_id}",
            "Field": e.field_name,
            "Chunk": f"{e.chunk_index + 1}/{e.total_chunks}",
            "Content": e.content[:100] + "..." if len(e.content) > 100 else e.content,
            "Created": e.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        }
        for e in embeddings
    ]

    # Convert to DataFrame
    df = pd.DataFrame(formatted_data)
    return gr.update(value=df, visible=True)


def create_toggle_button():
    return gr.Button("Start Session", variant="primary")


async def toggle_session(button_text, instructions_value, voice_value, tools_value):
    if button_text == "Start Session":
        print("Starting new session...")
        ws_manager.instructions = instructions_value
        ws_manager.voice = voice_value

        # Initialize tools first
        if tools_value:
            print(f"Selected tools: {tools_value}")
            available_tools = ws_manager.tool_manager.get_available_tools()
            selected_functions = [
                f for f in available_tools if f.__name__ in tools_value
            ]

            # Initialize Jupyter kernel before registering tools if Python is selected
            if "python" in tools_value:
                print(f"Python tool detected, initializing Jupyter kernel...")
                from jupyter_backend import JupyterKernel

                work_dir = f"./notebooks/{ws_manager.session_id}"
                ws_manager.tool_manager.jupyter_kernel = JupyterKernel(work_dir)

            # Register tools after kernel is initialized
            ws_manager.tool_manager.register_tools(selected_functions)
            ws_manager.tool_manager.tool_choice = "auto"
        else:
            ws_manager.tool_manager.tools = []
            ws_manager.tool_manager.tool_choice = "none"

        await ws_manager.connect()
        print("Session started successfully")

        # Check the actual kernel status
        kernel_active = ws_manager.tool_manager.jupyter_kernel is not None

        return (
            gr.update(value="End Session", variant="secondary"),
            instructions_value,
            voice_value,
            tools_value,
            "🟢 Connected",
            "🟢 Kernel Ready" if kernel_active else "🔴 No Kernel",
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(visible=True),
        )
    else:
        await ws_manager.disconnect()
        return (
            gr.update(value="Start Session", variant="primary"),
            instructions_value,
            voice_value,
            tools_value,
            "🔴 Disconnected",
            "🔴 No Kernel",
            gr.update(interactive=False),
            gr.update(interactive=True),
            gr.update(visible=False),
        )


def numpy_to_audio_bytes(audio_np, sample_rate):
    with io.BytesIO() as buffer:
        # Write the audio data to the buffer in WAV format
        sf.write(buffer, audio_np, samplerate=sample_rate, format="WAV")
        buffer.seek(0)  # Move to the beginning of the buffer
        wav_bytes = buffer.read()
    return wav_bytes


def audio_to_item_create_event(audio_data: tuple) -> str:
    sample_rate, audio_np = audio_data

    # Load and resample in one step using librosa
    audio_np = librosa.load(
        io.BytesIO(numpy_to_audio_bytes(audio_np, sample_rate)),
        sr=24000,  # Target sample rate
        mono=True,
    )[0]

    # Convert back to 16-bit PCM
    audio_np = (audio_np * 32768.0).astype(np.int16)

    audio_bytes = numpy_to_audio_bytes(audio_np, 24000)
    pcm_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    event = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_audio", "audio": pcm_base64}],
        },
    }
    return json.dumps(event)


async def voice_chat_response(audio_data, history):
    if not ws_manager.is_connected:
        return "Please start a session first", history, None

    audio_event = audio_to_item_create_event(audio_data)
    audio_response = await ws_manager.send_and_receive(audio_event)

    if isinstance(audio_response, bytes):
        audio_io = io.BytesIO(audio_response)
        audio_segment = AudioSegment.from_raw(
            audio_io, sample_width=2, frame_rate=24000, channels=1
        )

        with io.BytesIO() as buffered:
            audio_segment.export(buffered, format="wav")
            return buffered.getvalue(), history, None

    return None, history, None


def perform_vector_search(query, field_name, limit):
    """Perform vector similarity search and return results"""
    if not query:
        return gr.update(visible=False)

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
        for result in results:
            print(f"\nDocument: {result.vectorizable_type} #{result.vectorizable_id}")
            print(f"Score: {result.similarity_score:.3f}")
            print(f"Full Content: {result.content}")
            print("-" * 50)

        # Format results for display
        formatted_results = [
            {
                "Score": f"{result.similarity_score:.3f}",
                "Document Title": f"{result.vectorizable_type} #{result.vectorizable_id}",
                "Content Preview": (
                    result.content[:100] + "..."
                    if len(result.content) > 100
                    else result.content
                ),
            }
            for result in results
        ]

        df = pd.DataFrame(formatted_results)
        return gr.update(value=df, visible=True)


# Updated Gradio Interface
with gr.Blocks(
    head="""
    <script>
    let isRecording = false;
    
    function shortcuts(e) {
        // Skip if we're in an input or textarea
        if (['input', 'textarea'].includes(e.target.tagName.toLowerCase())) {
            return;
        }

        if (e.code === 'Space' && e.type === 'keydown' && !isRecording) {
            e.preventDefault();
            isRecording = true;
            const recordButton = document.querySelector('.record-button');
            if (recordButton && !recordButton.disabled) {
                recordButton.click();
            }
        } else if (e.code === 'Space' && e.type === 'keyup' && isRecording) {
            e.preventDefault();
            isRecording = false;
            const stopButton = document.querySelector('.stop-button');
            if (stopButton && !stopButton.disabled) {
                stopButton.click();
            }
        }
    }
    
    document.addEventListener('keydown', shortcuts, false);
    document.addEventListener('keyup', shortcuts, false);
    </script>
    <style>
    #status, #kernel-status {
        background: transparent;
        margin: 0;
        padding: 0.5rem;
    }
    .status-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    </style>
"""
) as demo:
    gr.Markdown("<h1 style='text-align: center;'>OpenAI Realtime API</h1>")
    with gr.Row(elem_classes="status-container"):
        status_indicator = gr.Markdown("🔴 Disconnected", elem_id="status")
        kernel_status = gr.Markdown("🔴 No Kernel", elem_id="kernel-status")

    with gr.Tab("VoiceChat"):
        gr.Markdown(
            """Start a session, then speak to interact with the OpenAI model in real-time.
            """
        )

        assistant_template = gr.Dropdown(
            choices=list(assistant_manager.get_all_assistants().keys()),
            label="Load Assistant",
            value="General Assistant",
            interactive=True,
        )
        session_btn = create_toggle_button()

        instructions = gr.State(
            assistant_manager.get_assistant("General Assistant")["instructions"]
        )
        voice_state = gr.State(
            assistant_manager.get_assistant("General Assistant")["voice"]
        )
        tools_state = gr.State(
            assistant_manager.get_assistant("General Assistant")["tools"]
        )

        # Add the update_session_settings function here, before it's used
        def update_session_settings(assistant_name):
            """
            Update the session settings when a different assistant is selected.

            Args:
                assistant_name: Name of the selected assistant

            Returns:
                Tuple of (instructions, voice, tools) for the selected assistant
            """
            assistant = assistant_manager.get_assistant(assistant_name)
            return (
                assistant.get(
                    "instructions", DEFAULT_INSTRUCTIONS["General Assistant"]
                ),
                assistant.get("voice", "alloy"),
                assistant.get("tools", []),
            )

        # Audio interaction section
        with gr.Group(visible=False) as audio_group:
            audio_input = gr.Audio(
                label="Record your voice (hold spacebar or click record)",
                sources="microphone",
                type="numpy",
                render=True,
                interactive=False,
            )
            audio_output = gr.Audio(autoplay=True, render=True)
            history_state = gr.State([])

        # Update toggle_session function outputs to include kernel_status
        session_btn.click(
            fn=toggle_session,
            inputs=[session_btn, instructions, voice_state, tools_state],
            outputs=[
                session_btn,
                instructions,
                voice_state,
                tools_state,
                status_indicator,
                kernel_status,
                audio_input,
                assistant_template,
                audio_group,
            ],
        )

        audio_input.stop_recording(
            fn=voice_chat_response,
            inputs=[audio_input, history_state],
            outputs=[audio_output, history_state, audio_input],
        )

        # Add this event handler right after the function definition
        assistant_template.change(
            fn=update_session_settings,
            inputs=[assistant_template],
            outputs=[instructions, voice_state, tools_state],
        )

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

    # Add new tab for viewing vector embeddings
    with gr.Tab("Vector Embeddings"):
        gr.Markdown("View and search vector embeddings in the system")

        with gr.Row():
            with gr.Column(scale=1):
                search_query = gr.Textbox(
                    label="Search Query", placeholder="Enter text to search...", lines=3
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
                    visible=False,
                )

        gr.Markdown("### All Embeddings")
        embeddings_display = gr.DataFrame(
            headers=["Document", "Field", "Chunk", "Content", "Created"],
            label="All Embeddings",
        )
        refresh_embeddings = gr.Button("Refresh Embeddings")

        # Wire up the search functionality
        search_button.click(
            fn=perform_vector_search,
            inputs=[search_query, search_field, search_limit],
            outputs=[search_results],
        )

        # Wire up the refresh functionality
        refresh_embeddings.click(fn=format_embeddings, outputs=[embeddings_display])

        # Initial load of embeddings
        embeddings_display.value = format_embeddings()

    # Add new tab for managing assistants
    with gr.Tab("Manage Assistants"):
        gr.Markdown("Create and manage AI assistants with custom voices and tools")

        with gr.Row():
            with gr.Column(scale=1):
                assistant_list = gr.Dropdown(
                    choices=["None"]
                    + list(assistant_manager.get_all_assistants().keys()),
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
                        f.__name__
                        for f in ws_manager.tool_manager.get_available_tools()
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

        # Update the helper functions
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
                    interactive=not assistant_manager.is_default_assistant(
                        assistant_name
                    )
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
                    gr.update(
                        choices=list(assistant_manager.get_all_assistants().keys())
                    ),
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

        # Update the event handlers
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

    # Add new tab for managing documents
    with gr.Tab("Manage Documents"):
        gr.Markdown("Create and manage documents")

        with gr.Row():
            with gr.Column(scale=1):
                document_list = gr.Dropdown(
                    choices=["None"],  # We'll populate this dynamically
                    label="Select Document",
                    value="None",
                )

                new_document_btn = gr.Button("Create New Document", variant="primary")
                delete_document_btn = gr.Button(
                    "Delete Selected Document", variant="stop"
                )

            with gr.Column(scale=2):
                document_title = gr.Textbox(
                    label="Document Title",
                    placeholder="Enter document title...",
                    interactive=True,
                )
                document_content = gr.TextArea(
                    label="Content",
                    placeholder="Enter the document content...",
                    lines=10,
                    interactive=True,
                )
                document_published = gr.Checkbox(
                    label="Published",
                    value=False,
                    interactive=True,
                )
                save_document_btn = gr.Button("Save Changes", variant="primary")

    # Add the necessary event handlers
    def load_document_list():
        documents = document_service.get_all_documents()
        # Convert to list of tuples (label, value) for the dropdown
        return [("None", "None")] + [(title, title) for title in documents.keys()]

    def load_document(title):
        if title == "None":
            return "", "", False
        document = document_service.get_document_by_title(title)
        if document:
            return document.title, document.content, document.published
        return "", "", False

    def create_new_document():
        return "", "", False, gr.update(value="None")

    def save_document(title, content, published, current_selection):
        if not title:
            return "Please enter a title", gr.update(choices=load_document_list())

        if current_selection == "None":
            document_service.create_document(title, content, published)
        else:
            document_service.update_document(current_selection, content, published)
            if current_selection != title:
                document_service.delete_document(current_selection)
                document_service.create_document(title, content, published)

        return "Document saved successfully!", gr.update(choices=load_document_list())

    def delete_document(title):
        if title and title != "None":
            document_service.delete_document(title)
            return (
                "",
                "",
                False,
                "Document deleted successfully!",
                gr.update(choices=load_document_list()),
            )
        return (
            "",
            "",
            False,
            "No document selected",
            gr.update(choices=load_document_list()),
        )

    # Wire up the event handlers
    document_list.change(
        load_document,
        inputs=[document_list],
        outputs=[document_title, document_content, document_published],
    )

    new_document_btn.click(
        create_new_document,
        outputs=[document_title, document_content, document_published, document_list],
    )

    save_document_btn.click(
        save_document,
        inputs=[document_title, document_content, document_published, document_list],
        outputs=[gr.Textbox(visible=False), document_list],
    )

    delete_document_btn.click(
        delete_document,
        inputs=[document_list],
        outputs=[
            document_title,
            document_content,
            document_published,
            gr.Textbox(visible=False),
            document_list,
        ],
    )

    # Initialize document list
    document_list.choices = load_document_list()

if __name__ == "__main__":
    demo.launch()

    demo.launch()
