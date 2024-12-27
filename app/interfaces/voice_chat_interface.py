import gradio as gr
import json
import io
import base64
import numpy as np
import librosa
from pydub import AudioSegment
import soundfile as sf
from app.core.websocket import WebSocketManager
from app.core.assistant_manager import AssistantManager, DEFAULT_INSTRUCTIONS
from app.interfaces.debug_interface import create_debug_interface
from app.interfaces.tool_history_interface import create_tool_history_interface
import time


def create_toggle_button():
    """Create a button that toggles between Start and End session states"""
    return gr.Button("Start Session", variant="primary")


def create_voice_chat_interface(
    ws_manager: WebSocketManager, assistant_manager: AssistantManager
):
    """Create the Voice Chat interface tab

    Args:
        ws_manager: The WebSocket manager instance for handling audio communication
        assistant_manager: The Assistant manager instance for managing assistant settings
    """
    with gr.Row():
        with gr.Column():
            status_indicator = gr.HighlightedText(
                value=[("WebSocket", "Inactive")],
                elem_id="status",
                show_label=False,
                color_map={"Active": "green", "Inactive": "red"},
                container=False,
            )
            kernel_status = gr.HighlightedText(
                value=[("Jupyter Kernel", "Inactive")],
                elem_id="kernel-status",
                show_label=False,
                color_map={"Active": "green", "Inactive": "red"},
                container=False,
            )
        with gr.Column():
            assistant_template = gr.Dropdown(
                choices=list(assistant_manager.get_all_assistants().keys()),
                label="Load Assistant",
                value="General Assistant",
                interactive=True,
                container=False,
            )
            session_btn = create_toggle_button()
    gr.Markdown(
        """Start a session, then speak to interact with the OpenAI model in real-time.
        """
    )

    instructions = gr.State(
        assistant_manager.get_assistant("General Assistant")["instructions"]
    )
    voice_state = gr.State(
        assistant_manager.get_assistant("General Assistant")["voice"]
    )
    tools_state = gr.State(
        assistant_manager.get_assistant("General Assistant")["tools"]
    )

    # Replace the separate output and input groups with a single container
    with gr.Column(visible=False, elem_id="chat-container") as chat_container:
        # Output group
        with gr.Group():
            audio_input = gr.Audio(
                sources="microphone",
                type="numpy",
                render=True,
                interactive=False,
                show_label=False,
            )
            chatbot = gr.Chatbot(type="messages", height=400, show_label=False)
            audio_output = gr.Audio(
                autoplay=True, render=True, show_label=False, visible=False
            )
            msg_box = gr.MultimodalTextbox(
                placeholder="Type a message, upload files, or use voice input...",
                container=False,
                show_label=False,
                file_count="multiple",
                interactive=True,
            )

        with gr.Accordion("Session Debug", open=False):
            create_debug_interface(ws_manager)

        with gr.Accordion("Tool History", open=False):
            create_tool_history_interface(ws_manager)

        clear_btn = gr.Button("Clear Chat")

        history_state = gr.State([])

    async def handle_text_message(message, history):
        """Handle text messages sent through the chat"""
        if not ws_manager.is_connected:
            return "", history
        # Handle text if present
        if message["text"]:
            text_event = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": message["text"]}],
                },
            }
            history.append({"role": "user", "content": message["text"]})
            await ws_manager.send_and_receive(json.dumps(text_event))

        return "", history

    async def process_bot_response(history):
        """Process the assistant's response and update chat history"""
        async for message in ws_manager.websocket:
            event = json.loads(message)

            # Handle text deltas
            if event.get("type") == "response.text.delta":
                if not history or history[-1]["role"] != "assistant":
                    history.append({"role": "assistant", "content": ""})
                history[-1]["content"] += event.get("delta", "")
                yield history

            # Handle audio transcript
            elif event.get("type") == "response.audio_transcript.done":
                if not history or history[-1]["role"] != "assistant":
                    history.append({"role": "assistant", "content": ""})
                history[-1]["content"] = event.get("transcript", "")
                yield history

            # Break when response is complete
            elif event.get("type") == "response.done":
                break

    def clear_chat():
        """Clear the chat history"""
        return []

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
                    from app.core.jupyter import JupyterKernel

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
                [("WebSocket", "Active")],
                [("Jupyter Kernel", "Active" if kernel_active else "Inactive")],
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
                [("WebSocket", "Inactive")],
                [("Jupyter Kernel", "Inactive")],
                gr.update(interactive=False),
                gr.update(interactive=True),
                gr.update(visible=False),
            )

    def update_session_settings(assistant_name):
        """Update the session settings when a different assistant is selected"""
        assistant = assistant_manager.get_assistant(assistant_name)
        return (
            assistant.get("instructions", DEFAULT_INSTRUCTIONS["General Assistant"]),
            assistant.get("voice", "alloy"),
            assistant.get("tools", []),
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
        """Handle voice input and update chat history"""
        if not ws_manager.is_connected:
            yield None, history, None
            return

        # Create audio event
        audio_event = audio_to_item_create_event(audio_data)

        # Add transcription placeholder to history
        history.append({"role": "user", "content": "🎤 Processing voice input..."})
        yield None, history, None

        # Send audio and get response
        audio_response = await ws_manager.send_and_receive(audio_event)

        # Update history with transcription if available
        if isinstance(audio_response, bytes) and len(audio_response) > 0:
            # Convert audio response to playable format
            audio_io = io.BytesIO(audio_response)
            audio_segment = AudioSegment.from_raw(
                audio_io, sample_width=2, frame_rate=24000, channels=1
            )

            with io.BytesIO() as buffered:
                audio_segment.export(buffered, format="wav")
                buffered_audio = buffered.getvalue()

                # Update history with assistant's audio response
                history[-1]["content"] = "🎤 Voice message sent"
                history.append(
                    {
                        "role": "assistant",
                        "content": "🔊 Audio response",
                    }
                )

                print(
                    f"Audio response size: {len(buffered_audio)} bytes"
                )  # Debug print
                # Yield the audio response once
                yield buffered_audio, history, None

    # Wire up all the event handlers
    assistant_template.change(
        fn=update_session_settings,
        inputs=[assistant_template],
        outputs=[instructions, voice_state, tools_state],
    )

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
            chat_container,
        ],
    )

    audio_input.stop_recording(
        fn=voice_chat_response,
        inputs=[audio_input, chatbot],
        outputs=[audio_output, chatbot, audio_input],
        show_progress="minimal",
    ).then(
        process_bot_response,
        chatbot,
        chatbot,
    )

    msg_box.submit(
        handle_text_message,
        [msg_box, chatbot],
        [msg_box, chatbot],
        queue=False,
    ).then(
        process_bot_response,
        chatbot,
        chatbot,
    )

    clear_btn.click(lambda: [], None, chatbot, queue=False)

    return assistant_template  # Return this for use in other interfaces
