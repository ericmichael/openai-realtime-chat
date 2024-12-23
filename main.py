import os
import io
import json
import asyncio
import base64
import websockets
from pydub import AudioSegment
import soundfile as sf
import gradio as gr
from dotenv import load_dotenv
from functools import partial
from gradio.components import Audio
import librosa
import numpy as np
from tools import ToolManager, schema
from datetime import datetime
import geocoder
from magic_variables import magic_manager

load_dotenv()

# Add this near the top of the file, after the imports but before the WebSocketManager class
AVAILABLE_VOICES = [
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "sage",
    "shimmer",
    "verse",
]

# Add near the top with other constants
DEFAULT_INSTRUCTIONS = {
    "General Assistant": "You are a helpful assistant.",
    "Spanish Language Teacher": "You are a Spanish (MX) language teacher. Help users practice speaking and correct their grammar and pronunciation.",
    "Technical Expert": "You are a technical expert. Provide detailed technical explanations and help debug problems.",
}

# Add this after the DEFAULT_INSTRUCTIONS constant
MAGIC_VARIABLES = {
    "todays_date": lambda: datetime.now().strftime("%B %d, %Y"),
    "current_time": lambda: datetime.now().strftime("%I:%M %p"),
    "user_location": lambda: f"{geocoder.ip('me').city}, {geocoder.ip('me').state}",
}


class WebSocketManager:
    def __init__(self):
        self.websocket = None
        self.is_connected = False
        self.last_message_id = None
        self.last_assistant_message_id = None
        self._raw_instructions = "You are a helpful assistant."
        self.instructions = self._process_instructions(self._raw_instructions)
        self.voice = "alloy"
        self.temperature = 0.6
        self.event_logs = []
        self.tool_manager = ToolManager()
        self.selected_tools = []
        self.jupyter_kernel = None
        self.session_id = None

    def _log_event(self, direction: str, event: str):
        """Helper to log WebSocket events, omitting base64 audio data"""
        try:
            # Parse JSON if it's a string
            parsed = json.loads(event) if isinstance(event, str) else event

            # Skip logging both transcript and audio delta events
            if isinstance(parsed, dict) and parsed.get("type") in [
                "response.audio.delta",
                "response.audio_transcript.delta",
            ]:
                return

            # Create a copy for logging to avoid modifying the original
            log_data = json.loads(json.dumps(parsed))

            # Omit base64 audio data from logs
            if isinstance(log_data, dict):
                # For outgoing messages
                if "item" in log_data and "content" in log_data["item"]:
                    for content in log_data["item"]["content"]:
                        if "audio" in content:
                            content["audio"] = "<base64_audio_omitted>"

                # For incoming audio deltas
                if log_data.get("type") == "response.audio.delta":
                    log_data["delta"] = "<base64_audio_omitted>"

            formatted = json.dumps(log_data, indent=2)
            print(f"\n{direction} WebSocket Event:")
            print(f"{'=' * 40}")
            print(formatted)
            print(f"{'=' * 40}\n")

            # Add log entry to our event_logs list
            self.event_logs.append(
                f"\n{direction} WebSocket Event:\n{'=' * 40}\n{formatted}\n{'=' * 40}\n"
            )
        except:
            # Fallback for non-JSON events
            log_entry = (
                f"\n{direction} WebSocket Event:\n{'=' * 40}\n{event}\n{'=' * 40}\n"
            )
            self.event_logs.append(log_entry)
            print(f"\n{direction} WebSocket Event:")
            print(f"{'=' * 40}")
            print(event)
            print(f"{'=' * 40}\n")

    def _process_instructions(self, instructions: str) -> str:
        """Process instructions using the magic variable manager"""
        return magic_manager.process_instructions(instructions)

    @property
    def instructions(self):
        return self._raw_instructions

    @instructions.setter
    def instructions(self, value):
        self._raw_instructions = value
        self._processed_instructions = self._process_instructions(value)

    async def connect(self):
        if self.is_connected:
            print("Already connected, skipping...")  # Debug print
            return

        print("Connecting to WebSocket...")  # Debug print
        # Generate a unique session ID
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Generated Session ID: {self.session_id}")  # Debug print

        # Initialize Jupyter kernel if Python tool is selected
        if any(tool.get("name") == "python" for tool in self.tool_manager.tools):
            print(
                f"Python tool detected, initializing Jupyter kernel in ./notebooks/{self.session_id}"
            )  # Debug print
            from jupyter_backend import JupyterKernel

            work_dir = f"./notebooks/{self.session_id}"
            self.jupyter_kernel = JupyterKernel(work_dir)
            # Update the Python tool to use the kernel
            self.tool_manager.jupyter_kernel = self.jupyter_kernel

        url = (
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
        )
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "OpenAI-Beta": "realtime=v1",
        }

        self.websocket = await websockets.connect(url, additional_headers=headers)
        print("WebSocket connected, waiting for session update confirmation...")

        session_update = {
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "instructions": self._processed_instructions,
                "voice": self.voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": None,
                "tools": self.tool_manager.tools,
                "tool_choice": self.tool_manager.tool_choice,
                "temperature": self.temperature,
            },
        }

        # Log and send the session update
        self._log_event("SENDING", session_update)
        await self.websocket.send(json.dumps(session_update))

        # Wait for session.updated confirmation
        async for message in self.websocket:
            self._log_event("RECEIVED", message)
            event = json.loads(message)
            if event.get("type") == "session.updated":
                self.is_connected = True
                print("Session updated, connection is now active.")
                break

    async def disconnect(self):
        if self.websocket and self.is_connected:
            # Cleanup Jupyter kernel if it exists
            if self.jupyter_kernel:
                self.jupyter_kernel.kernel_client.shutdown()
                self.jupyter_kernel = None

            await self.websocket.close()
            self.is_connected = False
            self.websocket = None
            self.last_message_id = None
            self.last_assistant_message_id = None
            print("Disconnected from server.")

    async def send_and_receive(self, audio_event):
        if not self.is_connected or not self.websocket:
            raise Exception("WebSocket not connected")

        audio_data_list = []

        if self.last_assistant_message_id:
            event_dict = json.loads(audio_event)
            event_dict["previous_item_id"] = self.last_assistant_message_id
            audio_event = json.dumps(event_dict)

        self._log_event("SENDING", audio_event)
        await self.websocket.send(audio_event)

        # Wait for the message to be created and send response.create
        async for message in self.websocket:
            self._log_event("RECEIVED", message)
            event = json.loads(message)
            if event.get("type") == "conversation.item.created":
                create_response = {"type": "response.create"}
                self._log_event("SENDING", create_response)
                await self.websocket.send(json.dumps(create_response))
                break

        # Now listen for the response
        async for message in self.websocket:
            self._log_event("RECEIVED", message)
            event = json.loads(message)

            # Handle audio responses
            if event.get("type") == "response.audio.delta":
                audio_data_list.append(event["delta"])

            # Process function calls when response is complete
            elif event.get("type") == "response.done":
                # Process function calls in the output
                for output_item in event["response"]["output"]:
                    if output_item["type"] == "function_call":
                        tool_name = output_item["name"]
                        raw_args = output_item["arguments"]

                        # Try to parse as JSON first
                        try:
                            tool_args = json.loads(raw_args)
                            # If it's Python tool but args aren't in expected format, wrap them
                            if tool_name == "python" and not isinstance(
                                tool_args, dict
                            ):
                                tool_args = {"code": raw_args}
                        except json.JSONDecodeError:
                            # If JSON parsing fails and it's Python tool, wrap the raw code
                            if tool_name == "python":
                                tool_args = {"code": raw_args}
                            else:
                                # For non-Python tools, re-raise the error
                                raise

                        # Execute the tool
                        result = await self.tool_manager.execute_tool(
                            tool_name, tool_args
                        )

                        # Send the result back
                        tool_response = {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "function_call_output",
                                "call_id": output_item["call_id"],
                                "output": json.dumps(result),
                            },
                        }
                        self._log_event("SENDING", tool_response)
                        await self.websocket.send(json.dumps(tool_response))

                        create_response = {"type": "response.create"}
                        self._log_event("SENDING", create_response)
                        await self.websocket.send(json.dumps(create_response))

            elif event.get("type") == "response.audio.done":
                full_audio_base64 = "".join(audio_data_list)
                return base64.b64decode(full_audio_base64)

    def get_logs(self):
        """Return all logged events as a single string"""
        return "\n".join(self.event_logs)


# Create a global WebSocket manager
ws_manager = WebSocketManager()


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


# Add this class after WebSocketManager
class AssistantManager:
    def __init__(self):
        self.assistants = {
            "General Assistant": {
                "instructions": "You are a helpful assistant.",
                "voice": "alloy",
                "tools": [],
            },
            "Spanish Language Teacher": {
                "instructions": "You are a Spanish (MX) language teacher. Help users practice speaking and correct their grammar and pronunciation.",
                "voice": "alloy",
                "tools": [],
            },
            "Technical Expert": {
                "instructions": "You are a technical expert. Provide detailed technical explanations and help debug problems.",
                "voice": "alloy",
                "tools": ["get_weather", "get_time"],
            },
        }
        self.load_assistants()

    def load_assistants(self):
        try:
            with open("assistants.json", "r") as f:
                saved_assistants = json.load(f)
                self.assistants.update(saved_assistants)
        except FileNotFoundError:
            self.save_assistants()

    def save_assistants(self):
        with open("assistants.json", "w") as f:
            json.dump(self.assistants, f)

    def add_assistant(self, name, data):
        self.assistants[name] = data
        self.save_assistants()
        return list(self.assistants.keys())

    def delete_assistant(self, name):
        if name in self.assistants and name not in DEFAULT_INSTRUCTIONS:
            del self.assistants[name]
            self.save_assistants()
        return list(self.assistants.keys())

    def edit_assistant(self, name, data):
        if name in self.assistants:
            self.assistants[name] = data
            self.save_assistants()
        return list(self.assistants.keys())


# Create a global assistant manager
assistant_manager = AssistantManager()


# Add this function after the existing functions
def update_session_settings(assistant_name):
    """Update the session settings based on selected assistant"""
    assistant = assistant_manager.assistants.get(assistant_name, {})
    return (
        assistant.get("instructions", "You are a helpful assistant."),
        assistant.get("voice", "alloy"),
        assistant.get("tools", []),
    )


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
            choices=list(assistant_manager.assistants.keys()),
            label="Load Assistant",
            value="General Assistant",
            interactive=True,
        )
        session_btn = create_toggle_button()

        instructions = gr.State(
            assistant_manager.assistants["General Assistant"]["instructions"]
        )
        voice_state = gr.State(
            assistant_manager.assistants["General Assistant"]["voice"]
        )
        tools_state = gr.State(
            assistant_manager.assistants["General Assistant"]["tools"]
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

        # Add this event handler
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

    # Add new tab for managing assistants
    with gr.Tab("Manage Assistants"):
        gr.Markdown("Create and manage AI assistants with custom voices and tools")

        with gr.Row():
            with gr.Column(scale=1):
                assistant_list = gr.Dropdown(
                    choices=["None"] + list(assistant_manager.assistants.keys()),
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
                    choices=AVAILABLE_VOICES,
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

        # Update the helper functions
        def load_assistant_details(assistant_name):
            if assistant_name == "None":
                return "", "alloy", "", [], gr.update(interactive=False)
            assistant = assistant_manager.assistants.get(assistant_name, {})
            return (
                assistant_name,
                assistant.get("voice", "alloy"),
                assistant.get("instructions", ""),
                assistant.get("tools", []),
                gr.update(interactive=assistant_name not in DEFAULT_INSTRUCTIONS),
            )

        def save_assistant_changes(name, voice, instructions, tools):
            data = {
                "voice": voice,
                "instructions": instructions,
                "tools": tools,
            }

            if name in assistant_manager.assistants:
                assistant_manager.edit_assistant(name, data)
                choices = ["None"] + list(assistant_manager.assistants.keys())
                # Update both dropdowns
                return (
                    gr.update(choices=choices),
                    gr.update(choices=list(assistant_manager.assistants.keys())),
                    "Assistant updated successfully!",
                )
            else:
                assistant_manager.add_assistant(name, data)
                choices = ["None"] + list(assistant_manager.assistants.keys())
                # Update both dropdowns
                return (
                    gr.update(choices=choices, value=name),
                    gr.update(choices=list(assistant_manager.assistants.keys())),
                    "New assistant created!",
                )

        def create_new_assistant():
            return "", "alloy", "", [], gr.update(interactive=True)

        def delete_current_assistant(name):
            if name in DEFAULT_INSTRUCTIONS:
                return (
                    gr.update(
                        choices=["None"] + list(assistant_manager.assistants.keys())
                    ),
                    "None",
                    "alloy",
                    "",
                    [],
                    gr.update(interactive=False),
                    gr.update(choices=list(assistant_manager.assistants.keys())),
                    "Cannot delete default assistants!",
                )

            assistant_manager.delete_assistant(name)
            choices = ["None"] + list(assistant_manager.assistants.keys())
            return (
                gr.update(choices=choices),
                "None",
                "alloy",
                "",
                [],
                gr.update(interactive=False),
                gr.update(choices=list(assistant_manager.assistants.keys())),
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


if __name__ == "__main__":
    demo.launch()
