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

load_dotenv()

# Add this near the top of the file, after the imports but before the WebSocketManager class
AVAILABLE_VOICES = [
    "alloy",
    "ash",
    "coral",
    "echo",
    "fable",
    "onyx",
    "nova",
    "sage",
    "shimmer",
]

# Add near the top with other constants
DEFAULT_INSTRUCTIONS = {
    "General Assistant": "You are a helpful assistant.",
    "Spanish Language Teacher": "You are a Spanish (MX) language teacher. Help users practice speaking and correct their grammar and pronunciation.",
    "Technical Expert": "You are a technical expert. Provide detailed technical explanations and help debug problems.",
}


class WebSocketManager:
    def __init__(self):
        self.websocket = None
        self.is_connected = False
        self.last_message_id = None
        self.last_assistant_message_id = None
        self.instructions = "You are a helpful assistant."
        self.voice = "alloy"
        self.temperature = 0.6
        self.event_logs = []

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

    async def connect(self):
        if self.is_connected:
            return

        url = (
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
        )
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "OpenAI-Beta": "realtime=v1",
        }

        self.websocket = await websockets.connect(url, additional_headers=headers)
        self.is_connected = True
        print("Connected to server.")

        session_update = {
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "instructions": self.instructions,
                "voice": self.voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": None,
                "tools": [],
                "tool_choice": "none",
                "temperature": self.temperature,
            },
        }

        # Log and send the session update
        self._log_event("SENDING", session_update)
        await self.websocket.send(json.dumps(session_update))

    async def disconnect(self):
        if self.websocket and self.is_connected:
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

        # Use last_assistant_message_id instead of last_message_id
        if self.last_assistant_message_id:
            event_dict = json.loads(audio_event)
            event_dict["previous_item_id"] = self.last_assistant_message_id
            audio_event = json.dumps(event_dict)

        # Log outgoing event
        self._log_event("SENDING", audio_event)
        await self.websocket.send(audio_event)

        # Wait for the message to be created
        async for message in self.websocket:
            self._log_event("RECEIVED", message)
            event = json.loads(message)

            if event.get("type") == "conversation.item.created":
                # After the message is created, send a response.create event
                create_response = {"type": "response.create"}
                self._log_event("SENDING", create_response)
                await self.websocket.send(json.dumps(create_response))
                break

        # Now continue listening for the response
        async for message in self.websocket:
            self._log_event("RECEIVED", message)
            event = json.loads(message)

            # Update tracking of message IDs
            if event.get("type") == "conversation.item.created":
                item = event.get("item", {})
                if item.get("role") == "assistant":
                    self.last_assistant_message_id = item.get("id")
                self.last_message_id = item.get("id")

            if event.get("type") == "response.audio.delta":
                audio_data_list.append(event["delta"])

            if event.get("type") == "response.audio.done":
                full_audio_base64 = "".join(audio_data_list)
                return base64.b64decode(full_audio_base64)

    def get_logs(self):
        """Return all logged events as a single string"""
        return "\n".join(self.event_logs)


# Create a global WebSocket manager
ws_manager = WebSocketManager()


def create_toggle_button():
    return gr.Button("Start Session", variant="primary")


async def toggle_session(button_text, instructions, voice):
    if button_text == "Start Session":
        ws_manager.instructions = instructions
        ws_manager.voice = voice
        await ws_manager.connect()
        return (
            gr.update(value="End Session", variant="secondary"),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "🟢 Connected",
            gr.update(open=False),
            gr.update(interactive=True),
        )
    else:
        await ws_manager.disconnect()
        return (
            gr.update(value="Start Session", variant="primary"),
            gr.update(interactive=True),
            gr.update(interactive=True),
            "🔴 Disconnected",
            gr.update(open=True),
            gr.update(interactive=False),
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
    audio_bytes = numpy_to_audio_bytes(audio_np, sample_rate)

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
class InstructionsManager:
    def __init__(self):
        self.instructions = DEFAULT_INSTRUCTIONS.copy()
        self.load_instructions()

    def load_instructions(self):
        try:
            with open("instructions.json", "r") as f:
                saved_instructions = json.load(f)
                self.instructions.update(saved_instructions)
        except FileNotFoundError:
            self.save_instructions()

    def save_instructions(self):
        with open("instructions.json", "w") as f:
            json.dump(self.instructions, f)

    def add_instruction(self, name, text):
        self.instructions[name] = text
        self.save_instructions()
        return list(self.instructions.keys())

    def delete_instruction(self, name):
        if name in self.instructions and name not in DEFAULT_INSTRUCTIONS:
            del self.instructions[name]
            self.save_instructions()
        return list(self.instructions.keys())

    def edit_instruction(self, name, new_text):
        if name in self.instructions:
            self.instructions[name] = new_text
            self.save_instructions()
        return list(self.instructions.keys())


# Create a global instructions manager
instructions_manager = InstructionsManager()


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
    #status {
        background: transparent;
        margin: 0;
        padding: 0.5rem;
    }
    </style>
"""
) as demo:
    gr.Markdown("<h1 style='text-align: center;'>OpenAI Realtime API</h1>")
    status_indicator = gr.Markdown("🔴 Disconnected", elem_id="status")

    with gr.Tab("VoiceChat"):
        gr.Markdown(
            "Start a session, then speak to interact with the OpenAI model in real-time."
        )

        # Create settings accordion for configuration
        with gr.Accordion("Settings", open=False) as settings_accordion:
            with gr.Row():
                instruction_template = gr.Dropdown(
                    choices=list(instructions_manager.instructions.keys()),
                    label="Load Template",
                    value="General Assistant",
                )
            instructions = gr.Textbox(
                label="Instructions",
                placeholder="Enter custom instructions for the AI assistant...",
                value=instructions_manager.instructions["General Assistant"],
                lines=3,
            )
            voice_dropdown = gr.Dropdown(
                choices=AVAILABLE_VOICES,
                value="alloy",
                label="Voice",
                info="Select the AI voice to use",
            )
            session_btn = create_toggle_button()

        # Add this function to handle template selection
        def load_template(template_name):
            return instructions_manager.instructions.get(template_name, "")

        instruction_template.change(
            fn=load_template, inputs=[instruction_template], outputs=[instructions]
        )

        # Audio interaction section
        with gr.Group():
            audio_input = gr.Audio(
                label="Record your voice (hold spacebar or click record)",
                sources="microphone",
                type="numpy",
                render=True,
                interactive=False,
            )
            audio_output = gr.Audio(autoplay=True, render=True)
            history_state = gr.State([])

        # Update the toggle_session function to remove status_text
        session_btn.click(
            fn=toggle_session,
            inputs=[session_btn, instructions, voice_dropdown],
            outputs=[
                session_btn,
                instructions,
                voice_dropdown,
                status_indicator,
                settings_accordion,
                audio_input,
            ],
        )

        audio_input.stop_recording(
            fn=voice_chat_response,
            inputs=[audio_input, history_state],
            outputs=[audio_output, history_state, audio_input],
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

    # Add new tab for managing instructions
    with gr.Tab("Manage Instructions"):
        gr.Markdown("Create and manage instruction templates")

        with gr.Row():
            with gr.Column(scale=1):
                template_list = gr.Dropdown(
                    choices=["None"] + list(instructions_manager.instructions.keys()),
                    label="Select Template",
                    value="None",
                )

                new_template_btn = gr.Button("Create New Template", variant="primary")
                delete_btn = gr.Button("Delete Selected Template", variant="stop")

            with gr.Column(scale=2):
                template_name = gr.Textbox(
                    label="Template Name",
                    placeholder="Enter template name...",
                    interactive=True,
                )
                template_text = gr.Textbox(
                    label="Template Instructions",
                    placeholder="Enter the instructions for this template...",
                    lines=5,
                    interactive=True,
                )
                save_btn = gr.Button("Save Changes", variant="primary")

        # Add these helper functions
        def load_template_details(template_name):
            if template_name == "None":
                return "", "", gr.update(interactive=False)
            text = instructions_manager.instructions.get(template_name, "")
            return (
                template_name,
                text,
                gr.update(interactive=template_name not in DEFAULT_INSTRUCTIONS),
            )

        def save_template_changes(name, text):
            if name in instructions_manager.instructions:
                instructions_manager.edit_instruction(name, text)
                choices = ["None"] + list(instructions_manager.instructions.keys())
                return gr.update(choices=choices), "Template updated successfully!"
            else:
                instructions_manager.add_instruction(name, text)
                choices = ["None"] + list(instructions_manager.instructions.keys())
                return gr.update(choices=choices, value=name), "New template created!"

        def create_new_template():
            return "", "", gr.update(interactive=True)

        def delete_current_template(name):
            if name in DEFAULT_INSTRUCTIONS:
                return (
                    gr.update(
                        choices=["None"]
                        + list(instructions_manager.instructions.keys())
                    ),
                    "None",
                    "",
                    gr.update(interactive=False),
                    "Cannot delete default templates!",
                )

            instructions_manager.delete_instruction(name)
            choices = ["None"] + list(instructions_manager.instructions.keys())
            return (
                gr.update(choices=choices),
                "None",
                "",
                gr.update(interactive=False),
                "Template deleted successfully!",
            )

        # Update the event handlers
        template_list.change(
            fn=load_template_details,
            inputs=[template_list],
            outputs=[template_name, template_text, delete_btn],
        )

        new_template_btn.click(
            fn=create_new_template, outputs=[template_name, template_text, delete_btn]
        )

        save_btn.click(
            fn=save_template_changes,
            inputs=[template_name, template_text],
            outputs=[template_list, gr.Textbox(visible=False)],  # For success message
        )

        delete_btn.click(
            fn=delete_current_template,
            inputs=[template_name],
            outputs=[
                template_list,
                template_name,
                template_text,
                delete_btn,
                gr.Textbox(visible=False),  # For success message
            ],
        )

if __name__ == "__main__":
    demo.launch()
