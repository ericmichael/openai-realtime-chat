import json
import base64
import websockets
from datetime import datetime
from app.core.tools import ToolManager
from app.utils.magic_variables import magic_manager
from app.config import Config


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
        self.api_key = Config.OPENAI_API_KEY
        self.debug = Config.DEBUG

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
            from app.core.jupyter import JupyterKernel

            work_dir = f"./notebooks/{self.session_id}"
            self.jupyter_kernel = JupyterKernel(work_dir)
            # Update the Python tool to use the kernel
            self.tool_manager.jupyter_kernel = self.jupyter_kernel

        url = (
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
        )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
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

    async def send_and_receive(self, event):
        """Send an event and handle the response stream"""
        if not self.is_connected or not self.websocket:
            raise Exception("WebSocket not connected")

        audio_data_list = []

        # Add previous message ID if it exists
        if isinstance(event, str):
            event_dict = json.loads(event)
        else:
            event_dict = event

        if self.last_assistant_message_id:
            event_dict["previous_item_id"] = self.last_assistant_message_id
            event = json.dumps(event_dict)

        self._log_event("SENDING", event)
        await self.websocket.send(
            event if isinstance(event, str) else json.dumps(event)
        )

        # Wait for the message to be created and store the message ID
        async for message in self.websocket:
            self._log_event("RECEIVED", message)
            event = json.loads(message)
            if event.get("type") == "conversation.item.created":
                self.last_message_id = event.get("item", {}).get("id")
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
                audio_data_list.append(base64.b64decode(event["delta"]))

            # Store the assistant's message ID when the response is complete
            elif event.get("type") == "response.done":
                self.last_assistant_message_id = event.get("item_id")

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
                # Concatenate all audio chunks
                full_audio = b"".join(audio_data_list)
                return full_audio

    def get_logs(self):
        """Return all logged events as a single string"""
        return "\n".join(self.event_logs)
