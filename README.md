# OpenAI Realtime Voice Chat

This project is an enhanced version of [NuclearGeekETH's OpenAI Realtime Python Example](https://github.com/nucleargeeketh/openai-realtime-python-example), extending it with additional features and improvements. The system provides a real-time voice chat interface using OpenAI's GPT-4o model, processing spoken input and returning instant audio responses.

## Enhanced Features

- **Persistent WebSocket Management**: Maintains a stable connection across multiple interactions
- **Customizable AI Instructions**: Create, save, and manage different instruction templates
- **Voice Selection**: Choose from multiple AI voices (alloy, echo, fable, onyx, nova, etc.)
- **Debug Interface**: Monitor WebSocket events and messages in real-time
- **Improved UI/UX**: 
  - Spacebar shortcuts for recording
  - Session management controls
  - Status indicators
  - Settings accordion
- **All Original Features**:
  - Real-time interaction with OpenAI's GPT-4o Model
  - Audio Processing
  - Gradio Interface
  - Asynchronous WebSocket Communication

## Features

- **Real-time interaction with OpenAI's GPT-4o Model**: Converse with the AI using your voice.
- **Audio Processing**: The application processes and decodes audio data for smooth interactions.
- **Gradio Interface**: Provides an easy-to-use web interface for voice recording and playback.
- **Asynchronous Websocket Communication**: Utilizes Python's `asyncio` and `websockets` for efficient real-time data transfer.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.7 or above
- Virtual Environment (recommended)
- API key from OpenAI with access to the realtime API
- Packages listed in the `requirements.txt` (see below for details)

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/ericmichael/openai-realtime-chat.git
   cd openai-realtime-chat
   ```

2. **Create and activate a virtual environment**:

   - On macOS and Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     python -m venv venv
     .\venv\Scripts\activate
     ```

3. **Install the dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the environment variables**:

   Create a `.env` file in the project's root directory and add your OpenAI API key:
   
   ```
   OPENAI_API_KEY=your-openai-api-key
   ```

### Usage

1. **Run the application**:

   ```bash
   python main.py
   ```

2. **Access the Gradio Interface**:

   Open your browser and navigate to the provided localhost URL (e.g., `http://127.0.0.1:7860/`).

3. **Configure and Start**:
   - Go to the "VoiceChat" tab
   - Open the Settings accordion
   - Select or create an instruction template
   - Choose your preferred AI voice
   - Click "Start Session"

4. **Interact with the Model**:
   - Hold spacebar or click the record button to speak
   - Release to send your message
   - Listen to the AI's response
   - Use the Debug tab to monitor the interaction

## How it Works

1. **Audio Input**: Capture user voice input through Gradio's audio interface set to numpy arrays.

2. **WebSocket Connection**: Establish a secure WebSocket connection to OpenAI's realtime API using the provided API key.

3. **Data Serialization**: Convert the audio data to base64 and package it in a JSON format for sending over the WebSocket.

4. **Response Handling**: Receive streamed audio data from the OpenAI server, decode it, and prepare it for playback.

5. **Output**: Play the AI-generated audio response back to the user.

## File Structure

- `main.py`: Main application script to run the Gradio interface.
- `requirements.txt`: Lists the necessary Python libraries to be installed.
- `.env`: Stores environment variables including sensitive API keys.

## Requirements

This project depends on several key libraries:

- `websockets`: For maintaining WebSocket connections.
- `pydub` and `soundfile`: For audio processing.
- `gradio`: For building and managing the web interface.
- `python-dotenv`: For loading environment variables from a `.env` file.

## Acknowledgments

This project builds upon the excellent foundation provided by [NuclearGeekETH's OpenAI Realtime Python Example](https://github.com/nucleargeeketh/openai-realtime-python-example). The original implementation has been enhanced with additional features while maintaining compatibility with the core functionality.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

For any issues or feature requests, please open an issue in this GitHub repository. Happy chatting with AI in real-time!