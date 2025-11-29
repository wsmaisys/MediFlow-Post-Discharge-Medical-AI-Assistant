# Clinical Agent Chat UI

A simple web-based chat interface for the Clinical Agent.

## Setup

1. **Install dependencies:**

   ```bash
   pip install flask flask-cors
   ```

2. **Ensure all other dependencies are installed:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Make sure your `.env` file contains:
   ```
   MISTRAL_API_KEY=your_mistral_api_key_here
   ```

## Running the Application

1. **Start the Flask server:**

   ```bash
   python app.py
   ```

2. **Open your browser:**
   Navigate to `https://mediflow-ai-medical-assistant-785629432566.us-central1.run.app`

3. **Start chatting:**
   - Type your questions in the input field
   - Press Enter or click Send
   - The agent will respond using the MCP tools and search capabilities

## Features

- **Real-time chat interface** with modern UI
- **Thread management** - conversations are tracked by thread ID
- **Error handling** - clear error messages if something goes wrong
- **Loading indicators** - visual feedback while the agent processes requests
- **Responsive design** - works on desktop and mobile devices

## API Endpoints

- `GET /` - Serve the chat UI
- `GET /api/health` - Health check endpoint
- `POST /api/chat` - Send a chat message
  ```json
  {
    "message": "What are ACE inhibitors used for?",
    "thread_id": "optional_thread_id"
  }
  ```
- `GET /api/threads` - List all conversation threads

## Troubleshooting

- **"Network error"** - Make sure the Flask server is running (`python app.py`)
- **"No response from agent"** - Check that your MISTRAL_API_KEY is set correctly
- **Import errors** - Make sure all dependencies are installed: `pip install -r requirements.txt`
