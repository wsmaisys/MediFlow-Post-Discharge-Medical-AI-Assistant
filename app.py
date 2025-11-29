#!/usr/bin/env python3
"""
Flask backend API for Clinical Agent Chat UI
"""

# Import python_multipart early to prevent Starlette deprecation warning
try:
    import python_multipart  # noqa: F401
except ImportError:
    pass

import sys # Standard library import for system-specific parameters and functions
import os # Standard library import for operating system interactions
import uuid # Standard library import for generating unique identifiers
import traceback # Standard library import for printing stack traces
import asyncio # Standard library import for async/await support

# Ensure src directory is discoverable for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# Third-party imports
from fastapi import FastAPI, HTTPException # Third-party import for FastAPI framework and HTTPException for error handling
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, RedirectResponse # Third-party imports for HTML, JSON, redirect and streaming responses
from fastapi.staticfiles import StaticFiles # Third-party import for serving static files
from fastapi.middleware.cors import CORSMiddleware # Third-party import for Cross-Origin Resource Sharing middleware
import uvicorn # Third-party import for running the ASGI server
from pydantic import BaseModel # Third-party import for data validation and settings management
from langchain_core.messages import HumanMessage # Import HumanMessage from langchain
import json # Standard library for JSON serialization
import logging

# Local modular imports
from chatbot_main import initialize_chatbot # Import chatbot initialization from modular structure
from utils_async import run_async # Import async utility for running async functions in sync context
from contextlib import asynccontextmanager # For lifespan context manager

# Helper function to convert non-serializable objects to JSON-serializable format
def make_serializable(obj):
    """Recursively convert objects to JSON-serializable format."""
    if hasattr(obj, "__dict__"):
        # Convert objects with __dict__ to their content if they have a "content" attribute
        if hasattr(obj, "content"):
            return obj.content
        return str(obj)
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_serializable(item) for item in obj)
    else:
        return obj

# Global chatbot instance
chatbot = None

# Configure basic logging for the application
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app startup and shutdown."""
    global chatbot
    # Startup
    print("Starting Clinical Agent API server...")
    print("Access the UI at: http://localhost:5000")
    print("API endpoints:")
    print("  - GET  /api/health - Health check")
    print("  - POST /api/chat - Send chat message")
    print("  - POST /api/chat/stream - Stream chat response")
    print("  - GET  /api/threads - List all threads")
    chatbot = await initialize_chatbot()
    print("Chatbot initialized successfully")
    yield
    # Shutdown
    print("Shutting down Clinical Agent API...")

# Initialize FastAPI app with lifespan
app = FastAPI( # Create a new FastAPI application instance
    title="Clinical Agent API", # Set API title
    description="Backend API for the Clinical Agent Chat UI, powered by LangChain and FastAPI.", # Set API description
    version="1.0.0", # Set API version
    lifespan=lifespan # Use lifespan context manager
)

# Configure CORS
app.add_middleware( # Add CORS middleware to the application
    CORSMiddleware, # Middleware class for CORS
    allow_origins=["http://localhost:5000", "http://127.0.0.1:5000"],  # Allow specified origins to access the API (adjust as needed in production)
    allow_credentials=True, # Allow credentials (e.g., cookies, authorization headers) to be sent with requests
    allow_methods=["*"], # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all HTTP headers
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static") # Mount the "static" directory to serve static files (e.g., index.html, CSS, JS)

@app.get("/api/health") # Decorator to define a GET endpoint for health check
async def health(): # Asynchronous function to handle health check requests
    """Health check endpoint"""
    return {"status": "ok", "message": "Clinical Agent API is running"} # Return a simple status message

class ChatMessage(BaseModel): # Pydantic model for validating incoming chat messages
    """Represents a chat message from the user."""
    message: str # The content of the chat message
    thread_id: str | None = None # Optional thread ID for continuing a conversation

@app.post("/api/chat") # Decorator to define a POST endpoint for chat messages
async def chat(chat_message: ChatMessage): # Asynchronous function to handle incoming chat messages
    """Handle chat messages by invoking the clinical agent chatbot."""
    try: # Error handling block
        if not chatbot:
            raise HTTPException(status_code=503, detail="Chatbot not initialized")
        
        message = chat_message.message # Extract message content from the validated request body
        thread_id = chat_message.thread_id # Extract thread ID from the validated request body
        
        if not message: # Validate if message content is empty
            raise HTTPException(status_code=400, detail="Message is required") # Raise HTTP 400 error if message is empty
        
        # Generate thread_id if not provided
        if not thread_id: # Check if thread_id is not provided
            thread_id = f"thread_{uuid.uuid4().hex[:8]}" # Generate a new unique thread ID
        
        # Create message for the chatbot
        messages = [HumanMessage(content=message)] # Wrap the user's message in a HumanMessage object
        config = {"configurable": {"thread_id": thread_id}} # Configuration for the chatbot, including the thread ID
        
        # Invoke chatbot: support both coroutine and immediate return values
        _maybe_coro = chatbot.ainvoke({"messages": messages}, config)
        if asyncio.iscoroutine(_maybe_coro):
            # run_async blocks until the coroutine completes and returns the result
            response = run_async(_maybe_coro)
        else:
            # already a concrete result (dict-like)
            response = _maybe_coro
        
        # Extract final message from the chatbot's response
        final_messages = response.get("messages", []) # Get the list of messages from the chatbot's response
        if final_messages: # If there are any messages in the response
            last_msg = final_messages[-1] # Get the last message from the list
            content = getattr(last_msg, "content", None) or str(last_msg) # Extract the content of the last message
            # Log the final response to the terminal for visibility
            try:
                logger.info("Final response (thread %s): %s", thread_id, content)
            except Exception:
                print(f"Final response (thread {thread_id}): {content}")

            return JSONResponse(content={ # Return a JSON response with the chatbot's reply
                "response": content, # The content of the chatbot's response
                "thread_id": thread_id, # The thread ID of the conversation
                "status": "success" # Status of the operation
            }) # End of JSONResponse
        
        # If no final messages were found in the response
        return JSONResponse(content={ # Return a JSON error response
            "error": "No response from agent", # Error message
            "thread_id": thread_id # The thread ID
        }, status_code=500) # HTTP 500 status code for internal server error
            
    except HTTPException as e: # Catch FastAPI's HTTPException directly
        raise e # Re-raise HTTPException without modification
    except Exception as e: # Catch any other unexpected exceptions
        error_trace = traceback.format_exc() # Get the full traceback of the exception
        print(f"Error in chat endpoint: {error_trace}") # Print the error traceback to console
        raise HTTPException( # Raise an HTTP 500 error for unexpected issues
            status_code=500, # HTTP 500 status code
            detail={ # Detailed error information
                "error": str(e), # String representation of the error
                "trace": error_trace # Full traceback for debugging
            }
        ) from e # Explicitly chain the original exception

@app.post("/api/chat/stream") # Decorator to define a POST endpoint for streaming chat responses
async def chat_stream(chat_message: ChatMessage): # Asynchronous function to handle streaming chat messages
    """Stream chat responses from the clinical agent chatbot using Server-Sent Events."""
    try: # Error handling block
        if not chatbot:
            raise HTTPException(status_code=503, detail="Chatbot not initialized")
        
        message = chat_message.message # Extract message content from the validated request body
        thread_id = chat_message.thread_id # Extract thread ID from the validated request body
        
        if not message: # Validate if message content is empty
            raise HTTPException(status_code=400, detail="Message is required") # Raise HTTP 400 error if message is empty
        
        # Generate thread_id if not provided
        if not thread_id: # Check if thread_id is not provided
            thread_id = f"thread_{uuid.uuid4().hex[:8]}" # Generate a new unique thread ID
        
        # Create message for the chatbot
        messages = [HumanMessage(content=message)] # Wrap the user's message in a HumanMessage object
        config = {"configurable": {"thread_id": thread_id}} # Configuration for the chatbot, including the thread ID
        
        async def stream_response():
            """Generator function for streaming responses using Server-Sent Events (SSE)."""
            try:
                # Accumulate the best/longest AI-generated content seen during streaming
                full_content = ""
                # Stream the chatbot response using astream for real-time chunks
                async for event in chatbot.astream({"messages": messages}, config):
                    # Convert non-serializable objects to JSON-serializable format
                    serializable_event = make_serializable(event)

                    # Inspect event for any AI assistant messages and update full_content
                    try:
                        if isinstance(serializable_event, dict):
                            msgs = serializable_event.get("messages")
                            if isinstance(msgs, list):
                                for m in msgs:
                                    if isinstance(m, str):
                                        candidate = m
                                    elif isinstance(m, dict):
                                        candidate = m.get("content") or m.get("text") or ""
                                    else:
                                        candidate = str(m)
                                    if candidate and len(candidate) > len(full_content):
                                        full_content = candidate
                            else:
                                # Iterate node-like entries (some backends emit nodes)
                                for node_value in serializable_event.values():
                                    if isinstance(node_value, dict) and isinstance(node_value.get("messages"), list):
                                        for m in node_value.get("messages", []):
                                            if isinstance(m, str):
                                                candidate = m
                                            elif isinstance(m, dict):
                                                candidate = m.get("content") or m.get("text") or ""
                                            else:
                                                candidate = str(m)
                                            if candidate and len(candidate) > len(full_content):
                                                full_content = candidate
                    except Exception:
                        # Best-effort: ignore parsing errors and continue streaming
                        pass

                    # Format as SSE (Server-Sent Events)
                    chunk_data = {
                        "type": "chunk",
                        "event": serializable_event,
                        "timestamp": str(__import__('datetime').datetime.now().isoformat())
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"

                # Log final AI response seen during streaming (if any)
                try:
                    if full_content:
                        logger.info("Final streamed response (thread %s): %s", thread_id, full_content)
                except Exception:
                    print(f"Final streamed response (thread {thread_id}): {full_content}")

                # Send final message indicating completion
                yield f"data: {json.dumps({'type': 'complete', 'thread_id': thread_id})}\n\n"

            except Exception as e:
                error_data = {
                    "type": "error",
                    "error": str(e),
                    "thread_id": thread_id
                }
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in stream endpoint: {error_trace}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "trace": error_trace
            }
        ) from e

@app.get("/api/threads") # Decorator to define a GET endpoint for retrieving all threads
async def get_threads(): # Asynchronous function to retrieve all conversation threads
    """Get all conversation threads from the checkpointer."""
    try: # Error handling block
        if not chatbot:
            raise HTTPException(status_code=503, detail="Chatbot not initialized")
        
        # Get all threads from the checkpointer
        # The checkpointer stores state with thread IDs as keys
        # This is a simplified version - in production you might need to query the database directly
        threads = []
        
        # For now, return empty list as we don't have direct access to all thread IDs
        # In production, you would query the SQLite database directly
        return { # Return a dictionary that FastAPI will automatically convert to JSON
            "threads": threads, # List of thread IDs
            "count": len(threads) # Number of threads found
        }
    except HTTPException as e:
        raise e
    except Exception as e: # Catch any exceptions during thread retrieval
        raise HTTPException(status_code=500, detail=str(e)) from e # Raise an HTTP 500 error with the exception details

@app.get("/") # Decorator to define the root GET endpoint
async def read_root(): # Asynchronous function to serve the main HTML page
    """Serves the main `index.html` file for the chat UI."""
    with open("static/index.html", "r", encoding="utf-8") as f: # Open the index.html file with UTF-8 encoding
        content = f.read() # Read the content of the file
    return HTMLResponse(content=content, status_code=200) # Return the HTML content with a 200 OK status


@app.get("/index.html")
async def serve_index_html():
    """Serve `static/index.html` at `/index.html` to support links that request it directly."""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content, status_code=200)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found")


# Note: we intentionally do not expose `/patients.html` directly. Keep a
# single canonical route `/patients` which serves the patients page.


@app.get("/api/documentation")
async def serve_api_documentation():
    """Serve the API documentation HTML page at `/api/documentation`."""
    try:
        with open("static/api_documentation.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content, status_code=200)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="api_documentation.html not found")


# Removed `/patients` HTML route to keep a minimal API surface.
# The canonical API endpoint for patient data remains `/api/patients`.


@app.get("/data/patients.json")
async def serve_patients_file():
    """Serve the bundled `data/patients.json` file for the static UI.

    This endpoint mirrors a static file path so client-side pages can fetch
    `/data/patients.json` without requiring the `data` folder to be mounted.
    """
    try:
        with open("data/patients.json", "r", encoding="utf-8") as f:
            patients = json.load(f)
        return JSONResponse(content=patients)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="patients.json not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/patients")
async def api_get_patients():
    """API-friendly endpoint returning the same dummy patient data.
    """
    try:
        with open("static/patients.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content, status_code=200)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="patients.html not found")


if __name__ == '__main__': # Standard boilerplate to run the application directly
    print("Starting Clinical Agent API server...") # Log message indicating server startup
    print("Access the UI at: http://localhost:5000") # Provide the URL for accessing the UI
    print("API endpoints:") # List available API endpoints
    print("  - GET  /api/health - Health check") # Health check endpoint description
    print("  - POST /api/chat - Send chat message") # Chat message endpoint description
    print("  - GET  /api/threads - List all threads") # List threads endpoint description
    uvicorn.run(app, host="0.0.0.0", port=5000) # Run the FastAPI application using Uvicorn on all available network interfaces and port 5000

