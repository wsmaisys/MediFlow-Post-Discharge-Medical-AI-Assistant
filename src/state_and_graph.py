"""
state_and_graph.py - Fixed state schema with proper message handling

CRITICAL FIX: Use add_messages reducer to append messages correctly
This ensures conversation history flows through all nodes.
"""

from typing import Annotated, TypedDict, Optional
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    """
    State schema for the chatbot conversation.
    
    IMPORTANT: The 'messages' field uses add_messages reducer.
    This means when nodes return {"messages": [new_msg]}, the new message
    is APPENDED to existing messages, not replaced.
    
    Without the reducer, each node would overwrite previous messages!
    """
    # Annotated with add_messages reducer for proper message accumulation
    messages: Annotated[list[BaseMessage], add_messages]
    
    # Patient information retrieved from database
    patient_info: Optional[dict]
    
    # Routing hint for conditional edges (optional)
    next_node: Optional[str]


# Initialize the state graph with our schema
graph = StateGraph(ChatState)

# Initialize checkpointer for conversation persistence
checkpointer = MemorySaver()


# ============================================================================
# Alternative: If you want to explicitly see what's happening
# ============================================================================

def debug_state_reducer(existing: list, new: list) -> list:
    """
    Custom reducer that logs state changes (for debugging).
    
    This does the same thing as add_messages but with logging.
    Use this temporarily if you need to debug state flow.
    """
    print(f"[STATE DEBUG] Existing messages: {len(existing)}")
    print(f"[STATE DEBUG] New messages: {len(new)}")
    result = existing + new
    print(f"[STATE DEBUG] Combined messages: {len(result)}")
    return result


# Uncomment this to use debug reducer instead:
# class ChatStateDebug(TypedDict):
#     messages: Annotated[list[BaseMessage], debug_state_reducer]
#     patient_info: Optional[dict]
#     next_node: Optional[str]