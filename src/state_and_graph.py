"""
state_and_graph.py - Fixed state schema with safe, optional stage flags

This file intentionally keeps the same exported symbols:
- ChatState (TypedDict)
- graph (StateGraph)
- checkpointer (MemorySaver)

We add optional state fields (stage, receptionist_done, patient_verified)
which are backward-compatible and safe: existing code that ignores them
continues to work. The add_messages reducer is preserved for 'messages'.
"""

from typing import Annotated, TypedDict, Optional
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# ChatState schema used by the rest of the application.
# NOTE: New fields (stage, receptionist_done, patient_verified) are optional
# and default to None/False — this keeps backward compatibility.
class ChatState(TypedDict):
    """
    State schema for the chatbot conversation.

    Fields:
      - messages: list of BaseMessage with add_messages reducer (append semantics)
      - patient_info: optional dict with patient lookup results
      - next_node: optional routing hint set by nodes
      - stage: optional string enum controlling dispatcher ("reception"|"lookup"|"clinical")
      - receptionist_done: optional flag set once receptionist work is completed
      - patient_verified: optional flag set once patient lookup/verification succeeded
    """
    # messages uses the add_messages reducer so nodes append rather than overwrite
    messages: Annotated[list[BaseMessage], add_messages]

    # existing optional fields
    patient_info: Optional[dict]
    next_node: Optional[str]

    # new optional flags (backwards compatible)
    stage: Optional[str]
    receptionist_done: Optional[bool]
    patient_verified: Optional[bool]


# Initialize the StateGraph and checkpointer (names preserved)
graph = StateGraph(ChatState)
checkpointer = MemorySaver()


# For development/debugging: optional custom debug reducer exists but is not active
def debug_state_reducer(existing: list, new: list) -> list:
    """
    Custom reducer that logs state changes (for debugging).
    Behaves like add_messages plus prints; useful temporarily.
    """
    print(f"[STATE DEBUG] Existing messages: {len(existing)}")
    print(f"[STATE DEBUG] New messages: {len(new)}")
    result = existing + new
    print(f"[STATE DEBUG] Combined messages: {len(result)}")
    return result

# If you want to temporarily enable the debug reducer, uncomment below and update
# the ChatState type references in your code to use ChatStateDebug.
# class ChatStateDebug(TypedDict):
#     messages: Annotated[list[BaseMessage], debug_state_reducer]
#     patient_info: Optional[dict]
#     next_node: Optional[str]
#     stage: Optional[str]
#     receptionist_done: Optional[bool]
#     patient_verified: Optional[bool]