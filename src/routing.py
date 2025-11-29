"""
routing.py - Fixed routing logic with proper state inspection

CRITICAL: Routing functions must read from state correctly
and return the correct destination node name.
"""

from langchain_core.messages import HumanMessage


def route_from_receptionist(state) -> str:
    """
    Determine next node after receptionist.
    
    Logic:
    - If next_node hint is set to "lookup_patient" -> go to patient_data_retrieval
    - Otherwise -> go directly to clinical_agent
    
    Args:
        state: ChatState with messages, patient_info, next_node
        
    Returns:
        str: Name of the next node ("patient_data_retrieval" or "clinical_agent")
    """
    # Check if receptionist set a routing hint
    next_node_hint = state.get("next_node")
    
    print(f"[ROUTING] route_from_receptionist:")
    print(f"  - next_node hint: {next_node_hint}")
    print(f"  - patient_info exists: {state.get('patient_info') is not None}")
    print(f"  - total messages: {len(state.get('messages', []))}")
    
    # If receptionist detected a patient name and set the hint
    if next_node_hint == "lookup_patient":
        print(f"  -> Routing to: patient_data_retrieval")
        return "patient_data_retrieval"
    
    # Default: go to clinical agent
    print(f"  -> Routing to: clinical_agent")
    return "clinical_agent"


def route_from_lookup(state) -> str:
    """
    Determine next node after patient data retrieval.
    
    After looking up patient data, always go to clinical_agent.
    
    Args:
        state: ChatState with messages and patient_info
        
    Returns:
        str: Always returns "clinical_agent"
    """
    print(f"[ROUTING] route_from_lookup:")
    print(f"  - patient_info exists: {state.get('patient_info') is not None}")
    print(f"  - total messages: {len(state.get('messages', []))}")
    print(f"  -> Routing to: clinical_agent")
    
    return "clinical_agent"


# ============================================================================
# Alternative: More sophisticated routing based on state
# ============================================================================

def route_from_receptionist_advanced(state) -> str:
    """
    Advanced routing with multiple checks.
    
    Use this if you need more complex logic.
    """
    messages = state.get("messages", [])
    patient_info = state.get("patient_info")
    next_node_hint = state.get("next_node")
    
    # Get the latest user message
    latest_user_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            latest_user_msg = msg.content
            break
    
    print(f"[ROUTING] Advanced routing from receptionist:")
    print(f"  - Latest user message: {latest_user_msg[:50] if latest_user_msg else None}...")
    print(f"  - Patient info exists: {patient_info is not None}")
    print(f"  - Next node hint: {next_node_hint}")
    
    # Priority 1: Check explicit routing hint
    if next_node_hint == "lookup_patient":
        print(f"  -> Routing to: patient_data_retrieval (hint)")
        return "patient_data_retrieval"
    
    # Priority 2: If patient info already exists, skip lookup
    if patient_info is not None:
        print(f"  -> Routing to: clinical_agent (patient info exists)")
        return "clinical_agent"
    
    # Priority 3: Check if user introduced themselves
    if latest_user_msg:
        name_patterns = [
            "my name is",
            "i am",
            "i'm",
            "this is"
        ]
        
        msg_lower = latest_user_msg.lower()
        if any(pattern in msg_lower for pattern in name_patterns):
            print(f"  -> Routing to: patient_data_retrieval (name detected)")
            return "patient_data_retrieval"
    
    # Default: go to clinical agent
    print(f"  -> Routing to: clinical_agent (default)")
    return "clinical_agent"


def route_from_clinical_agent(state) -> str:
    """
    Optional: Routing from clinical agent if you want to add more logic.
    
    Currently, clinical agent uses tools_condition for routing.
    But you could add custom logic here if needed.
    """
    messages = state.get("messages", [])
    
    if not messages:
        return "__end__"
    
    last_message = messages[-1]
    
    # Check if last message has tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "clinical_tools"
    
    return "__end__"
