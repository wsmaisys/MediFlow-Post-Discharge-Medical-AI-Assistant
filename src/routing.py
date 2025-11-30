"""
routing.py - Safe routing helpers that consult persisted state (non-destructive)

This module preserves the previously exported names:
- route_from_receptionist
- route_from_lookup
- route_from_receptionist_advanced
- route_receptionist (alias)

It also adds `route_from_start` which is used as a dispatcher from START.
The routing functions are defensive: they prefer explicit state flags
(receptionist_done, patient_info) and next_node hints, avoiding heuristics
that would re-run receptionist on follow-ups.
"""

from langchain_core.messages import HumanMessage
from typing import Any

def route_from_start(state: Any) -> str:
    """
    Dispatcher used as the START -> conditional routing entry.

    Decision logic (priority):
      1. If state.stage is set, route according to its value:
         - "reception" -> receptionist
         - "lookup"    -> patient_data_retrieval
         - "clinical"  -> clinical_agent
      2. If receptionist_done == True and patient_info exists -> clinical_agent
      3. If next_node hint requests lookup -> patient_data_retrieval
      4. Default -> receptionist (first-time sessions)
    """
    # Avoid KeyError by using safe gets
    stage = state.get("stage")
    receptionist_done = state.get("receptionist_done", False)
    patient_info = state.get("patient_info")
    next_hint = state.get("next_node")

    print("[ROUTING] route_from_start: stage=%s, receptionist_done=%s, patient_info=%s, next_node=%s" %
          (repr(stage), repr(receptionist_done), "present" if patient_info else "none", repr(next_hint)))

    if stage:
        # Trust explicit stage first
        if stage == "reception":
            print("  -> START routing to: receptionist (stage)")
            return "receptionist"
        if stage == "lookup":
            print("  -> START routing to: patient_data_retrieval (stage)")
            return "patient_data_retrieval"
        if stage == "clinical":
            print("  -> START routing to: clinical_agent (stage)")
            return "clinical_agent"

    # If receptionist already completed and we have patient info, go to clinical
    if receptionist_done and patient_info:
        print("  -> START routing to: clinical_agent (receptionist_done & patient_info)")
        return "clinical_agent"

    # Respect explicit hint to lookup
    if next_hint == "lookup_patient" or next_hint == "patient_data_retrieval":
        print("  -> START routing to: patient_data_retrieval (hint)")
        return "patient_data_retrieval"

    # Default: first-time sessions should go to receptionist
    print("  -> START routing to: receptionist (default)")
    return "receptionist"


def route_from_receptionist(state) -> str:
    """
    Determine next node after receptionist.
    Decision logic:
      - If receptionist has set explicit next_node hint -> follow it
      - If patient_info is already present or receptionist_done==True -> clinical_agent
      - If latest user message looks like an introduction -> patient_data_retrieval
      - Default -> clinical_agent
    """
    next_node_hint = state.get("next_node")
    patient_info = state.get("patient_info")
    receptionist_done = state.get("receptionist_done", False)

    messages = state.get("messages", [])
    latest_user_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            latest_user_msg = msg.content
            break

    print(f"[ROUTING] route_from_receptionist: hint={next_node_hint}, patient_info={'yes' if patient_info else 'no'}, receptionist_done={receptionist_done}, latest_user_msg_preview={(latest_user_msg[:50]+'...') if latest_user_msg else None}")

    # Respect explicit routing hints first
    if next_node_hint == "lookup_patient" or next_node_hint == "patient_data_retrieval":
        print("  -> Routing to: patient_data_retrieval (hint)")
        return "patient_data_retrieval"

    # If receptionist already completed and patient_info exists, skip lookup
    if receptionist_done and patient_info:
        print("  -> Routing to: clinical_agent (receptionist already done)")
        return "clinical_agent"

    # If a name/introduction is detected in the latest user message, request lookup
    if latest_user_msg:
        low = latest_user_msg.lower()
        name_patterns = ["my name is ", "i am ", "i'm ", "this is "]
        if any(p in low for p in name_patterns):
            print("  -> Routing to: patient_data_retrieval (name detected)")
            return "patient_data_retrieval"

    # Default to clinical agent otherwise
    print("  -> Routing to: clinical_agent (default)")
    return "clinical_agent"


def route_from_lookup(state) -> str:
    """
    After patient lookup, route to clinical_agent.
    This function logs state and ensures deterministic behavior.
    """
    patient_info = state.get("patient_info")
    print(f"[ROUTING] route_from_lookup: patient_info={'present' if patient_info else 'missing'} -> clinical_agent")
    return "clinical_agent"


def route_from_clinical_agent(state) -> str:
    """
    Optional routing from clinical agent. Keep simple:
    If last assistant message indicates tool calls, go to clinical_tools,
    otherwise end.
    """
    messages = state.get("messages", [])
    if not messages:
        return "__end__"

    last_message = messages[-1]
    # If last assistant output included a tool request object, route to tools
    if hasattr(last_message, "tool_calls") and getattr(last_message, "tool_calls"):
        return "clinical_tools"
    return "__end__"


# Compatibility alias: keep older name for backwards compatibility
route_receptionist = route_from_receptionist