"""
agents_nodes.py  — Fixed for Mistral API message ordering requirements
"""
from typing import Any
import re
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)
from llm_models import receptionist_llm, clinical_llm
from tools import patient_data_tool, clinical_agent_tools

# ---------------------------
# Tool selection helpers
# ---------------------------
def _score_tool_appropriateness(query: str) -> dict[str, float]:
    query_lower = (query or "").lower()
    scores = {
        "get_patient_discharge_report": 0.0,
        "query_nephrology_docs": 0.0,
        "search_web": 0.0,
    }
    patient_keywords = [
        "my medication",
        "my medications",
        "medications",
        "prescriptions",
        "my discharge",
        "discharge report",
        "discharge summary",
        "my restrictions",
        "dietary restriction",
        "restrictions",
        "my follow-up",
        "follow-up",
        "follow up appointment",
        "warning signs",
        "warning sign",
        "my condition",
        "my health",
        "my doctor",
        "doctor's orders",
        "my lab results",
        "lab results",
        "my diagnosis",
        "my treatment",
        "my care plan",
        "remind me",
        "remember",
        "what did",
        "hospital records",
    ]
    patient_pronouns = ["my", "me", "i have", "i'm on", "i take", "i need"]
    has_patient_pronoun = any(p in query_lower for p in patient_pronouns)
    has_patient_keyword = any(k in query_lower for k in patient_keywords)
    if has_patient_keyword:
        scores["get_patient_discharge_report"] += 0.8
    if has_patient_pronoun and has_patient_keyword:
        scores["get_patient_discharge_report"] += 0.15
    rag_keywords = [
        "what is",
        "what are",
        "explain",
        "how does",
        "how do",
        "treatment",
        "management",
        "mechanism",
        "causes",
        "symptoms",
        "pathophysiology",
        "stages",
        "classification",
        "definition",
        "diagnosis",
        "prognosis",
        "complications",
        "risk factors",
        "medication classes",
        "drug interactions",
        "nephrology",
    ]
    has_rag_keyword = any(k in query_lower for k in rag_keywords)
    if has_rag_keyword:
        scores["query_nephrology_docs"] += 0.8
    if has_rag_keyword and not has_patient_pronoun:
        scores["query_nephrology_docs"] += 0.1
    web_search_keywords = [
        "latest",
        "newest",
        "new",
        "recent",
        "breakthrough",
        "update",
        "research",
        "trial",
        "clinical trial",
        "study",
        "studies",
        "news",
        "publication",
        "article",
        "current",
        "today",
        "2024",
        "2025",
        "latest guideline",
        "recent finding",
    ]
    has_web_keyword = any(k in query_lower for k in web_search_keywords)
    if has_web_keyword:
        scores["search_web"] += 0.8
    # Anti-pattern adjustments
    if has_rag_keyword and not has_patient_pronoun:
        scores["get_patient_discharge_report"] = max(0.0, scores["get_patient_discharge_report"] - 0.3)
    if has_patient_keyword and not has_web_keyword:
        scores["search_web"] = max(0.0, scores["search_web"] - 0.3)
    max_score = max(scores.values()) if any(scores.values()) else 1.0
    if max_score > 1.0:
        for tool in scores:
            scores[tool] = min(1.0, scores[tool] / max_score)
    return scores

def _get_recommended_tools(query: str, threshold: float = 0.3) -> list[str]:
    scores = _score_tool_appropriateness(query)
    recommended = [(tool, score) for tool, score in scores.items() if score >= threshold]
    recommended.sort(key=lambda x: x[1], reverse=True)
    print(f"[TOOL_SELECTION] Query preview: '{(query or '')[:80]}...'")
    for tool, score in recommended:
        print(f"[TOOL_SELECTION]   {tool}: {score:.2f}")
    return [t for t, _ in recommended]

def _validate_tool_invocation(query: str, tool_name: str) -> tuple[bool, str]:
    scores = _score_tool_appropriateness(query)
    tool_score = scores.get(tool_name, 0.0)
    recommended_tools = _get_recommended_tools(query, threshold=0.2)
    if tool_name in recommended_tools[:2]:
        return True, f"Tool '{tool_name}' is appropriate (score: {tool_score:.2f})"
    elif tool_score > 0.5:
        return True, f"Tool '{tool_name}' has reasonable score ({tool_score:.2f}) but alternatives exist: {recommended_tools}"
    else:
        best_tool = recommended_tools[0] if recommended_tools else "None"
        explanation = f"Tool '{tool_name}' not recommended for this query (score: {tool_score:.2f}). Consider using: {best_tool}"
        print(f"[TOOL_VALIDATION_WARNING] {explanation}")
        return False, explanation

# ---------------------------
# Small utility helpers
# ---------------------------
def _extract_patient_name(text: str) -> str | None:
    if not text:
        return None
    patterns = [
        r"(?:my name is|i am|i'm)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"^(?:hi|hello|hey)[,\s]+([A-Z][a-z]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None

def _get_latest_user_message(messages: list[BaseMessage]) -> str:
    for message in reversed(messages or []):
        if isinstance(message, HumanMessage):
            return message.content
    return ""

def _sanitize_messages_for_mistral(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    Ensures message ordering is compatible with Mistral API requirements.
    
    Mistral API requirements:
    - ToolMessage (role='tool') must immediately follow an AIMessage with tool_calls
    - Cannot have ToolMessage after HumanMessage directly
    - Must maintain proper conversation flow
    """
    sanitized = []
    i = 0
    
    while i < len(messages):
        msg = messages[i]
        
        # Handle SystemMessage - always valid at start
        if isinstance(msg, SystemMessage):
            sanitized.append(msg)
            i += 1
            continue
        
        # Handle HumanMessage - always valid
        if isinstance(msg, HumanMessage):
            sanitized.append(msg)
            i += 1
            continue
        
        # Handle AIMessage
        if isinstance(msg, AIMessage):
            # Check if this AIMessage has tool_calls
            has_tool_calls = bool(getattr(msg, "tool_calls", None))
            
            if has_tool_calls:
                # This AIMessage expects tool results
                # Look ahead for corresponding ToolMessages
                sanitized.append(msg)
                i += 1
                
                # Collect all consecutive ToolMessages that follow
                tool_messages = []
                while i < len(messages) and isinstance(messages[i], ToolMessage):
                    tool_messages.append(messages[i])
                    i += 1
                
                # Add all tool messages
                sanitized.extend(tool_messages)
            else:
                # Regular AIMessage without tool calls
                sanitized.append(msg)
                i += 1
            continue
        
        # Handle ToolMessage - this is the problematic case
        if isinstance(msg, ToolMessage):
            # ToolMessage found but no preceding AIMessage with tool_calls
            # This violates Mistral's requirements
            print(f"[SANITIZE] Warning: Found ToolMessage without preceding AIMessage with tool_calls. Skipping.")
            i += 1
            continue
        
        # Unknown message type - keep it
        sanitized.append(msg)
        i += 1
    
    # Final validation: ensure no ToolMessage is orphaned
    validated = []
    for j, msg in enumerate(sanitized):
        if isinstance(msg, ToolMessage):
            # Check if previous message is AIMessage with tool_calls
            if j > 0 and isinstance(sanitized[j-1], AIMessage):
                prev_ai = sanitized[j-1]
                if getattr(prev_ai, "tool_calls", None):
                    validated.append(msg)
                else:
                    print(f"[SANITIZE] Removing orphaned ToolMessage at position {j}")
            elif j > 0 and isinstance(sanitized[j-1], ToolMessage):
                # Multiple consecutive ToolMessages are ok if they follow an AIMessage with tool_calls
                validated.append(msg)
            else:
                print(f"[SANITIZE] Removing orphaned ToolMessage at position {j}")
        else:
            validated.append(msg)
    
    return validated

# ---------------------------
# Node: receptionist_agent_node
# ---------------------------
async def receptionist_agent_node(state: dict) -> dict:
    """
    Greet user and detect patient name.
    """
    if state.get("receptionist_done"):
        print("[RECEP] receptionist_agent_node skipped because receptionist_done=True")
        return {}
    
    messages = state.get("messages", [])
    latest_user = _get_latest_user_message(messages)
    extracted_name = _extract_patient_name(latest_user)
    
    result: dict[str, Any] = {}
    
    if extracted_name and not state.get("patient_info"):
        response = AIMessage(content="Thank you for introducing yourself. Let me look up your medical records.")
        result["messages"] = [response]
        result["next_node"] = "lookup_patient"
        result["patient_name"] = extracted_name
        result["receptionist_done"] = True
        result["stage"] = "lookup"
        print(f"[RECEP] extracted_name='{extracted_name}' -> routing to lookup")
    else:
        response = AIMessage(content="Hello! How can I assist you today?")
        result["messages"] = [response]
        result["receptionist_done"] = True
        
        if state.get("patient_info"):
            result["stage"] = "clinical"
            print("[RECEP] patient_info present -> stage set to clinical")
        else:
            result["stage"] = "clinical"
            print("[RECEP] no patient_info -> defaulting stage to clinical")
    
    return result

# ---------------------------
# Node: patient_data_retrieval_node
# ---------------------------
async def patient_data_retrieval_node(state: dict) -> dict:
    """
    Look up patient information.
    """
    if state.get("patient_verified"):
        print("[LOOKUP] patient_data_retrieval_node skipped because patient_verified=True")
        return {}
    
    patient_name = state.get("patient_name")
    if not patient_name:
        latest_user = _get_latest_user_message(state.get("messages", []))
        patient_name = _extract_patient_name(latest_user)
    
    if not patient_name:
        msg = AIMessage(content="I need your name to look up your records. Could you please introduce yourself?")
        return {"messages": [msg]}
    
    try:
        print(f"[LOOKUP] Calling patient_data_tool for '{patient_name}'")
        patient_data = await patient_data_tool.ainvoke(patient_name)
        confirmation = AIMessage(content="I found your records. Now I'll connect you with clinical support.")
        
        return {
            "messages": [confirmation],
            "patient_info": patient_data,
            "patient_verified": True,
            "stage": "clinical",
        }
    except Exception as error:
        print(f"[LOOKUP] patient_data_tool error: {error}")
        err_msg = AIMessage(content=f"I couldn't find records for {patient_name}. Please check the name or contact support.")
        return {"messages": [err_msg]}

# ---------------------------
# Node: clinical_agent_node
# ---------------------------
async def clinical_agent_node(state: dict) -> dict:
    """
    Clinical assistant with proper message ordering for Mistral API.
    """
    stage = state.get("stage")
    patient_info = state.get("patient_info")
    
    if stage and stage != "clinical":
        print(f"[CLIN] clinical_agent_node skipped because stage={stage}")
        return {}
    
    messages = state.get("messages", [])
    print(f"[CLIN] Running clinical_agent_node: messages={len(messages)}, patient_info={'yes' if patient_info else 'no'}")
    
    # Prepare system instruction
    system_instruction = (
        "You are an expert clinical assistant specialized in post-discharge patient support and nephrology. "
        "Follow safety guidance and remind users to consult their healthcare provider. "
        "If patient-specific information is required (meds, doses, restrictions) and patient data is available, use it."
    )
    llm_messages = [SystemMessage(content=system_instruction)]
    
    # Add patient context if available
    if patient_info:
        patient_context = f"Patient Data: {str(patient_info)[:400]}"
        llm_messages.append(HumanMessage(content=patient_context))
        llm_messages.append(AIMessage(content="I have reviewed the patient data and can provide guidance."))
    
    # Append conversation history with sanitization for Mistral API
    for m in messages:
        if isinstance(m, (HumanMessage, AIMessage, ToolMessage)):
            llm_messages.append(m)
    
    # CRITICAL FIX: Sanitize messages to ensure proper ordering for Mistral API
    llm_messages = _sanitize_messages_for_mistral(llm_messages)
    print(f"[CLIN] After sanitization: {len(llm_messages)} messages")
    
    # Bind tools if available
    llm = clinical_llm()
    has_tool_results = any(isinstance(m, ToolMessage) for m in messages)
    
    if clinical_agent_tools and not has_tool_results:
        try:
            llm = llm.bind_tools(clinical_agent_tools)
            print("[CLIN] Bound clinical tools to LLM")
        except Exception as e:
            print(f"[CLIN] Warning: failed to bind tools: {e}")
    
    # Ensure we end with a user message for the model to respond to.
    # IMPORTANT: Do not place a `HumanMessage` after any `ToolMessage` (Mistral rejects 'user' after 'tool').
    if not llm_messages or not isinstance(llm_messages[-1], HumanMessage):
        trailing_prompt = HumanMessage(content="Please provide a clear, concise clinical response to the user's request.")

        # If there are tool messages in the conversation, insert the user prompt before the first ToolMessage
        # so that no `HumanMessage` appears after a `ToolMessage` (which triggers Mistral's "Unexpected role 'user' after role 'tool'" error).
        first_tool_idx = next((i for i, m in enumerate(llm_messages) if isinstance(m, ToolMessage)), None)
        if first_tool_idx is not None:
            llm_messages.insert(first_tool_idx, trailing_prompt)
        else:
            llm_messages.append(trailing_prompt)
    
    # Streaming accumulation
    response_content = ""
    tool_calls_dict: dict[int, dict] = {}
    response_metadata = {}
    
    try:
        print("[CLIN] Starting LLM streaming...")
        chunk_count = 0
        
        async for chunk in llm.astream(llm_messages):
            chunk_count += 1
            
            if hasattr(chunk, "content") and chunk.content:
                response_content += chunk.content
            
            if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                for tool_call in chunk.tool_calls:
                    tc_index = tool_call.get("index", 0)
                    if tc_index not in tool_calls_dict:
                        tool_calls_dict[tc_index] = {
                            "name": tool_call.get("name", ""),
                            "args": tool_call.get("args", {}),
                            "id": tool_call.get("id", ""),
                            "type": tool_call.get("type", "function"),
                        }
                    else:
                        existing = tool_calls_dict[tc_index]
                        if tool_call.get("name"):
                            existing["name"] = tool_call["name"]
                        if tool_call.get("args"):
                            if isinstance(existing["args"], dict) and isinstance(tool_call["args"], dict):
                                existing["args"].update(tool_call["args"])
                            else:
                                existing["args"] = tool_call.get("args")
                        if tool_call.get("id"):
                            existing["id"] = tool_call.get("id")
            
            if hasattr(chunk, "response_metadata") and chunk.response_metadata:
                response_metadata.update(chunk.response_metadata)
        
        print(f"[CLIN] LLM stream finished after {chunk_count} chunks")
        
        # Build final AIMessage
        tool_calls_list = []
        if tool_calls_dict:
            tool_calls_list = [tool_calls_dict[i] for i in sorted(tool_calls_dict.keys())]
            print(f"[CLIN] Aggregated {len(tool_calls_list)} tool calls")
            
            latest_user = _get_latest_user_message(messages)
            if latest_user:
                for tc in tool_calls_list:
                    ok, explain = _validate_tool_invocation(latest_user, tc.get("name", "unknown"))
                    print(f"[CLIN][TOOL_VALIDATION] {explain}")
        
        if tool_calls_list:
            final_msg = AIMessage(content=response_content or "", tool_calls=tool_calls_list)
        else:
            final_msg = AIMessage(content=response_content or "")
        
        if response_metadata:
            final_msg.response_metadata = response_metadata
        
        if final_msg.content:
            preview = final_msg.content[:120] + "..." if len(final_msg.content) > 120 else final_msg.content
            print(f"[CLIN] Final response preview: {preview}")
        
        print("[DEBUG] No tool calls - ending conversation")
            
    except Exception as exc:
        print(f"[CLIN] Streaming error: {exc}")
        import traceback
        traceback.print_exc()
        final_msg = AIMessage(content=f"I encountered an issue generating the response. Please try again. (Error: {str(exc)[:120]})")
    
    return {"messages": [final_msg]}

# ---------------------------
# Optional: helper stream function
# ---------------------------
async def stream_clinical_response(state: dict, callback=None):
    """
    Alternative streaming function with message sanitization.
    """
    messages = state.get("messages", [])
    patient_info = state.get("patient_info")
    llm = clinical_llm()
    
    system_instruction = (
        "You are an expert clinical assistant specializing in post-discharge patient support and nephrology. "
        "Follow safety guidance and remind users to consult their healthcare provider."
    )
    llm_messages = [SystemMessage(content=system_instruction)]
    
    if patient_info:
        patient_context = f"Patient Data: {str(patient_info)[:400]}"
        llm_messages.append(HumanMessage(content=patient_context))
        llm_messages.append(AIMessage(content="I have reviewed the patient data."))
    
    for m in messages:
        if isinstance(m, (HumanMessage, AIMessage, ToolMessage)):
            llm_messages.append(m)
    
    # CRITICAL: Sanitize messages for Mistral API
    llm_messages = _sanitize_messages_for_mistral(llm_messages)
    
    has_tool_results = any(isinstance(m, ToolMessage) for m in messages)
    if clinical_agent_tools and not has_tool_results:
        try:
            llm = llm.bind_tools(clinical_agent_tools)
        except Exception:
            pass

    # Ensure we do not place a `HumanMessage` after any `ToolMessage` (Mistral rejects 'user' after 'tool').
    if not llm_messages or not isinstance(llm_messages[-1], HumanMessage):
        trailing_prompt = HumanMessage(content="Please provide a clear, concise clinical response to the user's request.")
        first_tool_idx = next((i for i, m in enumerate(llm_messages) if isinstance(m, ToolMessage)), None)
        if first_tool_idx is not None:
            llm_messages.insert(first_tool_idx, trailing_prompt)
        else:
            llm_messages.append(trailing_prompt)

    async for chunk in llm.astream(llm_messages):
        if callback:
            await callback(chunk)
        yield chunk