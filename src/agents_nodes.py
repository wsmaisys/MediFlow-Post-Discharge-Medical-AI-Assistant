"""
Fully Integrated Streaming agents_nodes.py

Complete streaming implementation for all workflow nodes:
✓ Token-by-token streaming in clinical_agent_node
✓ Proper tool_calls aggregation across chunks
✓ Streaming-compatible message handling
✓ Efficient chunk processing with error recovery
✓ Debug logging for streaming diagnostics

Each node handles one stage of conversation:
1. receptionist_agent_node - Greet user and detect patient name
2. patient_data_retrieval_node - Look up patient records
3. clinical_agent_node - Answer medical questions with streaming support
"""

import re
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from llm_models import receptionist_llm, clinical_llm
from tools import patient_data_tool, clinical_agent_tools


def _extract_patient_name(text: str) -> str | None:
    """
    Extract patient name from user input.
    
    Looks for patterns like "my name is John" or "I'm Sarah".
    Returns None if no name pattern found.
    """
    if not text:
        return None
    
    # Two common patterns for name introduction
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
    """
    Find the most recent message from the user.
    
    Iterates through messages in reverse to find the last HumanMessage.
    Returns empty string if no user message found.
    """
    for message in reversed(messages or []):
        if isinstance(message, HumanMessage):
            return message.content
    
    return ""


async def receptionist_agent_node(state):
    """
    Node 1: Receptionist greets user and detects if patient needs lookup.
    
    This is the entry point of the conversation.
    Sets next_node to "lookup_patient" if a name was mentioned.
    
    Args:
        state: ChatState with messages
        
    Returns:
        Updated state with greeting message and routing instruction
    """
    messages = state.get("messages", [])
    latest_message = _get_latest_user_message(messages)
    
    # Try to extract patient name from their message
    patient_name = _extract_patient_name(latest_message)
    
    # Start building the response
    result = {}
    
    # If user mentioned their name and we don't have patient info yet
    if patient_name and not state.get("patient_info"):
        result["next_node"] = "lookup_patient"
        response = AIMessage(content="Thank you for introducing yourself. Let me look up your medical records.")
    else:
        # User hasn't introduced themselves yet, or we already have their info
        response = AIMessage(content="Hello! How can I assist you today?")
    
    # IMPORTANT: Return messages to be ADDED, not replaced
    # LangGraph will append these to existing messages
    result["messages"] = [response]
    
    return result


async def patient_data_retrieval_node(state):
    """
    Node 2: Look up patient information in the database.
    
    Uses patient_data_tool to retrieve discharge reports and medical history.
    Returns error message if patient not found.
    
    Args:
        state: ChatState with messages
        
    Returns:
        Updated state with patient_info populated and status message
    """
    messages = state.get("messages", [])
    latest_message = _get_latest_user_message(messages)
    
    # Extract the patient name to look up
    patient_name = _extract_patient_name(latest_message)
    
    if not patient_name:
        error_msg = AIMessage(
            content="I need your name to look up your records. Could you please introduce yourself?"
        )
        return {"messages": [error_msg]}
    
    try:
        # Call the tool to fetch patient data
        patient_data = await patient_data_tool.ainvoke(patient_name)
        
        confirmation = AIMessage(
            content="I found your records. Now I'll connect you with clinical support."
        )
        
        return {
            "messages": [confirmation],
            "patient_info": patient_data
        }
    
    except Exception as error:
        # Patient not found or database error
        error_msg = AIMessage(
            content=f"I couldn't find records for {patient_name}. Please check the name or contact support."
        )
        return {"messages": [error_msg]}


async def clinical_agent_node(state):
    """
    Node 3: Clinical assistant with full streaming support.
    
    STREAMING FEATURES:
    ✓ Token-by-token response generation using astream()
    ✓ Proper tool_calls aggregation across chunks
    ✓ Handles both text and tool call chunks
    ✓ Maintains conversation context with BaseMessage objects
    ✓ Error recovery with graceful fallbacks
    
    IMPORTANT: This node receives ALL previous messages from the state,
    including messages from receptionist and patient_data_retrieval nodes.
    
    Args:
        state: ChatState with messages and optional patient_info
        
    Returns:
        Updated state with clinical response (streamed internally)
    """
    messages = state.get("messages", [])
    patient_info = state.get("patient_info")
    
    # Debug: Log what messages we're receiving
    print(f"[DEBUG] Clinical agent received {len(messages)} messages")
    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__
        content_preview = str(msg.content)[:50] + "..." if len(str(msg.content)) > 50 else str(msg.content)
        print(f"[DEBUG]   Message {i}: {msg_type} - {content_preview}")
    
    # Initialize the LLM
    llm = clinical_llm()
    
    # Build the system message with enhanced clinical prompting following medical agentic best practices
    system_instruction = (
        "You are an expert clinical assistant specializing in post-discharge patient support and nephrology. "
        "You understand the complex multidimensional needs of patients with kidney disease, including medication management, "
        "dietary restrictions, symptom monitoring, and emergency warning signs. "
        "\n\nTOOL USAGE RULES (MANDATORY - FOLLOW STRICTLY):"
        "\n1. PATIENT DATA RETRIEVAL:"
        "\n   - Use ALWAYS when patient mentions: medications, discharge info, restrictions, follow-up, warnings"
        "\n   - Keywords: 'my medication', 'my discharge', 'restrictions', 'follow-up', 'warning signs', 'remind me'"
        "\n   - ALWAYS retrieve patient data FIRST before answering patient-specific questions"
        "\n2. NEPHROLOGY KNOWLEDGE BASE (RAG):"
        "\n   - Use for: disease mechanisms, treatment options, medical education, symptom management, pathophysiology"
        "\n   - Keywords: 'what is', 'explain', 'how does', 'treatment', 'management', 'stages', 'mechanism'"
        "\n3. WEB SEARCH:"
        "\n   - Use for: latest research, recent trials, new treatments, current guidelines, updates"
        "\n   - Keywords: 'latest', 'new', 'recent', 'research', 'trial', 'breakthrough', 'update'"
        "\n\nDECISION LOGIC:"
        "\n- 'my/me/I' + medication/restriction/warning → Patient Data Tool"
        "\n- 'what is/explain/how' + medical term → RAG Tool"
        "\n- 'latest/new research/treatment' → Web Search Tool"
        "\n- Multi-part questions: invoke appropriate tool for each part"
        "\n\nRESPONSE GUIDELINES:"
        "\n- Begin with patient-specific context from their discharge data"
        "\n- Provide evidence-based information with specifics (drug names, doses, routes)"
        "\n- Contextualize general medical info to patient's condition"
        "\n- Use detailed responses in clear language, maximum 20 sentences"
        "\n- ALWAYS remind to consult their healthcare provider"
        "\n- If uncertain, clearly state and suggest consulting provider"
    )
    
    # Start with system message using BaseMessage format
    llm_messages = [SystemMessage(content=system_instruction)]
    
    # Add patient context if available (limit length for token efficiency)
    if patient_info:
        patient_context = f"Patient Data: {str(patient_info)[:400]}"
        llm_messages.append(HumanMessage(content=patient_context))
        llm_messages.append(AIMessage(content="I have reviewed the patient data and I'm ready to assist."))
    
    # Check if we have tool results in the message history
    has_tool_results = any(isinstance(msg, ToolMessage) for msg in messages)
    
    if has_tool_results:
        print("[DEBUG] Processing tool results for synthesis")
        
        # We have tool results - include full conversation with proper message types
        for msg in messages:
            if isinstance(msg, (HumanMessage, AIMessage, ToolMessage)):
                llm_messages.append(msg)
        
        # Ensure last message is from user or tool (required for some models like Mistral)
        if llm_messages and isinstance(llm_messages[-1], AIMessage):
            # Check if the AIMessage has tool_calls (waiting for tool response)
            if hasattr(llm_messages[-1], 'tool_calls') and llm_messages[-1].tool_calls:
                # This is a tool-calling message, don't add another user message
                print("[DEBUG] Last message contains tool_calls - waiting for tool results")
                pass
            else:
                # Regular AI message - add synthesis request
                print("[DEBUG] Adding synthesis request")
                llm_messages.append(
                    HumanMessage(content="Based on the information above, please provide a comprehensive answer.")
                )
    
    else:
        # No tool results yet - bind tools so LLM can decide to call them
        print("[DEBUG] No tool results - binding tools to LLM")
        
        if clinical_agent_tools:
            try:
                llm = llm.bind_tools(clinical_agent_tools)
                print(f"[DEBUG] Bound {len(clinical_agent_tools)} tools to clinical LLM")
            except Exception as e:
                # Tools binding failed, continue without tools
                print(f"[WARN] Failed to bind tools: {e}")
                pass
        
        # Add conversation history using proper message types
        for msg in messages:
            if isinstance(msg, (HumanMessage, AIMessage, ToolMessage)):
                llm_messages.append(msg)
    
    # Final check: ensure last message is not AIMessage without tool_calls
    # (Required for Mistral and some other models)
    if llm_messages and isinstance(llm_messages[-1], AIMessage):
        last_ai_msg = llm_messages[-1]
        # If it's not a tool-calling message, add a user prompt
        if not (hasattr(last_ai_msg, 'tool_calls') and last_ai_msg.tool_calls):
            print("[DEBUG] Appending continuation prompt")
            llm_messages.append(
                HumanMessage(content="Please continue with your response.")
            )
    
    # ========================================================================
    # STREAMING IMPLEMENTATION
    # ========================================================================
    
    try:
        print("[DEBUG] Starting LLM stream...")
        
        # Accumulators for streaming chunks
        response_content = ""
        tool_calls_dict = {}  # Use dict to handle partial tool calls
        response_metadata = {}
        
        # Stream the LLM response token-by-token
        chunk_count = 0
        async for chunk in llm.astream(llm_messages):
            chunk_count += 1
            
            # ----------------------------------------------------------------
            # Process content chunks (text tokens)
            # ----------------------------------------------------------------
            if hasattr(chunk, 'content') and chunk.content:
                response_content += chunk.content
                # Optional: Log streaming progress
                # print(f"[STREAM] Token: {chunk.content}", end="", flush=True)
            
            # ----------------------------------------------------------------
            # Process tool call chunks
            # ----------------------------------------------------------------
            if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                for tool_call in chunk.tool_calls:
                    # Tool calls may come in multiple chunks - aggregate by index
                    tc_index = tool_call.get('index', 0)
                    
                    if tc_index not in tool_calls_dict:
                        # Initialize new tool call entry
                        tool_calls_dict[tc_index] = {
                            'name': tool_call.get('name', ''),
                            'args': tool_call.get('args', {}),
                            'id': tool_call.get('id', ''),
                            'type': tool_call.get('type', 'function')
                        }
                    else:
                        # Merge with existing tool call data
                        existing = tool_calls_dict[tc_index]
                        
                        # Update name if provided
                        if tool_call.get('name'):
                            existing['name'] = tool_call.get('name')
                        
                        # Update/merge args if provided
                        if tool_call.get('args'):
                            if isinstance(existing['args'], dict) and isinstance(tool_call['args'], dict):
                                existing['args'].update(tool_call['args'])
                            else:
                                existing['args'] = tool_call.get('args')
                        
                        # Update id if provided
                        if tool_call.get('id'):
                            existing['id'] = tool_call.get('id')
            
            # ----------------------------------------------------------------
            # Capture metadata from chunks
            # ----------------------------------------------------------------
            if hasattr(chunk, 'response_metadata') and chunk.response_metadata:
                response_metadata.update(chunk.response_metadata)
        
        print(f"[DEBUG] Stream completed - processed {chunk_count} chunks")
        
        # ----------------------------------------------------------------
        # Construct final AIMessage from accumulated chunks
        # ----------------------------------------------------------------
        
        # Convert tool_calls_dict to list format expected by AIMessage
        tool_calls_list = []
        if tool_calls_dict:
            tool_calls_list = [
                tool_calls_dict[idx] 
                for idx in sorted(tool_calls_dict.keys())
            ]
            print(f"[DEBUG] Aggregated {len(tool_calls_list)} tool calls")
        
        # Create the final response message
        # IMPORTANT: Don't pass tool_calls=None, either pass a list or omit it entirely
        if tool_calls_list:
            # Has tool calls
            response = AIMessage(
                content=response_content or "",
                tool_calls=tool_calls_list
            )
        else:
            # No tool calls - omit the parameter entirely
            response = AIMessage(
                content=response_content or ""
            )
        
        # Add metadata if available
        if response_metadata:
            response.response_metadata = response_metadata
        
        # Log final response summary
        if response.content:
            content_preview = response.content[:100] + "..." if len(response.content) > 100 else response.content
            print(f"[DEBUG] Response content: {content_preview}")
        
        if tool_calls_list:
            tool_names = [tc.get('name', 'unknown') for tc in tool_calls_list]
            print(f"[DEBUG] Tool calls: {tool_names}")
    
    except Exception as error:
        # ----------------------------------------------------------------
        # Error handling with graceful fallback
        # ----------------------------------------------------------------
        print(f"[ERROR] LLM streaming failed: {error}")
        import traceback
        traceback.print_exc()
        
        # Return helpful error message to user
        response = AIMessage(
            content=f"I encountered an issue processing your question. Please try rephrasing or contact support if this persists. (Error: {str(error)[:100]})"
        )
    
    return {"messages": [response]}


# ============================================================================
# Optional: Streaming helper for external use
# ============================================================================

async def stream_clinical_response(state, callback=None):
    """
    Alternative streaming function that yields chunks in real-time.
    
    This can be used if you want to stream responses directly to a UI
    rather than accumulating them in the node.
    
    Args:
        state: ChatState with messages and optional patient_info
        callback: Optional function to call with each chunk (token, tool_call)
        
    Yields:
        Streaming chunks as they arrive from the LLM
    """
    messages = state.get("messages", [])
    patient_info = state.get("patient_info")
    
    llm = clinical_llm()
    
    # Build messages (same as clinical_agent_node)
    system_instruction = (
        "You are an expert clinical assistant specializing in post-discharge patient support and nephrology. "
        "You understand the complex multidimensional needs of patients with kidney disease, including medication management, "
        "dietary restrictions, symptom monitoring, and emergency warning signs. "
        "\n\nTOOL USAGE RULES (MANDATORY - FOLLOW STRICTLY):"
        "\n1. PATIENT DATA RETRIEVAL:"
        "\n   - Use ALWAYS when patient mentions: medications, discharge info, restrictions, follow-up, warnings"
        "\n   - Keywords: 'my medication', 'my discharge', 'restrictions', 'follow-up', 'warning signs', 'remind me'"
        "\n   - ALWAYS retrieve patient data FIRST before answering patient-specific questions"
        "\n2. NEPHROLOGY KNOWLEDGE BASE (RAG):"
        "\n   - Use for: disease mechanisms, treatment options, medical education, symptom management, pathophysiology"
        "\n   - Keywords: 'what is', 'explain', 'how does', 'treatment', 'management', 'stages', 'mechanism'"
        "\n3. WEB SEARCH:"
        "\n   - Use for: latest research, recent trials, new treatments, current guidelines, updates"
        "\n   - Keywords: 'latest', 'new', 'recent', 'research', 'trial', 'breakthrough', 'update'"
        "\n\nDECISION LOGIC:"
        "\n- 'my/me/I' + medication/restriction/warning → Patient Data Tool"
        "\n- 'what is/explain/how' + medical term → RAG Tool"
        "\n- 'latest/new research/treatment' → Web Search Tool"
        "\n- Multi-part questions: invoke appropriate tool for each part"
        "\n\nRESPONSE GUIDELINES:"
        "\n- Begin with patient-specific context from their discharge data"
        "\n- Provide evidence-based information with specifics (drug names, doses, routes)"
        "\n- Contextualize general medical info to patient's condition"
        "\n- Use detailed responses in clear language, maximum 20 sentences"
        "\n- ALWAYS remind to consult their healthcare provider"
        "\n- If uncertain, clearly state and suggest consulting provider"
    )
    
    llm_messages = [SystemMessage(content=system_instruction)]
    
    if patient_info:
        patient_context = f"Patient Data: {str(patient_info)[:400]}"
        llm_messages.append(HumanMessage(content=patient_context))
        llm_messages.append(AIMessage(content="I have reviewed the patient data."))
    
    # Add conversation history
    for msg in messages:
        if isinstance(msg, (HumanMessage, AIMessage, ToolMessage)):
            llm_messages.append(msg)
    
    # Bind tools if no tool results yet
    has_tool_results = any(isinstance(msg, ToolMessage) for msg in messages)
    if not has_tool_results and clinical_agent_tools:
        try:
            llm = llm.bind_tools(clinical_agent_tools)
        except Exception:
            pass
    
    # Stream and yield chunks
    async for chunk in llm.astream(llm_messages):
        if callback:
            await callback(chunk)
        yield chunk