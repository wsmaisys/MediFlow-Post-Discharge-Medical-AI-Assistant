"""
chatbot_main.py - Reworked entry routing to avoid repeated receptionist runs

Changes are intentionally minimal and backwards-compatible:
- Replaces unconditional graph.add_edge(START, "receptionist")
  with graph.add_conditional_edges(START, route_from_start, {...})
  so the compiled graph's dispatcher consults persisted state and routes
  correctly for follow-up messages.
- Preserves node names, tool node behavior, checkpointer usage, and compile
  call signature.
"""

import asyncio
from dotenv import load_dotenv
from langgraph.graph import START, END
from langgraph.prebuilt import ToolNode

# Import components (kept exactly as before)
from llm_models import clinical_llm, receptionist_llm
from tools import clinical_agent_tools, receptionist_tools
from state_and_graph import graph, checkpointer
from agents_nodes import (
    receptionist_agent_node,
    clinical_agent_node,
    patient_data_retrieval_node
)
from routing import route_from_receptionist, route_from_lookup, route_from_start
from utilities import run_cli

load_dotenv()


def custom_tools_condition(state):
    """
    Custom routing logic to check if LLM wants to use tools.
    Examines the last message to see if it contains tool_calls.
    Returns "tools" if tools should be executed, "__end__" otherwise.
    """
    messages = state.get("messages", [])
    if not messages:
        return "__end__"
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and getattr(last_message, "tool_calls"):
        print(f"[DEBUG] Tool calls detected: {[tc.get('name', 'unknown') for tc in last_message.tool_calls]}")
        return "tools"
    print("[DEBUG] No tool calls - ending conversation")
    return "__end__"


async def initialize_chatbot():
    """
    Initialize and compile the complete chatbot graph with streaming support.
    """
    # Step 1: Create LLM instances
    llm_clinical = clinical_llm()
    llm_receptionist = receptionist_llm()

    # Step 2: Bind tools to LLMs (preserve original behavior)
    llm_clinical_with_tools = (
        llm_clinical.bind_tools(clinical_agent_tools)
        if clinical_agent_tools
        else llm_clinical
    )
    llm_receptionist_with_tools = (
        llm_receptionist.bind_tools(receptionist_tools)
        if receptionist_tools
        else llm_receptionist
    )

    print(f"[INFO] Clinical LLM tools: {'Bound' if clinical_agent_tools else 'None'}")
    print(f"[INFO] Receptionist tools: {'Bound' if receptionist_tools else 'None'}")

    # Step 3: Add nodes to the graph (names preserved)
    graph.add_node("receptionist", receptionist_agent_node)
    graph.add_node("patient_data_retrieval", patient_data_retrieval_node)
    graph.add_node("clinical_agent", clinical_agent_node)

    # Step 4: Define graph edges (transitions between nodes)
    # IMPORTANT: Use a START dispatcher that checks state before routing.
    graph.add_conditional_edges(
        START,
        route_from_start,
        {
            "receptionist": "receptionist",
            "patient_data_retrieval": "patient_data_retrieval",
            "clinical_agent": "clinical_agent",
        },
    )

    # Receptionist routing: detect if patient needs lookup or goes straight to clinical
    graph.add_conditional_edges(
        "receptionist",
        route_from_receptionist,
        {
            "patient_data_retrieval": "patient_data_retrieval",
            "clinical_agent": "clinical_agent",
        },
    )

    # After patient lookup, always go to clinical agent
    graph.add_conditional_edges(
        "patient_data_retrieval",
        route_from_lookup,
        {"clinical_agent": "clinical_agent"},
    )

    # Clinical agent routing: use tools if needed, or end conversation
    if clinical_agent_tools:
        tool_executor = ToolNode(clinical_agent_tools)
        graph.add_node("clinical_tools", tool_executor)
        graph.add_conditional_edges(
            "clinical_agent",
            custom_tools_condition,
            {"tools": "clinical_tools", "__end__": END},
        )
        # After tool execution, loop back to clinical agent to generate final response
        graph.add_edge("clinical_tools", "clinical_agent")
    else:
        graph.add_edge("clinical_agent", END)

    # Step 5: Compile the graph
    print("[INFO] Compiling chatbot graph.")
    compiled_graph = graph.compile(checkpointer=checkpointer)
    print("[INFO] Chatbot graph compiled successfully")
    print("[INFO] Recursion limit will be set at runtime in config")

    return compiled_graph


async def stream_chatbot_response(chatbot, user_input: str, thread_id: str = "default"):
    """
    Stream chatbot responses token-by-token for better UX.
    This function preserves the same external signature and streaming behavior.
    """
    from langchain_core.messages import HumanMessage

    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 100
    }

    # IMPORTANT: do not overwrite state persistence plan.
    # We pass only the new human message as in the original code.
    # The compiled graph is provided with checkpointer, which allows it to
    # restore persisted state for this thread. Therefore we keep input_state
    # as a single appended HumanMessage (the runtime will merge it).
    input_state = {"messages": [HumanMessage(content=user_input)]}

    print("\n[Assistant]: ", end="", flush=True)
    current_content = ""

    # Use astream_events for token-level streaming
    async for event in chatbot.astream_events(input_state, config, version="v2"):
        kind = event.get("event")

        # Stream AI message chunks (token-by-token)
        if kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, "content"):
                content = chunk.content
                if content:
                    print(content, end="", flush=True)
                    current_content += content

        # Handle tool calls (show which tools are being used)
        elif kind == "on_chat_model_end":
            output = event.get("data", {}).get("output")
            if output and hasattr(output, "tool_calls") and output.tool_calls:
                tool_names = [tc.get("name", "unknown") for tc in output.tool_calls]
                print(f"\n[Using tools: {', '.join(tool_names)}.]", flush=True)

    print("\n")  # New line after streaming completes


async def main():
    """
    Main entry point for the chatbot application.
    """
    try:
        print("[INFO] Initializing DataSmith AI Chatbot...")
        chatbot = await initialize_chatbot()
        print("[INFO] Starting CLI interface with streaming support...")
        print("[INFO] Type your messages and press Enter")
        print("[INFO] Type 'quit' or 'exit' to stop\n")
        await run_streaming_cli(chatbot)
    except KeyboardInterrupt:
        print("\n[INFO] Chatbot stopped by user")
    except Exception as error:
        print(f"[ERROR] Failed to start chatbot: {error}")
        import traceback
        traceback.print_exc()
        raise


async def run_streaming_cli(chatbot):
    """
    Run an interactive CLI with streaming support.
    """
    thread_id = "cli_session_001"

    while True:
        try:
            user_input = input("\n[You]: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "bye"]:
                print("[INFO] Goodbye!")
                break
            await stream_chatbot_response(chatbot, user_input, thread_id)
        except KeyboardInterrupt:
            print("\n[INFO] Exiting...")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()


# Entry point
if __name__ == "__main__":
    asyncio.run(main())