"""
diagnose_state_flow.py - Diagnostic script to identify state transition issues

This script helps debug the flow of messages and state between nodes.
Run this to see exactly what's happening at each step.
"""

import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()


async def diagnose_state_flow():
    """
    Test the state flow through receptionist → clinical_agent transition.
    """
    print("=" * 70)
    print("DIAGNOSTIC: State Flow Analysis")
    print("=" * 70)
    
    # Import your graph
    try:
        from chatbot_main import initialize_chatbot
        print("✓ Successfully imported chatbot")
    except Exception as e:
        print(f"✗ Failed to import: {e}")
        return
    
    # Initialize the chatbot
    try:
        chatbot = await initialize_chatbot()
        print("✓ Chatbot initialized")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 70)
    print("TEST 1: Simple greeting (should go directly to clinical agent)")
    print("=" * 70)
    
    config = {
        "configurable": {"thread_id": "diagnostic_test_1"},
        "recursion_limit": 100
    }
    
    input_state = {"messages": [HumanMessage(content="Hello, how are you?")]}
    
    print("\n[INPUT STATE]")
    print(f"  Messages: {len(input_state['messages'])}")
    print(f"  Content: {input_state['messages'][0].content}")
    
    # Track state through each node
    print("\n[STREAMING THROUGH NODES]")
    
    node_count = 0
    async for chunk in chatbot.astream(input_state, config):
        node_count += 1
        print(f"\n--- Node {node_count} Output ---")
        for node_name, node_state in chunk.items():
            print(f"Node: {node_name}")
            
            if "messages" in node_state:
                messages = node_state["messages"]
                print(f"  Total messages in state: {len(messages)}")
                
                for i, msg in enumerate(messages):
                    msg_type = type(msg).__name__
                    content = str(msg.content)[:100]
                    print(f"    [{i}] {msg_type}: {content}")
            
            if "patient_info" in node_state:
                print(f"  Patient info: {node_state.get('patient_info')}")
            
            if "next_node" in node_state:
                print(f"  Next node hint: {node_state.get('next_node')}")
    
    print("\n" + "=" * 70)
    print("TEST 2: Patient introduction (should go through lookup)")
    print("=" * 70)
    
    config2 = {
        "configurable": {"thread_id": "diagnostic_test_2"},
        "recursion_limit": 100
    }
    
    input_state2 = {"messages": [HumanMessage(content="Hi, I'm John Smith")]}
    
    print("\n[INPUT STATE]")
    print(f"  Messages: {len(input_state2['messages'])}")
    print(f"  Content: {input_state2['messages'][0].content}")
    
    print("\n[STREAMING THROUGH NODES]")
    
    node_count = 0
    async for chunk in chatbot.astream(input_state2, config2):
        node_count += 1
        print(f"\n--- Node {node_count} Output ---")
        for node_name, node_state in chunk.items():
            print(f"Node: {node_name}")
            
            if "messages" in node_state:
                messages = node_state["messages"]
                print(f"  Total messages in state: {len(messages)}")
                
                for i, msg in enumerate(messages):
                    msg_type = type(msg).__name__
                    content = str(msg.content)[:100]
                    print(f"    [{i}] {msg_type}: {content}")
            
            if "patient_info" in node_state:
                info = node_state.get('patient_info')
                info_preview = str(info)[:200] if info else None
                print(f"  Patient info: {info_preview}")
            
            if "next_node" in node_state:
                print(f"  Next node hint: {node_state.get('next_node')}")
    
    print("\n" + "=" * 70)
    print("TEST 3: Follow-up question in existing conversation")
    print("=" * 70)
    
    # Continue the conversation from test 2
    followup_state = {"messages": [HumanMessage(content="What medications should I take?")]}
    
    print("\n[INPUT STATE]")
    print(f"  Messages: {len(followup_state['messages'])}")
    print(f"  Content: {followup_state['messages'][0].content}")
    print(f"  Using same thread_id: diagnostic_test_2")
    
    print("\n[STREAMING THROUGH NODES]")
    
    node_count = 0
    async for chunk in chatbot.astream(followup_state, config2):
        node_count += 1
        print(f"\n--- Node {node_count} Output ---")
        for node_name, node_state in chunk.items():
            print(f"Node: {node_name}")
            
            if "messages" in node_state:
                messages = node_state["messages"]
                print(f"  Total messages in state: {len(messages)}")
                
                # Show last 3 messages
                recent_messages = messages[-3:] if len(messages) > 3 else messages
                for i, msg in enumerate(recent_messages):
                    actual_index = len(messages) - len(recent_messages) + i
                    msg_type = type(msg).__name__
                    content = str(msg.content)[:100]
                    print(f"    [{actual_index}] {msg_type}: {content}")
            
            if "patient_info" in node_state:
                info = node_state.get('patient_info')
                info_preview = str(info)[:200] if info else None
                print(f"  Patient info: {info_preview}")
    
    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)
    
    print("\n📊 What to look for:")
    print("  1. Does clinical_agent receive ALL previous messages?")
    print("  2. Is patient_info preserved across nodes?")
    print("  3. Are messages accumulating correctly?")
    print("  4. Does the routing work as expected?")
    
    print("\n⚠️  Common Issues:")
    print("  • If clinical_agent only sees 1 message: State reducer missing")
    print("  • If patient_info is None in clinical_agent: Not passed in return dict")
    print("  • If messages reset: Using wrong return format in nodes")
    print("  • If stuck in loop: Routing logic incorrect")


async def test_state_schema():
    """
    Test if the state schema has the correct reducer.
    """
    print("\n" + "=" * 70)
    print("STATE SCHEMA TEST")
    print("=" * 70)
    
    try:
        from state_and_graph import ChatState, graph
        import inspect
        
        print("\n✓ ChatState imported successfully")
        
        # Check if ChatState has proper annotations
        if hasattr(ChatState, '__annotations__'):
            print("\n📋 State Fields:")
            for field_name, field_type in ChatState.__annotations__.items():
                print(f"  • {field_name}: {field_type}")
                
                # Check if messages has a reducer
                if field_name == "messages":
                    if hasattr(field_type, '__metadata__'):
                        print(f"    ✓ Has reducer: {field_type.__metadata__}")
                    else:
                        print(f"    ⚠️  WARNING: No reducer found!")
                        print(f"       Messages will be REPLACED, not APPENDED")
                        print(f"       Fix: Use Annotated[list[BaseMessage], add_messages]")
        
        print("\n✓ State schema test complete")
        
    except Exception as e:
        print(f"\n✗ State schema test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🔍 DataSmith AI Chatbot Diagnostic Tool\n")
    
    async def run_all_diagnostics():
        await test_state_schema()
        await diagnose_state_flow()
    
    asyncio.run(run_all_diagnostics())