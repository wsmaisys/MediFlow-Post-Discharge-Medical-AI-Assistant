"""
Fixed utilities.py - Enhanced CLI interface with better error handling.

Minor improvements:
✓ Better formatted output
✓ Clearer error messages
✓ More robust response handling
"""

from langchain_core.messages import HumanMessage
from utils_async import run_async


def run_cli(chatbot):
    """
    Run the command-line interface for interactive chat.
    
    Handles user input loop with proper error handling and formatting.
    Type 'exit' or 'quit' to stop.
    
    Args:
        chatbot: Compiled LangGraph application
    """
    print("\n" + "="*70)
    print("DataSmith AI Chatbot - Post-Discharge Support")
    print("Type 'exit' or 'quit' to stop the conversation")
    print("="*70 + "\n")
    
    try:
        while True:
            # Get user input
            user_input = input("You: ").strip()
            
            # Check for exit command
            if user_input.lower() in ("exit", "quit"):
                print("Thank you for using DataSmith. Goodbye!")
                break
            
            # Skip empty input
            if not user_input:
                continue
            
            # Prepare message for the graph
            messages = [HumanMessage(content=user_input)]
            
            try:
                # Invoke the chatbot with thread configuration for state persistence
                thread_id = "default_thread"
                config = {"configurable": {"thread_id": thread_id}}
                
                # Call the graph (blocking call via async wrapper)
                response = run_async(chatbot.ainvoke({"messages": messages}, config))
                
                # Extract response data
                final_messages = response.get("messages", [])
                patient_info = response.get("patient_info")
                
                # Display the response
                if final_messages:
                    last_message = final_messages[-1]
                    
                    # Extract content from message object
                    content = (
                        getattr(last_message, "content", None)
                        or str(last_message)
                    )
                    
                    # Display with patient context if available
                    if patient_info:
                        print(f"\nAssistant: {content}")
                        print(f"[Context: Patient records loaded]\n")
                    else:
                        print(f"\nAssistant: {content}\n")
                else:
                    print("\nAssistant: No response generated.\n")
            
            except Exception as error:
                print(f"\n[ERROR] Failed to process message: {error}")
                print("Please try again.\n")
    
    except KeyboardInterrupt:
        print("\n[INFO] Chat interrupted by user")
    
    except EOFError:
        print("\n[INFO] Input stream closed")
