"""
LLM model initialization and configuration.
Centralizes all LLM setup in one place.
"""

# Import ChatMistralAI for using Mistral's models
from langchain_mistralai import ChatMistralAI


# Function 1: Create receptionist LLM instance
def receptionist_llm():
    """
    Initialize the receptionist language model.
    Uses Mistral Small for fast, efficient responses.
    Lower temperature (0) for consistent, predictable greetings.
    """
    # Create and return ChatMistralAI instance
    return ChatMistralAI(
        # Use the small model (faster, cheaper, sufficient for greetings)
        model_name="mistral-small-latest",
        # Temperature 0 = deterministic, consistent responses
        temperature=0
    )


# Function 2: Create clinical LLM instance
def clinical_llm():
    """
    Initialize the clinical language model.
    Uses Mistral Small for consistent medical responses.
    Lower temperature (0) for factual, reliable medical information.
    """
    # Create and return ChatMistralAI instance
    return ChatMistralAI(
        # Use the small model (sufficient for medical Q&A with tools)
        model_name="mistral-small-latest",
        # Temperature 0 = deterministic, consistent medical answers
        temperature=0
    )

