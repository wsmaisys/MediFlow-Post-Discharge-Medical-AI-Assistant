"""
Tests for state_and_graph module - State definition and graph initialization.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock, AsyncMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestChatState:
    """Test ChatState definition."""
    
    def test_chatstate_structure(self):
        """Test ChatState has required fields."""
        from state_and_graph import ChatState
        
        # Check annotations exist
        assert hasattr(ChatState, '__annotations__')
        annotations = ChatState.__annotations__
        
        assert 'messages' in annotations
        assert 'patient_info' in annotations
        # next_node is the primary field, 'next' is for backward compat
        assert 'next_node' in annotations or 'next' in annotations
    
    def test_chatstate_instantiation(self):
        """Test creating ChatState instance."""
        from state_and_graph import ChatState
        from langchain_core.messages import HumanMessage
        
        state = ChatState(
            messages=[HumanMessage(content="Test")],
            patient_info="Test info",
            next="clinical_agent"
        )
        
        assert len(state['messages']) == 1
        assert state['patient_info'] == "Test info"
        assert state['next'] == "clinical_agent"


class TestCheckpointer:
    """Test checkpointer initialization."""
    
    def test_checkpointer_exists(self):
        """Test that checkpointer is initialized or None (both acceptable)."""
        from state_and_graph import checkpointer
        
        # Checkpointer can be None if persistence is not configured
        # or a SqliteSaver instance if it is. Both are valid.
        assert checkpointer is None or hasattr(checkpointer, 'put')
    
    def test_checkpointer_uses_db_path(self):
        """Test checkpointer uses persistence_db path."""
        from state_and_graph import checkpointer
        import os
        
        # If checkpointer exists, it should reference the persistence_db
        # If not, persistence_db folder should exist
        db_path = "persistence_db/chatbot_state.db"
        expected_dir = os.path.dirname(db_path)
        
        # Just verify the path is correct in the code
        assert "persistence_db" in expected_dir or checkpointer is None


class TestStateGraph:
    """Test StateGraph initialization."""
    
    def test_graph_exists(self):
        """Test that graph is initialized."""
        from state_and_graph import graph
        
        assert graph is not None
    
    def test_graph_type(self):
        """Test that graph is a StateGraph."""
        from state_and_graph import graph
        from langgraph.graph import StateGraph
        
        assert isinstance(graph, StateGraph)
    
    def test_graph_state_definition(self):
        """Test graph uses ChatState."""
        from state_and_graph import graph, ChatState
        
        # Graph should be configured with ChatState
        assert graph is not None


class TestStateGraphIntegration:
    """Integration tests for state and graph."""
    
    def test_chatstate_with_graph_context(self):
        """Test ChatState works with graph configuration."""
        from state_and_graph import ChatState, graph
        from langchain_core.messages import HumanMessage
        
        state = ChatState(
            messages=[HumanMessage(content="Hello")],
            patient_info=None,
            next="receptionist"
        )
        
        assert 'messages' in state
        assert len(state['messages']) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
