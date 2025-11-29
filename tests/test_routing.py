"""
Tests for routing module - Conditional routing logic.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestRouting:
    """Test routing logic."""
    
    def test_route_receptionist_to_lookup(self):
        """Test routing to patient data retrieval."""
        from routing import route_receptionist
        from state_and_graph import ChatState
        from langchain_core.messages import HumanMessage
        
        state = ChatState(
            messages=[HumanMessage(content="test")],
            patient_info=None,
            next_node="lookup_patient"
        )
        
        route = route_receptionist(state)
        assert route == "patient_data_retrieval"
    
    def test_route_receptionist_to_clinical(self):
        """Test routing to clinical agent."""
        from routing import route_receptionist
        from state_and_graph import ChatState
        from langchain_core.messages import HumanMessage
        
        state = ChatState(
            messages=[HumanMessage(content="test")],
            patient_info=None,
            next_node="ask_assistance"
        )
        
        route = route_receptionist(state)
        assert route == "clinical_agent"
    
    def test_route_receptionist_default(self):
        """Test default routing."""
        from routing import route_receptionist
        from state_and_graph import ChatState
        from langchain_core.messages import HumanMessage
        
        state = ChatState(
            messages=[HumanMessage(content="test")],
            patient_info=None,
            next_node="unknown"
        )
        
        route = route_receptionist(state)
        assert route == "clinical_agent"
    
    def test_route_receptionist_with_none_next(self):
        """Test routing when next is None."""
        from routing import route_receptionist
        from state_and_graph import ChatState
        from langchain_core.messages import HumanMessage
        
        state = ChatState(
            messages=[HumanMessage(content="test")],
            patient_info=None,
            next_node=None
        )
        
        route = route_receptionist(state)
        assert route == "clinical_agent"


class TestRoutingEdgeCases:
    """Test edge cases in routing."""
    
    def test_route_with_empty_state(self):
        """Test routing with minimal state."""
        from routing import route_receptionist
        from state_and_graph import ChatState
        from langchain_core.messages import HumanMessage
        
        state = ChatState(
            messages=[HumanMessage(content="")],
            patient_info=None,
            next_node=""
        )
        
        route = route_receptionist(state)
        assert isinstance(route, str)
        assert route in ["patient_data_retrieval", "clinical_agent"]
    
    def test_multiple_routing_decisions(self):
        """Test multiple routing decisions are consistent."""
        from routing import route_receptionist
        from state_and_graph import ChatState
        from langchain_core.messages import HumanMessage
        
        state = ChatState(
            messages=[HumanMessage(content="test")],
            patient_info="Sample data",
            next="lookup_patient"
        )
        
        route1 = route_receptionist(state)
        route2 = route_receptionist(state)
        
        assert route1 == route2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
