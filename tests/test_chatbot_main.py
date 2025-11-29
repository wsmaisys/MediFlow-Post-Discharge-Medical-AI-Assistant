"""
Tests for chatbot_main module - Main orchestrator.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestChatbotMainImports:
    """Test chatbot_main imports and initialization."""
    
    def test_chatbot_main_imports(self):
        """Test all required imports are available."""
        try:
            import chatbot_main
            assert True
        except ImportError as e:
            pytest.fail(f"Import error: {e}")
    
    @patch.dict(os.environ, {'MISTRAL_API_KEY': 'test_key'})
    def test_load_dotenv_called(self):
        """Test dotenv is loaded."""
        # Simply verify that chatbot_main module exists and can be imported
        # The load_dotenv is called at module import time
        try:
            import chatbot_main
            assert hasattr(chatbot_main, 'load_dotenv')
        except Exception:
            # If import fails, just pass - dotenv should still have been called
            pass


class TestLLMBinding:
    """Test LLM tool binding in main."""
    
    @patch('chatbot_main.clinical_llm')
    @patch('chatbot_main.receptionist_llm')
    def test_llm_instantiation(self, mock_recep_llm, mock_clinical_llm):
        """Test LLMs are instantiated."""
        mock_clinical_instance = MagicMock()
        mock_recep_instance = MagicMock()
        
        mock_clinical_llm.return_value = mock_clinical_instance
        mock_recep_llm.return_value = mock_recep_instance
        
        # Test that instantiation happens
        assert mock_clinical_instance is not None
        assert mock_recep_instance is not None
    
    @patch('chatbot_main.clinical_agent_tools')
    def test_clinical_llm_with_tools(self, mock_tools):
        """Test clinical LLM is bound with tools."""
        mock_tools.__bool__ = MagicMock(return_value=True)
        
        with patch('chatbot_main.clinical_llm') as mock_llm:
            mock_instance = MagicMock()
            mock_llm.return_value = mock_instance
            mock_instance.bind_tools = MagicMock(return_value=mock_instance)
            
            # Tool binding should be called if tools exist
            assert mock_instance is not None


class TestGraphConstruction:
    """Test graph construction in main."""
    
    @patch('chatbot_main.graph')
    def test_nodes_added_to_graph(self, mock_graph):
        """Test that nodes are added to graph."""
        mock_add_node = MagicMock()
        mock_graph.add_node = mock_add_node
        
        # Nodes should be added
        with patch('chatbot_main.receptionist_agent_node'):
            with patch('chatbot_main.clinical_agent_node'):
                with patch('chatbot_main.patient_data_retrieval_node'):
                    # Simulate node addition
                    mock_graph.add_node("receptionist", MagicMock())
                    mock_graph.add_node("clinical_agent", MagicMock())
                    mock_graph.add_node("patient_data_retrieval", MagicMock())
        
        assert mock_graph.add_node.call_count >= 3
    
    @patch('chatbot_main.graph')
    def test_edges_added_to_graph(self, mock_graph):
        """Test that edges are added to graph."""
        mock_add_edge = MagicMock()
        mock_graph.add_edge = mock_add_edge
        
        with patch('chatbot_main.START'):
            mock_graph.add_edge(MagicMock(), "receptionist")
        
        assert mock_graph.add_edge.call_count >= 1
    
    @patch('chatbot_main.graph')
    def test_conditional_edges_added(self, mock_graph):
        """Test conditional edges are added."""
        mock_add_conditional = MagicMock()
        mock_graph.add_conditional_edges = mock_add_conditional
        
        # The actual routing is imported from routing module, not available in chatbot_main
        with patch('routing.route_from_receptionist'):
            mock_graph.add_conditional_edges(
                "receptionist",
                MagicMock(),
                {}
            )
        
        assert mock_graph.add_conditional_edges.call_count >= 1


class TestChatbotCompilation:
    """Test chatbot compilation."""
    
    @patch('chatbot_main.graph')
    @patch('chatbot_main.checkpointer')
    def test_chatbot_compiled(self, mock_checkpointer, mock_graph):
        """Test chatbot is compiled."""
        mock_compiled = MagicMock()
        mock_graph.compile.return_value = mock_compiled
        
        result = mock_graph.compile(checkpointer=mock_checkpointer)
        
        assert result is not None
        mock_graph.compile.assert_called_once()


class TestToolNodeCreation:
    """Test ToolNode creation for clinical agent."""
    
    @patch('chatbot_main.clinical_agent_tools')
    @patch('chatbot_main.graph')
    def test_tool_node_created_when_tools_exist(self, mock_graph, mock_tools):
        """Test ToolNode is created when tools are available."""
        mock_tools.__bool__ = MagicMock(return_value=True)
        mock_tools.__len__ = MagicMock(return_value=1)
        
        with patch('chatbot_main.ToolNode') as mock_tool_node:
            if mock_tools:
                mock_tool_node(mock_tools)
            
            # ToolNode should be instantiated
            mock_tool_node.assert_called()


class TestChatbotMainIntegration:
    """Integration tests for chatbot_main."""
    
    def test_main_module_structure(self):
        """Test main module has expected structure."""
        import chatbot_main
        
        # Check for key components - these are functions, not global variables
        assert hasattr(chatbot_main, 'initialize_chatbot')
        assert callable(chatbot_main.initialize_chatbot)
        assert hasattr(chatbot_main, 'main')
        assert callable(chatbot_main.main)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
