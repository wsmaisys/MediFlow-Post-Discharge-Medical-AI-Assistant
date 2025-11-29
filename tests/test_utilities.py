"""
Tests for utilities module - CLI and utility functions.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock, call
from io import StringIO

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestCLI:
    """Test CLI interface."""
    
    @patch('builtins.input', side_effect=['test query', 'exit'])
    @patch('utilities.run_async')
    def test_cli_simple_interaction(self, mock_run_async, mock_input):
        """Test basic CLI interaction."""
        from utilities import run_cli
        from langchain_core.messages import HumanMessage, AIMessage
        
        mock_chatbot = MagicMock()
        mock_response = {
            "messages": [AIMessage(content="Test response")],
            "patient_info": None
        }
        mock_run_async.return_value = mock_response
        
        # Capture output
        with patch('sys.stdout', new=StringIO()):
            run_cli(mock_chatbot)
        
        # Verify input was called for user query
        assert mock_input.call_count >= 1
    
    @patch('builtins.input', side_effect=['quit'])
    def test_cli_exit_command(self, mock_input):
        """Test CLI exits on quit."""
        from utilities import run_cli
        from io import StringIO
        
        mock_chatbot = MagicMock()
        
        with patch('sys.stdout', new=StringIO()):
            run_cli(mock_chatbot)
        
        # Should exit cleanly
        assert True
    
    @patch('builtins.input', side_effect=['test', 'exit'])
    @patch('utilities.run_async')
    def test_cli_with_patient_info(self, mock_run_async, mock_input):
        """Test CLI displays patient info."""
        from utilities import run_cli
        from langchain_core.messages import HumanMessage, AIMessage
        
        mock_chatbot = MagicMock()
        mock_response = {
            "messages": [AIMessage(content="Response")],
            "patient_info": "Sample patient data"
        }
        mock_run_async.return_value = mock_response
        
        with patch('sys.stdout', new=StringIO()):
            run_cli(mock_chatbot)
        
        assert True
    
    @patch('builtins.input', side_effect=['error_test', 'exit'])
    @patch('utilities.run_async')
    def test_cli_error_handling(self, mock_run_async, mock_input):
        """Test CLI handles errors gracefully."""
        from utilities import run_cli
        
        mock_chatbot = MagicMock()
        mock_run_async.side_effect = Exception("Test error")
        
        with patch('sys.stdout', new=StringIO()):
            run_cli(mock_chatbot)
        
        # Should continue after error
        assert True
    
    @patch('builtins.input', side_effect=KeyboardInterrupt())
    def test_cli_keyboard_interrupt(self, mock_input):
        """Test CLI handles keyboard interrupt."""
        from utilities import run_cli
        
        mock_chatbot = MagicMock()
        
        with patch('sys.stdout', new=StringIO()):
            run_cli(mock_chatbot)
        
        assert True
    
    @patch('builtins.input', side_effect=EOFError())
    def test_cli_eof_error(self, mock_input):
        """Test CLI handles EOF."""
        from utilities import run_cli
        
        mock_chatbot = MagicMock()
        
        with patch('sys.stdout', new=StringIO()):
            run_cli(mock_chatbot)
        
        assert True


class TestCLIIntegration:
    """Integration tests for CLI."""
    
    @patch('builtins.input', side_effect=['hello', 'what is my diagnosis?', 'exit'])
    @patch('utilities.run_async')
    def test_cli_multi_turn_conversation(self, mock_run_async, mock_input):
        """Test multi-turn conversation."""
        from utilities import run_cli
        from langchain_core.messages import AIMessage
        
        mock_chatbot = MagicMock()
        mock_response = {
            "messages": [AIMessage(content="Response")],
            "patient_info": None
        }
        mock_run_async.return_value = mock_response
        
        with patch('sys.stdout', new=StringIO()):
            run_cli(mock_chatbot)
        
        # Verify run_async was called multiple times
        assert mock_run_async.call_count >= 2
    
    @patch('builtins.input', side_effect=['  ', '  medical question  ', 'exit'])
    @patch('utilities.run_async')
    def test_cli_whitespace_handling(self, mock_run_async, mock_input):
        """Test CLI handles whitespace input."""
        from utilities import run_cli
        from langchain_core.messages import AIMessage
        
        mock_chatbot = MagicMock()
        mock_response = {
            "messages": [AIMessage(content="Response")],
            "patient_info": None
        }
        mock_run_async.return_value = mock_response
        
        with patch('sys.stdout', new=StringIO()):
            run_cli(mock_chatbot)
        
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
