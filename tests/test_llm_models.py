"""
Tests for llm_models module - LLM initialization.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestLLMModels:
    """Test LLM model initialization."""
    
    @patch('llm_models.ChatMistralAI')
    def test_receptionist_llm_initialization(self, mock_mistral):
        """Test receptionist LLM is initialized correctly."""
        from llm_models import receptionist_llm
        
        mock_instance = MagicMock()
        mock_mistral.return_value = mock_instance
        
        llm = receptionist_llm()
        
        # Verify ChatMistralAI was called with correct parameters
        mock_mistral.assert_called_once_with(
            model_name="mistral-small-latest",
            temperature=0
        )
        assert llm == mock_instance
    
    @patch('llm_models.ChatMistralAI')
    def test_clinical_llm_initialization(self, mock_mistral):
        """Test clinical LLM is initialized correctly."""
        from llm_models import clinical_llm
        
        mock_instance = MagicMock()
        mock_mistral.return_value = mock_instance
        
        llm = clinical_llm()
        
        # Verify ChatMistralAI was called with correct parameters
        mock_mistral.assert_called_once_with(
            model_name="mistral-small-latest",
            temperature=0
        )
        assert llm == mock_instance
    
    @patch('llm_models.ChatMistralAI')
    def test_llm_temperature_setting(self, mock_mistral):
        """Test that temperature is set to 0 for deterministic responses."""
        from llm_models import receptionist_llm, clinical_llm
        
        mock_instance = MagicMock()
        mock_mistral.return_value = mock_instance
        
        receptionist_llm()
        clinical_llm()
        
        # Both should have temperature=0
        for call in mock_mistral.call_args_list:
            assert call[1]['temperature'] == 0
    
    @patch('llm_models.ChatMistralAI')
    def test_llm_model_name(self, mock_mistral):
        """Test that correct model name is used."""
        from llm_models import receptionist_llm, clinical_llm
        
        mock_instance = MagicMock()
        mock_mistral.return_value = mock_instance
        
        receptionist_llm()
        clinical_llm()
        
        # Both should use mistral-small-latest
        for call in mock_mistral.call_args_list:
            assert call[1]['model_name'] == "mistral-small-latest"


class TestLLMIntegration:
    """Integration tests for LLM models."""
    
    @patch('llm_models.ChatMistralAI')
    def test_multiple_llm_instances(self, mock_mistral):
        """Test creating multiple LLM instances."""
        from llm_models import receptionist_llm, clinical_llm
        
        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_mistral.side_effect = [mock_instance1, mock_instance2]
        
        llm1 = receptionist_llm()
        llm2 = clinical_llm()
        
        assert llm1 != llm2
        assert mock_mistral.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
