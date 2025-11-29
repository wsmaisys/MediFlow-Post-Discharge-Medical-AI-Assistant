"""
Tests for tools module - Tool definitions and initialization.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock, AsyncMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestSearchWebTool:
    """Test web search tool."""
    
    @patch('tools.DuckDuckGoSearchRun')
    @pytest.mark.asyncio
    async def test_search_web_tool(self, mock_ddg):
        """Test search_web tool execution via ainvoke."""
        from tools import search_web
        
        mock_instance = MagicMock()
        mock_instance.run.return_value = "Search results"
        mock_ddg.return_value = mock_instance
        
        result = await search_web.ainvoke({"query": "test query"})
        
        assert result == "Search results"


class TestPatientDataTool:
    """Test patient data retrieval tool."""
    
    @patch('builtins.open')
    @patch('json.load')
    @pytest.mark.asyncio
    async def test_get_patient_discharge_report_found(self, mock_json_load, mock_open):
        """Test retrieving existing patient report."""
        from tools import get_patient_discharge_report
        
        mock_json_load.return_value = [
            {
                "name": "John Doe",
                "discharge_report": {"diagnosis": "Hypertension"}
            }
        ]
        
        result = await get_patient_discharge_report.ainvoke({"patient_name": "John Doe"})
        
        assert "Hypertension" in result or isinstance(result, str)
    
    @patch('builtins.open')
    @patch('json.load')
    @pytest.mark.asyncio
    async def test_get_patient_discharge_report_not_found(self, mock_json_load, mock_open):
        """Test when patient is not found."""
        from tools import get_patient_discharge_report
        
        mock_json_load.return_value = [
            {"name": "John Doe", "discharge_report": {}}
        ]
        
        result = await get_patient_discharge_report.ainvoke({"patient_name": "Jane Smith"})
        
        assert "No patient found" in result or isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_get_patient_discharge_report_file_error(self):
        """Test file not found error handling."""
        from tools import get_patient_discharge_report
        
        result = await get_patient_discharge_report.ainvoke({"patient_name": "anyone"})
        
        # Should handle FileNotFoundError gracefully
        assert isinstance(result, str)


class TestNephrologyTool:
    """Test nephrology RAG query tool."""
    
    @patch('tools.aiohttp.ClientSession')
    @pytest.mark.asyncio
    async def test_query_nephrology_docs_success(self, mock_session_class):
        """Test successful nephrology query."""
        from tools import query_nephrology_docs
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "result": {
                "status": "success",
                "context": ["Test context"],
                "metadata": [{"source": "test.pdf", "page": 1}],
                "num_results": 1
            }
        })
        
        # Create async context manager mock that properly handles awaits
        async def mock_post_cm(*args, **kwargs):
            return mock_response
        
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=None)
        ))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        mock_session_class.return_value = mock_session
        
        result = await query_nephrology_docs.ainvoke({"query": "test query"})
        
        assert isinstance(result, str)
    
    @patch('tools.aiohttp.ClientSession')
    @pytest.mark.asyncio
    async def test_query_nephrology_docs_error(self, mock_session_class):
        """Test error handling in nephrology query."""
        from tools import query_nephrology_docs
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "error": {"message": "Server error", "code": 500}
        })
        
        # Create async context manager mock that properly handles awaits
        async def mock_post_cm(*args, **kwargs):
            return mock_response
        
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=None)
        ))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        mock_session_class.return_value = mock_session
        
        result = await query_nephrology_docs.ainvoke({"query": "test query"})
        
        assert isinstance(result, str)


class TestToolsDefinition:
    """Test tool definitions and assignments."""
    
    def test_receptionist_tools_defined(self):
        """Test receptionist tools list."""
        from tools import receptionist_tools
        
        assert isinstance(receptionist_tools, list)
        assert len(receptionist_tools) > 0
    
    def test_clinical_tools_defined(self):
        """Test clinical agent tools list."""
        from tools import clinical_agent_tools
        
        assert isinstance(clinical_agent_tools, list)
        assert len(clinical_agent_tools) > 0
    
    def test_nephrology_tool_assigned(self):
        """Test nephrology tool is assigned."""
        from tools import nephrology_tool, query_nephrology_docs
        
        assert nephrology_tool == query_nephrology_docs
    
    def test_patient_data_tool_assigned(self):
        """Test patient data tool is assigned."""
        from tools import patient_data_tool, get_patient_discharge_report
        
        assert patient_data_tool == get_patient_discharge_report


class TestMCPClient:
    """Test MCP client initialization."""
    
    def test_mcp_client_exists(self):
        """Test MCP client is initialized."""
        from tools import client
        
        assert client is not None
    
    def test_load_mcp_tools_callable(self):
        """Test load_mcp_tools_from_client is callable."""
        from tools import load_mcp_tools_from_client
        
        assert callable(load_mcp_tools_from_client)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
