"""
Tests for utils_async module - async/sync bridge utilities.
"""

import pytest
import asyncio
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils_async import run_async, submit_async_task, _submit_async


class TestAsyncUtilities:
    """Test async/sync bridge utilities."""
    
    def test_run_async_simple_coroutine(self):
        """Test run_async with a simple async function."""
        async def simple_coro():
            return 42
        
        result = run_async(simple_coro())
        assert result == 42
    
    def test_run_async_with_await(self):
        """Test run_async with async operations."""
        async def async_sleep():
            await asyncio.sleep(0.01)
            return "done"
        
        result = run_async(async_sleep())
        assert result == "done"
    
    def test_run_async_exception_handling(self):
        """Test run_async handles exceptions properly."""
        async def failing_coro():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            run_async(failing_coro())
    
    def test_submit_async_task(self):
        """Test submit_async_task schedules a task."""
        async def test_task():
            await asyncio.sleep(0.01)
            return "task_result"
        
        future = submit_async_task(test_task())
        assert future is not None
        # Future should complete eventually
        result = future.result(timeout=5)
        assert result == "task_result"
    
    def test_submit_async_returns_future(self):
        """Test _submit_async returns a Future."""
        async def dummy():
            return None
        
        future = _submit_async(dummy())
        assert hasattr(future, 'result')
        assert callable(future.result)


class TestAsyncIntegration:
    """Integration tests for async utilities."""
    
    def test_multiple_concurrent_operations(self):
        """Test running multiple async operations."""
        async def operation(value):
            await asyncio.sleep(0.01)
            return value * 2
        
        # Run sequentially through run_async
        result1 = run_async(operation(5))
        result2 = run_async(operation(10))
        
        assert result1 == 10
        assert result2 == 20
    
    def test_async_with_complex_logic(self):
        """Test async with complex operations."""
        async def complex_operation():
            results = []
            for i in range(3):
                await asyncio.sleep(0.01)
                results.append(i)
            return sum(results)
        
        result = run_async(complex_operation())
        assert result == 3  # 0 + 1 + 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
