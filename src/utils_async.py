"""
Async utilities for handling async operations in a sync context.
"""

import asyncio
import threading
import atexit
import warnings
from langsmith import traceable

# Dedicated async loop for backend tasks
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _cleanup_event_loop():
    """Cleanup event loop on exit."""
    try:
        if not _ASYNC_LOOP.is_closed():
            _ASYNC_LOOP.call_soon_threadsafe(_ASYNC_LOOP.stop)
            _ASYNC_THREAD.join(timeout=2)
            _ASYNC_LOOP.close()
    except Exception:
        pass


# Register cleanup function to run on exit
atexit.register(_cleanup_event_loop)


@traceable(name="_submit_async", tags=["Utility", "Async"], metadata={"purpose": "Execute Coroutine in Async Loop"})
def _submit_async(coro):
    """Submit a coroutine to the background async loop."""
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


@traceable(name="run_async", tags=["Utility", "Async"], metadata={"purpose": "Run Async Coroutine Blocking"})
def run_async(coro):
    """Run an async coroutine and block until it completes."""
    return _submit_async(coro).result()


@traceable(name="submit_async_task", tags=["Utility", "Async"], metadata={"purpose": "Schedule Async Task"})
def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)
