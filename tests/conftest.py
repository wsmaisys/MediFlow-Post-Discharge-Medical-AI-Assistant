"""
Pytest configuration file for test suite.
"""

import os
import sys
import pytest

os.environ["LANGSMITH_TRACING"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = ""
os.environ["LANGCHAIN_TRACING"] = ""
os.environ["LANGSMITH_API_KEY"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""

# Import python_multipart at the earliest possible time to prevent Starlette warning
try:
    import python_multipart  # noqa: F401
except ImportError:
    pass

try:
    from langsmith import client as langsmith_client
    from types import SimpleNamespace

    def _noop_langsmith(*args, **kwargs):
        return None

    langsmith_client.Client.info = property(lambda self: SimpleNamespace(instance_flags={}))
    langsmith_client.Client.batch_ingest_runs = _noop_langsmith
    langsmith_client.Client.multipart_ingest = _noop_langsmith
    langsmith_client.Client._post_batch_ingest_runs = _noop_langsmith
except Exception:
    pass


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Suppress known external library warnings
    warnings_filter = [
        # Suppress ResourceWarnings from anyio (external library stream management)
        ("ignore", ResourceWarning, "anyio.streams.memory", None),
        # Suppress PydanticDeprecatedSince20 from langchain_core (external dependency)
        ("ignore", DeprecationWarning, "langchain_core.tools.base", None),
    ]
    
    for category, warn_type, module, lineno in warnings_filter:
        # These are handled in pytest.ini filterwarnings
        pass
