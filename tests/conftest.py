"""
Pytest configuration file for test suite.
"""

import sys
import pytest

# Import python_multipart at the earliest possible time to prevent Starlette warning
try:
    import python_multipart  # noqa: F401
except ImportError:
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
