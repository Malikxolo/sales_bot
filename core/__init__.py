"""
Brain-Heart Deep Research System - Core Module
Simple version that definitely works with Python 3.12.3
"""

__version__ = "2.0.0"

# Import exceptions first - these should always work
from .exceptions import (
    BrainHeartException,
    LLMClientError,
    BrainAgentError,
    HeartAgentError,
    ToolExecutionError,
    ConfigurationError,
    APIKeyError,
    ModelNotAvailableError
)

# Import config - minimal dependencies
from .config import Config

# Import other components - may fail if dependencies missing
try:
    from .llm_client import LLMClient
except ImportError:
    LLMClient = None

try:
    from .tools import ToolManager
except ImportError:
    ToolManager = None

try:
    from .brain_agent import BrainAgent
except ImportError:
    BrainAgent = None

try:
    from .heart_agent import HeartAgent
except ImportError:
    HeartAgent = None

try:
    from .optimized_agent import OptimizedAgent
except ImportError:
    OptimizedAgent = None

# Export everything that was imported successfully
__all__ = [
    'BrainHeartException',
    'LLMClientError', 
    'BrainAgentError',
    'HeartAgentError',
    'ToolExecutionError',
    'ConfigurationError',
    'APIKeyError',
    'ModelNotAvailableError',
    'Config'
]

# Add optional components if they loaded
if LLMClient is not None:
    __all__.append('LLMClient')
if ToolManager is not None:
    __all__.append('ToolManager')
if BrainAgent is not None:
    __all__.append('BrainAgent')
if HeartAgent is not None:
    __all__.append('HeartAgent')
