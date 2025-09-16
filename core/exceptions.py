"""
Custom exceptions for Brain-Heart Deep Research System
Simplified version to avoid import issues
"""

class BrainHeartException(Exception):
    """Base exception for Brain-Heart system"""
    pass

class LLMClientError(BrainHeartException):
    """LLM client related errors"""
    pass

class BrainAgentError(BrainHeartException):
    """Brain agent specific errors"""
    pass

class HeartAgentError(BrainHeartException):
    """Heart agent specific errors"""
    pass

class ToolExecutionError(BrainHeartException):
    """Tool execution errors"""
    pass

class ConfigurationError(BrainHeartException):
    """Configuration related errors"""
    pass

class APIKeyError(ConfigurationError):
    """API key related errors"""
    pass

class ModelNotAvailableError(LLMClientError):
    """Model not available error"""
    pass