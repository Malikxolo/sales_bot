"""
🔐 Secure Logging Module for Brain-Heart Deep Research System

This module provides environment-aware logging functions that prevent sensitive
data leakage (PII, user conversations, business secrets) in application logs.

Key Features:
- Environment detection (production/development/staging)
- Automatic content truncation in development mode
- Metadata-only logging in production mode
- PII redaction (emails, phone numbers, API keys)
- User ID hashing for privacy

Usage:
    from core.logging_security import safe_log_response, safe_log_user_data, safe_log_error
    
    # Log AI response safely
    safe_log_response(result, level='info')
    
    # Log user activity safely
    safe_log_user_data(user_id, 'chat_request')
    
    # Log errors safely
    safe_log_error(exception, context={'endpoint': 'chat'})

Environment Variables:
    ENVIRONMENT: Set to 'production', 'development', or 'staging' (default: 'production')

Security Considerations:
    - Production logs are assumed to be accessible by unauthorized parties
    - Never log full user queries, AI responses, or PII in production
    - Always use metadata (length, time, counts) instead of content
"""

import os
import logging
import hashlib
import re
from typing import Dict, Any, Optional
from datetime import datetime

# Configure logger
logger = logging.getLogger(__name__)


def get_environment() -> str:
    """
    Detect the current runtime environment.
    
    Returns:
        str: One of 'production', 'development', or 'staging'
        
    Default:
        If ENVIRONMENT is not set, defaults to 'production' for security
    
    Examples:
        >>> os.environ['ENVIRONMENT'] = 'development'
        >>> get_environment()
        'development'
        
        >>> os.environ.pop('ENVIRONMENT', None)
        >>> get_environment()
        'production'
    """
    env = os.getenv('ENVIRONMENT', 'production').lower()
    
    # Validate environment value
    valid_environments = ['production', 'development', 'staging']
    if env not in valid_environments:
        logger.warning(f"Invalid ENVIRONMENT value: {env}. Defaulting to 'production'")
        return 'production'
    
    return env


def _hash_identifier(identifier: str, prefix: str = "") -> str:
    """
    Create a consistent hash of an identifier for privacy-preserving logging.
    
    Args:
        identifier: The identifier to hash (e.g., user_id, email)
        prefix: Optional prefix for the hash (e.g., 'user', 'session')
    
    Returns:
        str: Format "prefix_HASH" where HASH is first 8 chars of SHA-256
        
    Examples:
        >>> _hash_identifier('user_12345', 'user')
        'user_a3f7c8d2'
    """
    hash_value = hashlib.sha256(identifier.encode('utf-8')).hexdigest()[:8]
    return f"{prefix}_{hash_value}" if prefix else hash_value


def _truncate_content(content: str, max_length: int = 200) -> str:
    """
    Truncate content to a maximum length for development logging.
    
    Args:
        content: The content to truncate
        max_length: Maximum length (default: 200 characters)
    
    Returns:
        str: Truncated content with ellipsis if truncated
        
    Examples:
        >>> _truncate_content("This is a long response...", max_length=10)
        'This is a ...'
    """
    if not content:
        return ""
    
    content_str = str(content)
    if len(content_str) <= max_length:
        return content_str
    
    return content_str[:max_length] + "..."


def _redact_pii(text: str) -> str:
    """
    Redact PII (Personally Identifiable Information) from text.
    
    Redacts:
        - Email addresses → [EMAIL]
        - Phone numbers → [PHONE]
        - API keys (pattern: starts with 'sk-', 'pk-', etc.) → [API_KEY]
        - Credit card numbers → [CREDIT_CARD]
        - Social Security Numbers → [SSN]
    
    Args:
        text: The text to redact
    
    Returns:
        str: Text with PII redacted
        
    Examples:
        >>> _redact_pii("Contact me at john@example.com or 555-1234")
        'Contact me at [EMAIL] or [PHONE]'
    """
    if not text:
        return text
    
    # Email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    
    # API keys (common patterns) - Check BEFORE phone numbers to avoid conflicts
    text = re.sub(r'\b(sk|pk|api|token)[-_][a-zA-Z0-9]{20,}\b', '[API_KEY]', text, flags=re.IGNORECASE)
    
    # Credit card numbers (16 digits with optional separators) - Check BEFORE phone numbers
    text = re.sub(r'\b\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}\b', '[CREDIT_CARD]', text)
    text = re.sub(r'\b\d{16}\b', '[CREDIT_CARD]', text)
    
    # Social Security Numbers (US format: XXX-XX-XXXX) - Check BEFORE phone numbers
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
    
    # Phone numbers (various formats) - Check AFTER credit cards and SSN
    text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]', text)
    text = re.sub(r'\b\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b', '[PHONE]', text)
    
    return text


def safe_log_response(
    response_data: Dict[str, Any],
    level: str = 'info',
    logger_instance: Optional[logging.Logger] = None
) -> None:
    """
    Safely log AI response data with environment-aware behavior.
    
    Production Mode:
        Logs only metadata (length, time, status)
        Example: "Response generated successfully (length: 1245 chars, time: 3.2s)"
    
    Development Mode:
        Logs truncated content (first 200 chars) + metadata
        Example: "Response preview: Once upon a time in... (length: 1245 chars)"
    
    Args:
        response_data: Dict containing response info. Expected keys:
            - 'response': The actual response content (str)
            - 'total_time': Time taken to generate response (float, optional)
            - 'success': Whether the operation succeeded (bool, optional)
        level: Logging level ('info', 'debug', 'warning', 'error')
        logger_instance: Optional logger instance (defaults to module logger)
    
    Examples:
        >>> result = {'response': 'Hello world!', 'total_time': 2.5, 'success': True}
        >>> safe_log_response(result)
        # Production: "Response generated successfully (length: 12 chars, time: 2.5s)"
        # Development: "Response preview: Hello world! (length: 12 chars, time: 2.5s)"
    """
    log = logger_instance or logger
    log_func = getattr(log, level.lower(), log.info)
    
    env = get_environment()
    
    # Extract response content
    response_content = response_data.get('response', '')
    total_time = response_data.get('total_time', 0)
    success = response_data.get('success', True)
    
    # Calculate metadata
    response_length = len(str(response_content))
    status = "successfully" if success else "with errors"
    
    if env == 'production':
        # Production: Metadata only - NO CONTENT
        log_func(
            f"Response generated {status} "
            f"(length: {response_length} chars, time: {total_time:.2f}s)"
        )
    
    elif env == 'development':
        # Development: Truncated content for debugging
        truncated_response = _truncate_content(str(response_content), max_length=200)
        redacted_response = _redact_pii(truncated_response)
        log_func(
            f"Response preview: {redacted_response} "
            f"(length: {response_length} chars, time: {total_time:.2f}s)"
        )
    
    else:  # staging
        # Staging: Production-like behavior (metadata only)
        log_func(
            f"Response generated {status} "
            f"(length: {response_length} chars, time: {total_time:.2f}s)"
        )


def safe_log_user_data(
    user_id: str,
    action: str,
    level: str = 'info',
    logger_instance: Optional[logging.Logger] = None,
    **kwargs
) -> None:
    """
    Safely log user activity with environment-aware privacy protection.
    
    Production Mode:
        Logs hashed user ID to prevent user tracking
        Example: "User action: chat_request (user_hash: a3f7c8d2)"
    
    Development Mode:
        Logs actual user ID for debugging
        Example: "User action: chat_request (user_id: user_12345)"
    
    Args:
        user_id: The user identifier
        action: The action being performed (e.g., 'chat_request', 'file_upload')
        level: Logging level ('info', 'debug', 'warning', 'error')
        logger_instance: Optional logger instance (defaults to module logger)
        **kwargs: Additional metadata to log (e.g., message_count=5)
    
    Examples:
        >>> safe_log_user_data('user_12345', 'chat_request', message_count=3)
        # Production: "User action: chat_request (user_hash: a3f7c8d2, message_count: 3)"
        # Development: "User action: chat_request (user_id: user_12345, message_count: 3)"
    """
    log = logger_instance or logger
    log_func = getattr(log, level.lower(), log.info)
    
    env = get_environment()
    
    # Build metadata string
    metadata_parts = []
    for key, value in kwargs.items():
        metadata_parts.append(f"{key}: {value}")
    
    metadata_str = ", ".join(metadata_parts) if metadata_parts else ""
    
    if env == 'production':
        # Production: Hash user ID for privacy
        user_hash = _hash_identifier(user_id, prefix='user')
        base_msg = f"User action: {action} (user_hash: {user_hash}"
        log_func(f"{base_msg}, {metadata_str})" if metadata_str else f"{base_msg})")
    
    elif env == 'development':
        # Development: Actual user ID for debugging
        base_msg = f"User action: {action} (user_id: {user_id}"
        log_func(f"{base_msg}, {metadata_str})" if metadata_str else f"{base_msg})")
    
    else:  # staging
        # Staging: Production-like behavior (hashed)
        user_hash = _hash_identifier(user_id, prefix='user')
        base_msg = f"User action: {action} (user_hash: {user_hash}"
        log_func(f"{base_msg}, {metadata_str})" if metadata_str else f"{base_msg})")


def safe_log_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    level: str = 'error',
    logger_instance: Optional[logging.Logger] = None
) -> str:
    """
    Safely log errors with environment-aware exception handling.
    
    Production Mode:
        Logs error code only (no exception details to prevent info leakage)
        Example: "Operation failed (error_code: PROC_001, endpoint: chat)"
    
    Development Mode:
        Logs full exception with sanitized context for debugging
        Example: "Error: ValueError: Invalid format (endpoint: chat, user_hash: a3f7c8d2)"
    
    Args:
        error: The exception object
        context: Optional dict with context (e.g., {'endpoint': 'chat', 'user_id': '123'})
        level: Logging level ('error', 'warning', 'critical')
        logger_instance: Optional logger instance (defaults to module logger)
    
    Returns:
        str: Error code (for production) or error message (for development)
    
    Examples:
        >>> try:
        ...     raise ValueError("Invalid query format")
        ... except Exception as e:
        ...     safe_log_error(e, context={'endpoint': 'chat'})
        # Production: "Operation failed (error_code: PROC_001, endpoint: chat)"
        # Development: "Error: ValueError: Invalid query format (endpoint: chat)"
    """
    log = logger_instance or logger
    log_func = getattr(log, level.lower(), log.error)
    
    env = get_environment()
    context = context or {}
    
    # Generate error code (hash of exception type + message)
    error_signature = f"{type(error).__name__}:{str(error)}"
    error_code = hashlib.sha256(error_signature.encode()).hexdigest()[:8].upper()
    
    # Sanitize context (hash user IDs, redact PII)
    sanitized_context = {}
    for key, value in context.items():
        if 'user' in key.lower() and 'id' in key.lower():
            # Hash user IDs in context
            sanitized_context[key] = _hash_identifier(str(value), prefix='user')
        elif isinstance(value, str):
            # Redact PII from string values
            sanitized_context[key] = _redact_pii(value)
        else:
            sanitized_context[key] = value
    
    # Build context string
    context_parts = [f"{k}: {v}" for k, v in sanitized_context.items()]
    context_str = ", ".join(context_parts) if context_parts else ""
    
    if env == 'production':
        # Production: Error code only - NO EXCEPTION DETAILS
        log_func(
            f"Operation failed (error_code: {error_code}"
            f"{', ' + context_str if context_str else ''})"
        )
        return error_code
    
    elif env == 'development':
        # Development: Full exception with sanitized context
        error_type = type(error).__name__
        error_msg = _redact_pii(str(error))
        log_func(
            f"Error: {error_type}: {error_msg}"
            f"{' (' + context_str + ')' if context_str else ''}",
            exc_info=True  # Include stack trace in development
        )
        return f"{error_type}: {error_msg}"
    
    else:  # staging
        # Staging: Error code with error type (no details)
        error_type = type(error).__name__
        log_func(
            f"Operation failed (error_type: {error_type}, error_code: {error_code}"
            f"{', ' + context_str if context_str else ''})"
        )
        return error_code


def safe_log_query(
    query: str,
    level: str = 'info',
    logger_instance: Optional[logging.Logger] = None
) -> None:
    """
    Safely log user queries with environment-aware behavior.
    
    Production Mode:
        Logs only query length (NO CONTENT)
        Example: "Query received (length: 45 chars)"
    
    Development Mode:
        Logs truncated query (first 100 chars) with PII redaction
        Example: "Query preview: What is the weather in... (length: 45 chars)"
    
    Args:
        query: The user query string
        level: Logging level ('info', 'debug', 'warning')
        logger_instance: Optional logger instance (defaults to module logger)
    
    Examples:
        >>> safe_log_query("What is the capital of France?")
        # Production: "Query received (length: 32 chars)"
        # Development: "Query preview: What is the capital of France? (length: 32 chars)"
    """
    log = logger_instance or logger
    log_func = getattr(log, level.lower(), log.info)
    
    env = get_environment()
    query_length = len(query)
    
    if env == 'production':
        # Production: Length only - NO CONTENT
        log_func(f"Query received (length: {query_length} chars)")
    
    elif env == 'development':
        # Development: Truncated query with PII redaction
        truncated_query = _truncate_content(query, max_length=100)
        redacted_query = _redact_pii(truncated_query)
        log_func(f"Query preview: {redacted_query} (length: {query_length} chars)")
    
    else:  # staging
        # Staging: Production-like behavior
        log_func(f"Query received (length: {query_length} chars)")


# Export public API
__all__ = [
    'get_environment',
    'safe_log_response',
    'safe_log_user_data',
    'safe_log_error',
    'safe_log_query',
]
