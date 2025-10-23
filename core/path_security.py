"""
Path Security Module
====================

Provides comprehensive security utilities to prevent directory traversal attacks
and path injection vulnerabilities.

Security Features:
- Directory traversal prevention (../, ..\\)
- Path injection blocking
- Null byte injection detection
- Hidden file protection
- Windows reserved name validation
- Path length enforcement
- Boundary validation

Usage:
    from core.path_security import create_safe_user_path, sanitize_filename
    
    # Create safe paths
    safe_path = create_safe_user_path(base_dir, user_id, collection_name)
    
    # Sanitize filenames
    safe_filename = sanitize_filename(uploaded_file.name)
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Windows reserved names that should be blocked
WINDOWS_RESERVED_NAMES = {
    'CON', 'PRN', 'AUX', 'NUL',
    'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
    'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
}

# Dangerous characters that should be removed from path components
DANGEROUS_CHARS = r'[<>:"|?*\x00-\x1f]'


def sanitize_path_component(component: str, allow_dots: bool = False) -> str:
    """
    Sanitize a single path component to prevent directory traversal and path injection.
    
    Args:
        component: The path component to sanitize (e.g., user_id, collection_name)
        allow_dots: If True, allows dots in the component (but not .. sequences)
    
    Returns:
        Sanitized path component safe for use in file paths
    
    Raises:
        ValueError: If the component is invalid or contains dangerous patterns
    
    Examples:
        >>> sanitize_path_component("user123")
        'user123'
        >>> sanitize_path_component("../../../etc")
        'etc'
        >>> sanitize_path_component("file\x00.txt")
        ValueError: Path component contains null bytes
    """
    if not component or not isinstance(component, str):
        raise ValueError("Path component must be a non-empty string")
    
    # Check for null bytes (null byte injection attack)
    if '\x00' in component:
        logger.warning(f"Null byte injection attempt detected: {repr(component)}")
        raise ValueError("Path component contains null bytes")
    
    # Remove any directory traversal attempts
    component = component.replace('../', '').replace('..\\', '')
    component = component.replace('..', '')
    
    # Remove path separators
    component = component.replace('/', '').replace('\\', '')
    
    # Remove dangerous characters
    component = re.sub(DANGEROUS_CHARS, '', component)
    
    # Strip whitespace and dots from edges
    component = component.strip()
    if not allow_dots:
        component = component.strip('.')
    
    # Validate against empty result
    if not component:
        raise ValueError("Path component is empty after sanitization")
    
    # Block hidden files (starting with .) - must be done AFTER stripping
    if component.startswith('.'):
        logger.warning(f"Hidden file access attempt detected: {component}")
        raise ValueError("Path component cannot start with '.'")
    
    # Check against Windows reserved names
    component_upper = component.upper()
    # Also check without extension
    base_name = component_upper.split('.')[0] if '.' in component_upper else component_upper
    
    if component_upper in WINDOWS_RESERVED_NAMES or base_name in WINDOWS_RESERVED_NAMES:
        logger.warning(f"Windows reserved name detected: {component}")
        raise ValueError(f"Path component uses reserved name: {component}")
    
    # Enforce reasonable length limits (255 is typical filesystem limit)
    if len(component) > 255:
        raise ValueError(f"Path component too long: {len(component)} characters (max 255)")
    
    return component


def validate_safe_path(base_path: str, *components: str) -> str:
    """
    Validate that a constructed path stays within the base directory.
    
    Args:
        base_path: The base directory that paths must stay within
        *components: Path components to join with base_path
    
    Returns:
        Absolute path as a string, guaranteed to be within base_path
    
    Raises:
        ValueError: If the resulting path escapes the base directory
    
    Examples:
        >>> validate_safe_path("/app/data", "user123", "collection")
        '/app/data/user123/collection'
        >>> validate_safe_path("/app/data", "../etc")
        ValueError: Path escapes base directory
    """
    # Convert to Path objects and resolve to absolute paths
    base = Path(base_path).resolve()
    
    # Join all components
    target = base.joinpath(*components).resolve()
    
    # Check if target is within base
    try:
        target.relative_to(base)
    except ValueError:
        logger.warning(f"Path escape attempt: base={base}, target={target}")
        raise ValueError(f"Path escapes base directory: {target}")
    
    return str(target)


def create_safe_user_path(base_path: str, user_id: str, *components: str) -> str:
    """
    Create a safe, validated path for user data storage.
    
    Combines sanitization and validation to ensure paths are both clean and
    confined to the intended directory structure.
    
    Args:
        base_path: Base directory for user data
        user_id: User identifier (will be sanitized)
        *components: Additional path components like collection_name (will be sanitized)
    
    Returns:
        Absolute path as a string, sanitized and validated
    
    Raises:
        ValueError: If any component is invalid or path escapes base
    
    Examples:
        >>> create_safe_user_path("/app/data", "user123", "my_collection")
        '/app/data/user123/my_collection'
        >>> create_safe_user_path("/app/data", "../admin", "secrets")
        '/app/data/admin/secrets'  # Traversal attempt sanitized
    """
    # Sanitize user_id
    safe_user_id = sanitize_path_component(user_id)
    
    # Sanitize all additional components
    safe_components = [sanitize_path_component(comp) for comp in components]
    
    # Build and validate the full path
    all_components = [safe_user_id] + safe_components
    return validate_safe_path(base_path, *all_components)


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize an uploaded filename to prevent path injection.
    
    Args:
        filename: Original filename from upload
        max_length: Maximum allowed filename length
    
    Returns:
        Sanitized filename safe for storage
    
    Raises:
        ValueError: If filename is invalid
    
    Examples:
        >>> sanitize_filename("document.pdf")
        'document.pdf'
        >>> sanitize_filename("../../etc/passwd")
        'passwd'
        >>> sanitize_filename(".hidden.txt")
        'hidden.txt'
    """
    if not filename or not isinstance(filename, str):
        raise ValueError("Filename must be a non-empty string")
    
    # Check for null bytes first
    if '\x00' in filename:
        logger.warning(f"Null byte injection attempt in filename: {repr(filename)}")
        raise ValueError("Filename contains null bytes")
    
    # Remove any path components (in case full path was provided)
    filename = os.path.basename(filename)
    
    # Split into name and extension
    if '.' in filename:
        parts = filename.rsplit('.', 1)
        name = parts[0]
        extension = parts[1]
    else:
        name = filename
        extension = ''
    
    # Sanitize the name part (but allow hidden files to be converted)
    # Remove leading dots first before sanitizing
    name = name.lstrip('.')
    if not name:
        raise ValueError("Filename is empty after removing leading dots")
    
    try:
        safe_name = sanitize_path_component(name, allow_dots=False)
    except ValueError as e:
        # If it's a length error, we'll handle it below
        if "too long" in str(e):
            safe_name = name[:255]  # Truncate to max length
        else:
            raise ValueError(f"Invalid filename: {e}")
    
    # Sanitize extension (allow dots in extension)
    if extension:
        # Remove dangerous characters from extension
        extension = re.sub(DANGEROUS_CHARS, '', extension)
        extension = extension.strip()
        
        if extension:
            filename = f"{safe_name}.{extension}"
        else:
            filename = safe_name
    else:
        filename = safe_name
    
    # Enforce length limit
    if len(filename) > max_length:
        # Truncate but preserve extension
        if extension:
            max_name_length = max_length - len(extension) - 1
            if max_name_length > 0:
                safe_name = safe_name[:max_name_length]
                filename = f"{safe_name}.{extension}"
            else:
                # If extension alone is too long, just truncate everything
                filename = filename[:max_length]
        else:
            filename = filename[:max_length]
    
    return filename


def is_safe_path_component(component: str) -> bool:
    """
    Check if a path component is safe without raising exceptions.
    
    Args:
        component: Path component to check
    
    Returns:
        True if component is safe, False otherwise
    
    Examples:
        >>> is_safe_path_component("user123")
        True
        >>> is_safe_path_component("../etc")
        False
    """
    try:
        sanitize_path_component(component)
        return True
    except (ValueError, Exception):
        return False
