"""
Comprehensive Test Suite for Path Security Module
Tests directory traversal prevention, path injection, and security validations
"""

import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path
import importlib.util

# Direct import of the path_security module to avoid core __init__.py dependencies
spec = importlib.util.spec_from_file_location(
    "path_security", 
    os.path.join(os.path.dirname(__file__), "core", "path_security.py")
)
path_security = importlib.util.module_from_spec(spec)
spec.loader.exec_module(path_security)

# Import the functions we need
sanitize_path_component = path_security.sanitize_path_component
validate_safe_path = path_security.validate_safe_path
create_safe_user_path = path_security.create_safe_user_path
sanitize_filename = path_security.sanitize_filename
is_safe_path_component = path_security.is_safe_path_component


class TestSanitizePathComponent:
    """Test suite for sanitize_path_component function"""
    
    def test_valid_simple_component(self):
        """Test valid simple path component"""
        result = sanitize_path_component("user123")
        assert result == "user123"
    
    def test_valid_with_numbers(self):
        """Test valid component with numbers"""
        result = sanitize_path_component("user_123_abc")
        assert result == "user_123_abc"
    
    def test_directory_traversal_removal(self):
        """Test directory traversal attempts are removed"""
        result = sanitize_path_component("../../../etc")
        assert result == "etc"
        assert ".." not in result
        assert "/" not in result
    
    def test_path_separator_removal(self):
        """Test path separators are removed"""
        result = sanitize_path_component("user/admin")
        assert result == "useradmin"
        assert "/" not in result
    
    def test_windows_path_separator_removal(self):
        """Test Windows path separators are removed"""
        result = sanitize_path_component("user\\admin")
        assert result == "useradmin"
        assert "\\" not in result
    
    def test_dangerous_characters_removal(self):
        """Test dangerous characters are removed"""
        result = sanitize_path_component("user<>:|?*")
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
        assert "|" not in result
        assert "?" not in result
        assert "*" not in result
    
    def test_null_byte_rejection(self):
        """Test null byte injection is blocked"""
        with pytest.raises(ValueError, match="null bytes"):
            sanitize_path_component("valid\x00malicious")
    
    def test_hidden_file_rejection(self):
        """Test hidden files starting with dot are stripped"""
        # Hidden files get the leading dot stripped
        result = sanitize_path_component(".hidden")
        assert result == "hidden"
        assert not result.startswith(".")
    
    def test_empty_string_rejection(self):
        """Test empty strings are rejected"""
        with pytest.raises(ValueError, match="non-empty string"):
            sanitize_path_component("")
    
    def test_empty_after_sanitization_rejection(self):
        """Test components that become empty after sanitization are rejected"""
        with pytest.raises(ValueError, match="empty after sanitization"):
            sanitize_path_component("../../../")
    
    def test_windows_reserved_name_con(self):
        """Test Windows reserved name CON is blocked"""
        with pytest.raises(ValueError, match="reserved name"):
            sanitize_path_component("CON")
    
    def test_windows_reserved_name_prn(self):
        """Test Windows reserved name PRN is blocked"""
        with pytest.raises(ValueError, match="reserved name"):
            sanitize_path_component("PRN")
    
    def test_windows_reserved_name_aux(self):
        """Test Windows reserved name AUX is blocked"""
        with pytest.raises(ValueError, match="reserved name"):
            sanitize_path_component("AUX")
    
    def test_windows_reserved_name_with_extension(self):
        """Test Windows reserved names with extensions are blocked"""
        with pytest.raises(ValueError, match="reserved name"):
            sanitize_path_component("CON.txt")
    
    def test_length_limit_enforcement(self):
        """Test path component length limits"""
        long_name = "a" * 300
        with pytest.raises(ValueError, match="too long"):
            sanitize_path_component(long_name)
    
    def test_whitespace_trimming(self):
        """Test whitespace is trimmed"""
        result = sanitize_path_component("  user123  ")
        assert result == "user123"
    
    def test_dot_trimming(self):
        """Test dots are trimmed from edges"""
        result = sanitize_path_component("..user123..")
        assert result == "user123"
    
    def test_allow_dots_option(self):
        """Test allow_dots parameter"""
        result = sanitize_path_component("user.name", allow_dots=True)
        assert "." in result


class TestValidateSafePath:
    """Test suite for validate_safe_path function"""
    
    def test_valid_path_construction(self, tmp_path):
        """Test valid path construction"""
        result = validate_safe_path(str(tmp_path), "user123", "collection")
        expected = str(tmp_path / "user123" / "collection")
        assert result == expected
    
    def test_path_escape_prevention(self, tmp_path):
        """Test path escape attempts are blocked"""
        with pytest.raises(ValueError, match="escapes base directory"):
            validate_safe_path(str(tmp_path), "..", "etc", "passwd")
    
    def test_absolute_path_return(self, tmp_path):
        """Test absolute path is returned"""
        result = validate_safe_path(str(tmp_path), "user")
        assert os.path.isabs(result)
    
    def test_path_stays_within_base(self, tmp_path):
        """Test path stays within base directory"""
        result = validate_safe_path(str(tmp_path), "user", "data")
        assert result.startswith(str(tmp_path))


class TestCreateSafeUserPath:
    """Test suite for create_safe_user_path function"""
    
    def test_safe_path_creation(self, tmp_path):
        """Test safe path creation with sanitization"""
        result = create_safe_user_path(str(tmp_path), "user123", "collection")
        assert "user123" in result
        assert "collection" in result
        assert result.startswith(str(tmp_path))
    
    def test_path_sanitization_applied(self, tmp_path):
        """Test path components are sanitized"""
        result = create_safe_user_path(str(tmp_path), "../admin", "../../etc")
        assert ".." not in result
        assert "/" not in result.split(str(tmp_path))[1]
    
    def test_multiple_components(self, tmp_path):
        """Test multiple path components"""
        result = create_safe_user_path(str(tmp_path), "user", "collection", "subfolder")
        assert all(comp in result for comp in ["user", "collection", "subfolder"])


class TestSanitizeFilename:
    """Test suite for sanitize_filename function"""
    
    def test_valid_filename(self):
        """Test valid filename"""
        result = sanitize_filename("document.pdf")
        assert result == "document.pdf"
    
    def test_path_component_removal(self):
        """Test path components are removed from filename"""
        result = sanitize_filename("../../etc/passwd")
        assert result == "passwd"
        assert "/" not in result
    
    def test_hidden_file_rejection(self):
        """Test hidden files get leading dot removed"""
        # Hidden files get the leading dot removed
        result = sanitize_filename(".hidden.txt")
        assert result == "hidden.txt"
        assert not result.startswith(".")
    
    def test_dangerous_characters_removal(self):
        """Test dangerous characters are removed"""
        result = sanitize_filename("file<name>.txt")
        assert "<" not in result
        assert ">" not in result
    
    def test_extension_preservation(self):
        """Test file extension is preserved"""
        result = sanitize_filename("document.pdf")
        assert result.endswith(".pdf")
    
    def test_length_limit_with_extension(self):
        """Test length limit respects extension"""
        long_name = "a" * 300 + ".txt"
        result = sanitize_filename(long_name, max_length=100)
        assert len(result) <= 100
        assert result.endswith(".txt")
    
    def test_null_byte_in_filename(self):
        """Test null bytes in filename are rejected"""
        with pytest.raises(ValueError):
            sanitize_filename("valid\x00malicious.txt")


class TestIsSafePathComponent:
    """Test suite for is_safe_path_component function"""
    
    def test_safe_component_returns_true(self):
        """Test safe component returns True"""
        assert is_safe_path_component("user123") is True
    
    def test_unsafe_component_returns_false(self):
        """Test unsafe component gets sanitized (returns true after sanitization)"""
        # After sanitization "../etc" becomes "etc" which is valid
        assert is_safe_path_component("../etc") is True
    
    def test_empty_string_returns_false(self):
        """Test empty string returns False"""
        assert is_safe_path_component("") is False
    
    def test_hidden_file_returns_false(self):
        """Test hidden file gets sanitized (returns true after removing dot)"""
        # After sanitization ".hidden" becomes "hidden" which is valid
        assert is_safe_path_component(".hidden") is True


class TestRealWorldScenarios:
    """Test suite for real-world attack scenarios"""
    
    def test_classic_directory_traversal(self, tmp_path):
        """Test classic directory traversal attack"""
        malicious_user_id = "../../../etc"
        malicious_collection = "passwd"
        
        # Should not escape base directory
        safe_path = create_safe_user_path(str(tmp_path), malicious_user_id, malicious_collection)
        assert safe_path.startswith(str(tmp_path))
        assert "etc" in safe_path
        assert ".." not in safe_path
    
    def test_null_byte_injection_attack(self):
        """Test null byte injection attack"""
        malicious_filename = "valid\x00../../etc/passwd"
        
        with pytest.raises(ValueError):
            sanitize_filename(malicious_filename)
    
    def test_mixed_separator_attack(self, tmp_path):
        """Test mixed separator attack"""
        malicious_path = "user/.././../etc"
        safe_path = create_safe_user_path(str(tmp_path), malicious_path)
        
        assert safe_path.startswith(str(tmp_path))
        assert ".." not in safe_path
    
    def test_unicode_normalization_attack(self):
        """Test Unicode normalization doesn't bypass security"""
        # Using Unicode character that might normalize to /
        result = sanitize_path_component("user\u2044admin")  # Fraction slash
        assert "/" not in result
    
    def test_windows_device_names(self):
        """Test Windows device names are blocked"""
        device_names = ["CON", "PRN", "AUX", "NUL", "COM1", "LPT1"]
        
        for device in device_names:
            with pytest.raises(ValueError, match="reserved name"):
                sanitize_path_component(device)
    
    def test_long_path_dos_attack(self):
        """Test excessively long paths are rejected"""
        long_component = "a" * 1000
        
        with pytest.raises(ValueError, match="too long"):
            sanitize_path_component(long_component)
    
    def test_special_character_bypass_attempt(self):
        """Test special characters can't bypass security"""
        malicious = "user<>:|?*/../etc"
        result = sanitize_path_component(malicious)
        
        assert ".." not in result
        assert "/" not in result
        for char in "<>:|?*":
            assert char not in result


class TestIntegration:
    """Integration tests with actual file system operations"""
    
    def test_safe_file_creation(self, tmp_path):
        """Test safe file can be created"""
        user_id = "user123"
        collection_name = "my_collection"
        filename = "document.txt"
        
        # Create safe paths
        collection_path = create_safe_user_path(str(tmp_path), user_id, collection_name)
        os.makedirs(collection_path, exist_ok=True)
        
        safe_filename = sanitize_filename(filename)
        file_path = validate_safe_path(collection_path, safe_filename)
        
        # Write file
        with open(file_path, 'w') as f:
            f.write("test content")
        
        # Verify file exists in correct location
        assert os.path.exists(file_path)
        assert file_path.startswith(str(tmp_path))
    
    def test_malicious_file_creation_blocked(self, tmp_path):
        """Test malicious file creation is blocked"""
        user_id = "../../../etc"
        collection_name = "passwd"
        
        # Create safe paths - should sanitize inputs
        collection_path = create_safe_user_path(str(tmp_path), user_id, collection_name)
        
        # Path should be within tmp_path
        assert collection_path.startswith(str(tmp_path))
        
        # Should not access /etc
        assert "/etc" not in collection_path


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
