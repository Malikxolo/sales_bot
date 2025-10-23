"""
🔐 Comprehensive Test Suite for Logging Security Module

Tests cover:
- Environment detection (production/development/staging)
- Response content truncation and metadata-only logging
- User ID hashing for privacy
- PII redaction (emails, phone numbers, API keys)
- Error sanitization and context protection
- Query logging safety

Run with: pytest test_logging_security.py -v
"""

import os
import pytest
import logging
import importlib.util
from unittest.mock import MagicMock, patch
from io import StringIO

# Import the module directly to avoid core/__init__.py issues
spec = importlib.util.spec_from_file_location(
    "logging_security",
    "core/logging_security.py"
)
logging_security = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logging_security)

# Import functions
get_environment = logging_security.get_environment
safe_log_response = logging_security.safe_log_response
safe_log_user_data = logging_security.safe_log_user_data
safe_log_error = logging_security.safe_log_error
safe_log_query = logging_security.safe_log_query
_hash_identifier = logging_security._hash_identifier
_truncate_content = logging_security._truncate_content
_redact_pii = logging_security._redact_pii


class TestEnvironmentDetection:
    """Test environment detection and defaults"""
    
    def test_production_environment(self):
        """Should detect production environment"""
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            assert get_environment() == 'production'
    
    def test_development_environment(self):
        """Should detect development environment"""
        with patch.dict(os.environ, {'ENVIRONMENT': 'development'}):
            assert get_environment() == 'development'
    
    def test_staging_environment(self):
        """Should detect staging environment"""
        with patch.dict(os.environ, {'ENVIRONMENT': 'staging'}):
            assert get_environment() == 'staging'
    
    def test_default_to_production(self):
        """Should default to production when ENVIRONMENT is not set"""
        with patch.dict(os.environ, {}, clear=True):
            # Remove ENVIRONMENT if it exists
            os.environ.pop('ENVIRONMENT', None)
            assert get_environment() == 'production'
    
    def test_invalid_environment_defaults_to_production(self):
        """Should default to production for invalid ENVIRONMENT values"""
        with patch.dict(os.environ, {'ENVIRONMENT': 'invalid_env'}):
            assert get_environment() == 'production'
    
    def test_case_insensitive_environment(self):
        """Should handle case-insensitive environment values"""
        with patch.dict(os.environ, {'ENVIRONMENT': 'PRODUCTION'}):
            assert get_environment() == 'production'
        
        with patch.dict(os.environ, {'ENVIRONMENT': 'Development'}):
            assert get_environment() == 'development'


class TestResponseLogging:
    """Test safe_log_response function"""
    
    def setup_method(self):
        """Setup test logger"""
        self.test_logger = logging.getLogger('test_response')
        self.test_logger.setLevel(logging.DEBUG)
        self.log_stream = StringIO()
        handler = logging.StreamHandler(self.log_stream)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.test_logger.handlers = [handler]
    
    def get_log_output(self):
        """Get log output as string"""
        return self.log_stream.getvalue()
    
    def test_production_response_metadata_only(self):
        """Production should log ONLY metadata (no content)"""
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            response_data = {
                'response': 'This is a sensitive AI response with user data',
                'total_time': 2.5,
                'success': True
            }
            
            safe_log_response(response_data, logger_instance=self.test_logger)
            log_output = self.get_log_output()
            
            # Should contain metadata (actual length is 46, not 45)
            assert 'length: 46 chars' in log_output
            assert 'time: 2.50s' in log_output
            assert 'successfully' in log_output
            
            # Should NOT contain actual response content
            assert 'sensitive' not in log_output.lower()
            assert 'AI response' not in log_output
            assert 'user data' not in log_output
    
    def test_development_response_truncated_content(self):
        """Development should log truncated content (max 200 chars)"""
        with patch.dict(os.environ, {'ENVIRONMENT': 'development'}):
            long_response = 'A' * 500  # 500 character response
            response_data = {
                'response': long_response,
                'total_time': 1.5,
                'success': True
            }
            
            safe_log_response(response_data, logger_instance=self.test_logger)
            log_output = self.get_log_output()
            
            # Should contain truncated preview
            assert 'Response preview:' in log_output
            assert 'A' * 200 in log_output  # First 200 chars
            assert '...' in log_output  # Truncation indicator
            
            # Should NOT contain full response
            assert 'A' * 500 not in log_output
    
    def test_production_no_sensitive_data_leakage(self):
        """Production should never leak sensitive data"""
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            response_data = {
                'response': 'User email is john@example.com and phone is 555-1234',
                'total_time': 1.0,
                'success': True
            }
            
            safe_log_response(response_data, logger_instance=self.test_logger)
            log_output = self.get_log_output()
            
            # Should NOT contain PII
            assert 'john@example.com' not in log_output
            assert '555-1234' not in log_output
            assert 'email' not in log_output.lower()
    
    def test_development_pii_redaction(self):
        """Development should redact PII even in truncated content"""
        with patch.dict(os.environ, {'ENVIRONMENT': 'development'}):
            response_data = {
                'response': 'Contact me at john@example.com or call 555-1234',
                'total_time': 1.0,
                'success': True
            }
            
            safe_log_response(response_data, logger_instance=self.test_logger)
            log_output = self.get_log_output()
            
            # Should redact email and phone
            assert '[EMAIL]' in log_output
            assert '[PHONE]' in log_output
            assert 'john@example.com' not in log_output
            assert '555-1234' not in log_output
    
    def test_staging_behaves_like_production(self):
        """Staging should behave like production (metadata only)"""
        with patch.dict(os.environ, {'ENVIRONMENT': 'staging'}):
            response_data = {
                'response': 'Sensitive staging data',
                'total_time': 2.0,
                'success': True
            }
            
            safe_log_response(response_data, logger_instance=self.test_logger)
            log_output = self.get_log_output()
            
            # Should be metadata only
            assert 'length:' in log_output
            assert 'time:' in log_output
            
            # Should NOT contain content
            assert 'Sensitive' not in log_output


class TestUserDataLogging:
    """Test safe_log_user_data function"""
    
    def setup_method(self):
        """Setup test logger"""
        self.test_logger = logging.getLogger('test_user')
        self.test_logger.setLevel(logging.DEBUG)
        self.log_stream = StringIO()
        handler = logging.StreamHandler(self.log_stream)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.test_logger.handlers = [handler]
    
    def get_log_output(self):
        """Get log output as string"""
        return self.log_stream.getvalue()
    
    def test_production_user_id_hashed(self):
        """Production should hash user IDs"""
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            safe_log_user_data(
                'user_12345',
                'chat_request',
                logger_instance=self.test_logger
            )
            log_output = self.get_log_output()
            
            # Should contain hash, not actual user ID
            assert 'user_hash:' in log_output
            assert 'user_' in log_output
            
            # Should NOT contain actual user ID
            assert 'user_12345' not in log_output
    
    def test_development_user_id_plaintext(self):
        """Development should log actual user IDs for debugging"""
        with patch.dict(os.environ, {'ENVIRONMENT': 'development'}):
            safe_log_user_data(
                'user_12345',
                'chat_request',
                logger_instance=self.test_logger
            )
            log_output = self.get_log_output()
            
            # Should contain actual user ID
            assert 'user_id: user_12345' in log_output
    
    def test_user_hash_consistency(self):
        """Same user ID should produce same hash"""
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            hash1 = _hash_identifier('user_12345', prefix='user')
            hash2 = _hash_identifier('user_12345', prefix='user')
            
            assert hash1 == hash2
            assert hash1.startswith('user_')
    
    def test_different_user_ids_different_hashes(self):
        """Different user IDs should produce different hashes"""
        hash1 = _hash_identifier('user_12345', prefix='user')
        hash2 = _hash_identifier('user_67890', prefix='user')
        
        assert hash1 != hash2
    
    def test_additional_metadata_logged(self):
        """Should log additional metadata"""
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            safe_log_user_data(
                'user_123',
                'chat_request',
                message_count=5,
                session_id='abc123',
                logger_instance=self.test_logger
            )
            log_output = self.get_log_output()
            
            assert 'message_count: 5' in log_output
            assert 'session_id: abc123' in log_output


class TestErrorLogging:
    """Test safe_log_error function"""
    
    def setup_method(self):
        """Setup test logger"""
        self.test_logger = logging.getLogger('test_error')
        self.test_logger.setLevel(logging.DEBUG)
        self.log_stream = StringIO()
        handler = logging.StreamHandler(self.log_stream)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.test_logger.handlers = [handler]
    
    def get_log_output(self):
        """Get log output as string"""
        return self.log_stream.getvalue()
    
    def test_production_error_code_only(self):
        """Production should log error code only (no details)"""
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            try:
                raise ValueError("Sensitive error message with user data")
            except Exception as e:
                error_code = safe_log_error(
                    e,
                    context={'endpoint': 'chat'},
                    logger_instance=self.test_logger
                )
                log_output = self.get_log_output()
                
                # Should contain error code
                assert 'error_code:' in log_output
                assert error_code in log_output
                assert len(error_code) == 8  # 8-char hash
                
                # Should NOT contain exception details
                assert 'Sensitive error message' not in log_output
                assert 'ValueError' not in log_output
    
    def test_development_full_exception_details(self):
        """Development should log full exception for debugging"""
        with patch.dict(os.environ, {'ENVIRONMENT': 'development'}):
            try:
                raise ValueError("Debug error message")
            except Exception as e:
                safe_log_error(
                    e,
                    context={'endpoint': 'chat'},
                    logger_instance=self.test_logger
                )
                log_output = self.get_log_output()
                
                # Should contain exception type and message
                assert 'ValueError' in log_output
                assert 'Debug error message' in log_output
    
    def test_error_context_sanitization(self):
        """Should sanitize user IDs in error context"""
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            try:
                raise RuntimeError("Error occurred")
            except Exception as e:
                safe_log_error(
                    e,
                    context={'user_id': 'user_12345', 'endpoint': 'chat'},
                    logger_instance=self.test_logger
                )
                log_output = self.get_log_output()
                
                # Should contain hashed user_id
                assert 'user_id: user_' in log_output
                
                # Should NOT contain actual user_id
                assert 'user_12345' not in log_output
    
    def test_error_pii_redaction_in_context(self):
        """Should redact PII from error context"""
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            try:
                raise RuntimeError("Error")
            except Exception as e:
                safe_log_error(
                    e,
                    context={'email': 'john@example.com'},
                    logger_instance=self.test_logger
                )
                log_output = self.get_log_output()
                
                # Should redact email
                assert '[EMAIL]' in log_output
                assert 'john@example.com' not in log_output


class TestPIIRedaction:
    """Test PII redaction functionality"""
    
    def test_redact_email_addresses(self):
        """Should redact email addresses"""
        text = "Contact me at john@example.com or jane.doe@company.org"
        redacted = _redact_pii(text)
        
        assert '[EMAIL]' in redacted
        assert 'john@example.com' not in redacted
        assert 'jane.doe@company.org' not in redacted
    
    def test_redact_phone_numbers(self):
        """Should redact various phone number formats"""
        text = "Call 555-1234 or 555.5678 or 5559999 or +1-555-1234"
        redacted = _redact_pii(text)
        
        assert '[PHONE]' in redacted
        assert '555-1234' not in redacted
        assert '555.5678' not in redacted
    
    def test_redact_api_keys(self):
        """Should redact API keys"""
        text = "API key: sk-1234567890abcdefghijklmnop and token-abcd1234567890efgh"
        redacted = _redact_pii(text)
        
        assert '[API_KEY]' in redacted
        assert 'sk-1234567890abcdefghijklmnop' not in redacted
    
    def test_redact_credit_cards(self):
        """Should redact credit card numbers"""
        text = "Card: 1234-5678-9012-3456 or 1234567890123456"
        redacted = _redact_pii(text)
        
        assert '[CREDIT_CARD]' in redacted
        assert '1234-5678-9012-3456' not in redacted
    
    def test_redact_ssn(self):
        """Should redact Social Security Numbers"""
        text = "SSN: 123-45-6789"
        redacted = _redact_pii(text)
        
        assert '[SSN]' in redacted
        assert '123-45-6789' not in redacted
    
    def test_multiple_pii_types(self):
        """Should redact multiple PII types in same text"""
        text = "Email: john@example.com, Phone: 555-1234, Card: 1234-5678-9012-3456"
        redacted = _redact_pii(text)
        
        assert '[EMAIL]' in redacted
        assert '[PHONE]' in redacted
        assert '[CREDIT_CARD]' in redacted
        assert 'john@example.com' not in redacted
        assert '555-1234' not in redacted


class TestContentTruncation:
    """Test content truncation functionality"""
    
    def test_truncate_long_content(self):
        """Should truncate content longer than max_length"""
        content = "A" * 500
        truncated = _truncate_content(content, max_length=200)
        
        assert len(truncated) == 203  # 200 + "..."
        assert truncated.endswith("...")
        assert truncated.startswith("A" * 200)
    
    def test_keep_short_content(self):
        """Should not truncate content shorter than max_length"""
        content = "Short content"
        truncated = _truncate_content(content, max_length=200)
        
        assert truncated == content
        assert not truncated.endswith("...")
    
    def test_truncate_exact_length(self):
        """Should not truncate content exactly at max_length"""
        content = "A" * 200
        truncated = _truncate_content(content, max_length=200)
        
        assert truncated == content
        assert not truncated.endswith("...")
    
    def test_truncate_empty_content(self):
        """Should handle empty content"""
        truncated = _truncate_content("", max_length=200)
        assert truncated == ""
    
    def test_truncate_custom_max_length(self):
        """Should respect custom max_length"""
        content = "A" * 100
        truncated = _truncate_content(content, max_length=50)
        
        assert len(truncated) == 53  # 50 + "..."
        assert truncated.endswith("...")


class TestQueryLogging:
    """Test safe_log_query function"""
    
    def setup_method(self):
        """Setup test logger"""
        self.test_logger = logging.getLogger('test_query')
        self.test_logger.setLevel(logging.DEBUG)
        self.log_stream = StringIO()
        handler = logging.StreamHandler(self.log_stream)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.test_logger.handlers = [handler]
    
    def get_log_output(self):
        """Get log output as string"""
        return self.log_stream.getvalue()
    
    def test_production_query_length_only(self):
        """Production should log query length only"""
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            query = "What is the capital of France?"
            safe_log_query(query, logger_instance=self.test_logger)
            log_output = self.get_log_output()
            
            # Should contain length (actual length is 30, not 32)
            assert 'length: 30 chars' in log_output
            
            # Should NOT contain query content
            assert 'capital' not in log_output
            assert 'France' not in log_output
    
    def test_development_query_preview(self):
        """Development should log truncated query"""
        with patch.dict(os.environ, {'ENVIRONMENT': 'development'}):
            query = "What is the weather today?"
            safe_log_query(query, logger_instance=self.test_logger)
            log_output = self.get_log_output()
            
            # Should contain preview
            assert 'Query preview:' in log_output
            assert 'weather' in log_output
    
    def test_development_query_pii_redaction(self):
        """Development should redact PII from query preview"""
        with patch.dict(os.environ, {'ENVIRONMENT': 'development'}):
            query = "Send email to john@example.com"
            safe_log_query(query, logger_instance=self.test_logger)
            log_output = self.get_log_output()
            
            # Should redact email
            assert '[EMAIL]' in log_output
            assert 'john@example.com' not in log_output


class TestIntegrationScenarios:
    """Test real-world integration scenarios"""
    
    def setup_method(self):
        """Setup test logger"""
        self.test_logger = logging.getLogger('test_integration')
        self.test_logger.setLevel(logging.DEBUG)
        self.log_stream = StringIO()
        handler = logging.StreamHandler(self.log_stream)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.test_logger.handlers = [handler]
    
    def get_log_output(self):
        """Get log output as string"""
        return self.log_stream.getvalue()
    
    def test_production_chat_flow_no_leakage(self):
        """Complete production chat flow should not leak any sensitive data"""
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            # User makes request
            user_id = 'user_12345'
            safe_log_user_data(user_id, 'chat_request', logger_instance=self.test_logger)
            
            # AI generates response
            response_data = {
                'response': 'The capital of France is Paris. Contact me at support@example.com',
                'total_time': 2.5,
                'success': True
            }
            safe_log_response(response_data, logger_instance=self.test_logger)
            
            # Check logs
            log_output = self.get_log_output()
            
            # Should NOT contain any sensitive data
            assert 'user_12345' not in log_output
            assert 'France' not in log_output
            assert 'Paris' not in log_output
            assert 'support@example.com' not in log_output
            
            # Should contain only metadata
            assert 'user_hash:' in log_output
            assert 'length:' in log_output
            assert 'time:' in log_output
    
    def test_development_chat_flow_with_debugging(self):
        """Development chat flow should provide debugging info"""
        with patch.dict(os.environ, {'ENVIRONMENT': 'development'}):
            # User makes request
            user_id = 'user_12345'
            safe_log_user_data(user_id, 'chat_request', logger_instance=self.test_logger)
            
            # AI generates response
            response_data = {
                'response': 'The capital of France is Paris',
                'total_time': 2.5,
                'success': True
            }
            safe_log_response(response_data, logger_instance=self.test_logger)
            
            # Check logs
            log_output = self.get_log_output()
            
            # Should contain user ID and response preview
            assert 'user_id: user_12345' in log_output
            assert 'Response preview:' in log_output
            assert 'capital' in log_output


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
