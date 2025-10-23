# 🔐 Secure Logging Developer Guide

**Brain-Heart Deep Research System**

---

## 📚 Table of Contents

1. [Quick Start](#quick-start)
2. [Why Secure Logging Matters](#why-secure-logging-matters)
3. [Environment Configuration](#environment-configuration)
4. [Usage Guide](#usage-guide)
5. [Best Practices](#best-practices)
6. [Common Patterns](#common-patterns)
7. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Import the Module

```python
from core.logging_security import (
    safe_log_response,
    safe_log_user_data,
    safe_log_error,
    safe_log_query
)
```

### Basic Usage

```python
# Log AI responses safely
result = {'response': 'Hello!', 'total_time': 2.5, 'success': True}
safe_log_response(result)
# Production: "Response generated successfully (length: 6 chars, time: 2.50s)"
# Development: "Response preview: Hello! (length: 6 chars, time: 2.50s)"

# Log user activity safely
safe_log_user_data('user_12345', 'chat_request', message_count=3)
# Production: "User action: chat_request (user_hash: user_a3f7c8d2, message_count: 3)"
# Development: "User action: chat_request (user_id: user_12345, message_count: 3)"

# Log errors safely
try:
    raise ValueError("Invalid input")
except Exception as e:
    safe_log_error(e, context={'endpoint': 'chat'})
# Production: "Operation failed (error_code: 5F3B2A1C, endpoint: chat)"
# Development: "Error: ValueError: Invalid input (endpoint: chat)"
```

---

## 🎯 Why Secure Logging Matters

### The Problem

Traditional logging can expose:

- **User Conversations** → Privacy violation (GDPR/CCPA)
- **Personal Identifiable Information (PII)** → Regulatory fines
- **Business Secrets** → Competitive disadvantage
- **API Keys/Tokens** → Security breach

### Real-World Impact

```python
# ❌ DANGEROUS: Traditional Logging
logging.info(f"User query: {user_query}")  # Exposes user conversations
logging.info(f"Response: {ai_response}")    # Exposes business logic
logging.info(f"User: {user_email}")         # GDPR violation
logging.error(f"Error: {str(e)}")           # May contain API keys

# ✅ SAFE: Secure Logging
safe_log_query(user_query)                  # Metadata only in production
safe_log_response(result)                   # Length/time only in production
safe_log_user_data(user_id, 'action')       # Hashed user ID in production
safe_log_error(e, context)                  # Error code only in production
```

### Compliance Requirements

- **GDPR (EU):** User data in logs = personal data processing
- **CCPA (California):** Consumers have right to know what's logged
- **HIPAA (Healthcare):** Health-related queries in logs = PHI violation
- **SOC 2:** Logs must be protected and access-controlled

---

## ⚙️ Environment Configuration

### Setting the Environment

The logging behavior changes based on the `ENVIRONMENT` variable:

**Windows PowerShell:**

```powershell
$env:ENVIRONMENT = "production"   # Metadata-only logging
$env:ENVIRONMENT = "development"  # Truncated content for debugging
$env:ENVIRONMENT = "staging"      # Production-like behavior
```

**Linux/Mac Bash:**

```bash
export ENVIRONMENT=production
export ENVIRONMENT=development
export ENVIRONMENT=staging
```

**Python (for testing):**

```python
import os
os.environ['ENVIRONMENT'] = 'development'
```

### Default Behavior

**If `ENVIRONMENT` is not set, defaults to `production` for security.**

This ensures that even if you forget to set the environment variable, your logs will be secure by default.

---

## 📖 Usage Guide

### 1. Logging AI Responses

#### Function Signature

```python
safe_log_response(
    response_data: Dict[str, Any],
    level: str = 'info',
    logger_instance: Optional[logging.Logger] = None
) -> None
```

#### Parameters

- `response_data`: Dictionary with keys:
  - `'response'`: The actual response content (str)
  - `'total_time'`: Time taken (float, optional)
  - `'success'`: Whether successful (bool, optional)
- `level`: Logging level ('info', 'debug', 'warning', 'error')
- `logger_instance`: Custom logger (optional)

#### Examples

```python
# Basic usage
result = {
    'response': 'The capital of France is Paris',
    'total_time': 2.5,
    'success': True
}
safe_log_response(result)

# With custom logger
import logging
custom_logger = logging.getLogger('my_app')
safe_log_response(result, level='debug', logger_instance=custom_logger)

# Failed response
result = {
    'response': '',
    'total_time': 0.5,
    'success': False
}
safe_log_response(result)
# Output: "Response generated with errors (length: 0 chars, time: 0.50s)"
```

#### Production Output

```
Response generated successfully (length: 32 chars, time: 2.50s)
```

#### Development Output

```
Response preview: The capital of France is Paris (length: 32 chars, time: 2.50s)
```

---

### 2. Logging User Activity

#### Function Signature

```python
safe_log_user_data(
    user_id: str,
    action: str,
    level: str = 'info',
    logger_instance: Optional[logging.Logger] = None,
    **kwargs
) -> None
```

#### Parameters

- `user_id`: User identifier (will be hashed in production)
- `action`: Action being performed (e.g., 'chat_request', 'file_upload')
- `level`: Logging level
- `logger_instance`: Custom logger (optional)
- `**kwargs`: Additional metadata to log

#### Examples

```python
# Basic usage
safe_log_user_data('user_12345', 'chat_request')

# With metadata
safe_log_user_data(
    'user_12345',
    'chat_request',
    message_count=5,
    session_duration=120.5,
    model='gpt-4'
)

# Custom logger
safe_log_user_data(
    'user_12345',
    'file_upload',
    file_count=3,
    logger_instance=custom_logger
)
```

#### Production Output

```
User action: chat_request (user_hash: user_a3f7c8d2, message_count: 5, session_duration: 120.5, model: gpt-4)
```

#### Development Output

```
User action: chat_request (user_id: user_12345, message_count: 5, session_duration: 120.5, model: gpt-4)
```

---

### 3. Logging Errors

#### Function Signature

```python
safe_log_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    level: str = 'error',
    logger_instance: Optional[logging.Logger] = None
) -> str
```

#### Parameters

- `error`: The exception object
- `context`: Optional context dict (will be sanitized)
- `level`: Logging level ('error', 'warning', 'critical')
- `logger_instance`: Custom logger (optional)

#### Returns

- `str`: Error code (production) or error message (development)

#### Examples

```python
# Basic usage
try:
    result = risky_operation()
except Exception as e:
    error_code = safe_log_error(e)
    return {"error": "Operation failed", "code": error_code}

# With context
try:
    process_user_data(user_id, data)
except Exception as e:
    safe_log_error(e, context={
        'endpoint': 'process_data',
        'user_id': user_id,
        'data_size': len(data)
    })

# Critical error
try:
    critical_operation()
except Exception as e:
    safe_log_error(e, context={'service': 'database'}, level='critical')
```

#### Production Output

```
Operation failed (error_code: 5F3B2A1C, endpoint: process_data, user_id: user_a3f7c8d2, data_size: 1024)
```

#### Development Output

```
Error: ValueError: Invalid data format (endpoint: process_data, user_id: user_12345, data_size: 1024)
```

_Note: Development mode includes full stack trace_

---

### 4. Logging User Queries

#### Function Signature

```python
safe_log_query(
    query: str,
    level: str = 'info',
    logger_instance: Optional[logging.Logger] = None
) -> None
```

#### Parameters

- `query`: The user query string
- `level`: Logging level
- `logger_instance`: Custom logger (optional)

#### Examples

```python
# Basic usage
user_query = "What is the weather in New York?"
safe_log_query(user_query)

# With custom logger
safe_log_query(user_query, level='debug', logger_instance=custom_logger)
```

#### Production Output

```
Query received (length: 34 chars)
```

#### Development Output

```
Query preview: What is the weather in New York? (length: 34 chars)
```

---

## ✅ Best Practices

### DO ✅

#### 1. Always Use Safe Logging Functions

```python
# ✅ GOOD
safe_log_response(result)
safe_log_user_data(user_id, 'action')
safe_log_error(exception, context)
```

#### 2. Log Metadata Instead of Content

```python
# ✅ GOOD: Metadata only
logging.info(f"Processed {len(items)} items in {duration:.2f}s")
logging.info(f"Response size: {len(response)} bytes")
logging.info(f"Request completed (status: {status_code})")
```

#### 3. Use Context for Error Logging

```python
# ✅ GOOD: Provide context without sensitive data
safe_log_error(e, context={
    'endpoint': 'chat',
    'operation': 'process_query',
    'duration': elapsed_time
})
```

#### 4. Hash Identifiers in Custom Logs

```python
# ✅ GOOD: Hash user IDs manually if needed
from core.logging_security import _hash_identifier
user_hash = _hash_identifier(user_id, prefix='user')
logging.info(f"Session created for {user_hash}")
```

### DON'T ❌

#### 1. Never Log User Content in Production

```python
# ❌ BAD: Exposes user conversations
logging.info(f"User query: {user_query}")
logging.info(f"AI response: {ai_response}")

# ✅ GOOD: Use safe functions
safe_log_query(user_query)
safe_log_response(result)
```

#### 2. Never Log PII Directly

```python
# ❌ BAD: GDPR/CCPA violation
logging.info(f"User email: {email}")
logging.info(f"User ID: {user_id}")
logging.info(f"Phone: {phone_number}")

# ✅ GOOD: Hash or use safe functions
safe_log_user_data(user_id, 'action')
```

#### 3. Never Log API Keys/Tokens

```python
# ❌ BAD: Security breach
logging.debug(f"API key: {api_key}")
logging.debug(f"Headers: {headers}")  # May contain tokens

# ✅ GOOD: Never log credentials
logging.info("API authentication successful")
```

#### 4. Don't Log Full Exception Details in Production

```python
# ❌ BAD: May expose sensitive data
logging.error(f"Error: {str(e)}")

# ✅ GOOD: Use safe_log_error
safe_log_error(e, context={'endpoint': 'chat'})
```

---

## 🎨 Common Patterns

### Pattern 1: Chat Endpoint Logging

```python
@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Log user request
        safe_log_user_data(
            request.user_id,
            'chat_request',
            message_count=len(request.messages)
        )

        # Process request
        result = await process_chat(request)

        # Log response
        safe_log_response(result)

        return JSONResponse(content=result, status_code=200)

    except Exception as e:
        # Log error
        safe_log_error(e, context={'endpoint': 'chat'})
        return JSONResponse(
            content={"error": "Processing failed"},
            status_code=500
        )
```

### Pattern 2: File Upload Logging

```python
@router.post("/upload")
async def upload_file(user_id: str, file: UploadFile):
    try:
        # Log upload start
        safe_log_user_data(
            user_id,
            'file_upload_start',
            filename_length=len(file.filename),
            content_type=file.content_type
        )

        # Process file
        result = process_file(file)

        # Log success
        safe_log_user_data(
            user_id,
            'file_upload_complete',
            file_size=result['size'],
            processing_time=result['time']
        )

        return {"status": "success"}

    except Exception as e:
        safe_log_error(e, context={
            'endpoint': 'upload',
            'user_id': user_id
        })
        return {"error": "Upload failed"}
```

### Pattern 3: Database Query Logging

```python
def query_database(user_id: str, query_params: dict):
    try:
        # Log query metadata (NOT the query itself)
        logging.info(f"Database query (params: {len(query_params)}, user_hash: {_hash_identifier(user_id, 'user')})")

        # Execute query
        results = db.execute(query_params)

        # Log results metadata
        logging.info(f"Query completed (results: {len(results)}, duration: {duration:.2f}s)")

        return results

    except Exception as e:
        safe_log_error(e, context={
            'operation': 'database_query',
            'user_id': user_id,
            'param_count': len(query_params)
        })
        raise
```

### Pattern 4: Authentication Logging

```python
def authenticate_user(user_id: str, credentials: dict):
    try:
        # ❌ NEVER log passwords or tokens
        # logging.debug(f"Auth attempt: {credentials}")  # DANGEROUS!

        # ✅ Log metadata only
        safe_log_user_data(user_id, 'auth_attempt')

        # Authenticate
        is_valid = verify_credentials(credentials)

        if is_valid:
            safe_log_user_data(user_id, 'auth_success')
        else:
            safe_log_user_data(user_id, 'auth_failure')

        return is_valid

    except Exception as e:
        safe_log_error(e, context={'operation': 'authentication'})
        return False
```

---

## 🔧 Troubleshooting

### Issue 1: Logs Show Too Much Detail in Production

**Symptom:** Production logs contain user conversations or PII

**Solution:**

```powershell
# Check environment variable
echo $env:ENVIRONMENT

# Set to production
$env:ENVIRONMENT = "production"

# Verify
python -c "from core.logging_security import get_environment; print(get_environment())"
```

### Issue 2: Logs Show Too Little Detail in Development

**Symptom:** Can't debug because logs don't show content

**Solution:**

```powershell
# Set to development
$env:ENVIRONMENT = "development"

# Restart application to pick up new environment
```

### Issue 3: PII Not Being Redacted

**Symptom:** Emails/phones visible in development logs

**Solution:**
The PII redaction works automatically in both environments. If you see PII:

1. Ensure you're using `safe_log_*` functions (not `logging.info` directly)
2. Check that content passes through `_redact_pii()` function
3. Verify regex patterns match your PII format

### Issue 4: User IDs Not Hashed in Production

**Symptom:** Actual user IDs visible in production logs

**Solution:**

```python
# ❌ Don't use direct logging
logging.info(f"User: {user_id}")

# ✅ Use safe function
safe_log_user_data(user_id, 'action')
```

### Issue 5: Error Codes Not Consistent

**Symptom:** Same error generates different codes

**Solution:**
Error codes are deterministic hashes of `Exception Type + Message`. Same exception always produces same code.

```python
# Same exception = same code
try:
    raise ValueError("Invalid input")
except Exception as e:
    code1 = safe_log_error(e)

try:
    raise ValueError("Invalid input")
except Exception as e:
    code2 = safe_log_error(e)

assert code1 == code2  # True
```

---

## 📊 Environment Comparison Table

| Feature               | Production         | Development              | Staging            |
| --------------------- | ------------------ | ------------------------ | ------------------ |
| **Response Content**  | ❌ Hidden          | ✅ Truncated (200 chars) | ❌ Hidden          |
| **Query Content**     | ❌ Hidden          | ✅ Truncated (100 chars) | ❌ Hidden          |
| **User IDs**          | 🔒 Hashed          | ✅ Plaintext             | 🔒 Hashed          |
| **Exception Details** | ❌ Error code only | ✅ Full stack trace      | ❌ Error code only |
| **PII Redaction**     | ✅ Always          | ✅ Always                | ✅ Always          |
| **Metadata Logging**  | ✅ Always          | ✅ Always                | ✅ Always          |

---

## 🎓 Learning Examples

### Example 1: Migrating Old Code

**Before (Insecure):**

```python
logging.info(f"User {user_id} asked: {query}")
result = process_query(query)
logging.info(f"Response: {result['response']}")
```

**After (Secure):**

```python
safe_log_user_data(user_id, 'chat_request')
safe_log_query(query)
result = process_query(query)
safe_log_response(result)
```

### Example 2: Error Handling

**Before (Insecure):**

```python
try:
    process_payment(user_id, card_number)
except Exception as e:
    logging.error(f"Payment failed for {user_id}: {str(e)}")
```

**After (Secure):**

```python
try:
    process_payment(user_id, card_number)
except Exception as e:
    safe_log_error(e, context={
        'operation': 'payment',
        'user_id': user_id
    })
```

### Example 3: Custom Integration

**Integrating with existing logger:**

```python
import logging
from core.logging_security import safe_log_response

# Your existing logger
app_logger = logging.getLogger('myapp')
app_logger.setLevel(logging.INFO)

# Use with safe logging
result = {'response': 'Hello!', 'total_time': 1.5}
safe_log_response(result, logger_instance=app_logger)
```

---

## 🔐 Security Checklist

Before deploying to production:

- [ ] `ENVIRONMENT=production` is set
- [ ] All `logging.info(f"...{user_data}...")` replaced with `safe_log_*`
- [ ] No direct logging of user queries/responses
- [ ] No API keys/tokens in logs
- [ ] No email addresses in logs
- [ ] Error handling uses `safe_log_error`
- [ ] User IDs are hashed in production logs
- [ ] Tests confirm no PII leakage (run `pytest test_logging_security.py`)

---

## 📚 Additional Resources

- **Security Plan:** `.azure/logging_security_plan.md`
- **Tests:** `test_logging_security.py` (36 test cases)
- **Source Code:** `core/logging_security.py`
- **OWASP Logging Guide:** https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html

---

**Questions or Issues?**  
Review the test suite (`test_logging_security.py`) for comprehensive examples of every function and edge case.

**Last Updated:** 2025  
**Version:** 1.0
