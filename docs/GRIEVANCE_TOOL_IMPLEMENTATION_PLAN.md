# Grievance Tool Implementation Plan

## ✅ IMPLEMENTATION COMPLETE

All components have been implemented and integrated.

---

## Overview

Add a **Grievance Tool** to the CS-Agent bot that takes natural language instructions from District Magistrates (DM) in India, converts them to structured parameters, and handles missing information through clarification questions.

---

## Architecture Flow

```
User Query → OptimizedAgent
                ↓
        LLM Analysis (detects grievance tool needed)
                ↓
        tools_to_use: ["grievance"]
                ↓
        ToolManager.execute_tool("grievance", query=...)
                ↓
        GrievanceAgent (core/mcp/grievance_agent.py)
            ├─ Converts natural language → structured params
            ├─ If params missing → returns clarification_message
            └─ If params complete → returns structured grievance data
                ↓
        Response back to OptimizedAgent
                ↓
        Final response to user (or clarification question)
```

---

## Grievance Parameters (From Claude Chat)

### Required Parameters (must ask if missing)

| Parameter     | Description          | Example Values                                                                            |
| ------------- | -------------------- | ----------------------------------------------------------------------------------------- |
| `category`    | Department/Service   | PDS, Revenue, Police, Health, Education, Public Works, Social Welfare, Sanitation, Others |
| `location`    | Block/Tehsil/Village | "Lucknow", "Ward 5", "Village XYZ"                                                        |
| `description` | The actual grievance | Free text describing the issue                                                            |

### Optional Parameters (can be inferred or left blank)

| Parameter             | Description            | Example Values                                |
| --------------------- | ---------------------- | --------------------------------------------- |
| `sub_category`        | Specific issue type    | "Ration card not issued", "Dealer misconduct" |
| `priority`            | Severity level         | Critical, High, Medium, Low                   |
| `complainant_type`    | Who is complaining     | Individual, Group, Organization               |
| `expected_resolution` | What complainant wants | "Issue new ration card", "Take action"        |

---

## Files to Create/Modify

### 1. NEW: `core/mcp/grievance_agent.py`

**Purpose:** Converts natural language grievance → structured parameters  
**Pattern:** Same as `QueryAgent` for MongoDB/Redis

```python
# Key components:
- GrievanceParams dataclass (all params)
- GrievanceResult dataclass (success/clarification/error)
- GrievanceAgent class
  - LLM call to extract params
  - Validation for required fields
  - Clarification question generation
  - Detailed logging of extracted params
```

### 2. MODIFY: `core/tools.py`

**Changes:**

- Add `_grievance_manager` and `_grievance_enabled` to ToolManager
- Add `initialize_grievance_async()` method
- Add `grievance_available` property
- Update `get_available_tools()` to include grievance
- Update `execute_tool()` to handle grievance tool
- Add grievance to tool descriptions

### 3. MODIFY: `core/config.py`

**Changes:**

- Add `GRIEVANCE_ENABLED` env variable support
- No other changes needed (simple boolean toggle)

### 4. MODIFY: `core/optimized_agent.py`

**Changes:**

- Update `_get_tools_prompt_section()` to include grievance tool
- Update `get_available_tools()` call to include `include_grievance=True`
- Track `_grievance_available` flag
- Log grievance availability on init

### 5. MODIFY: `.env`

**Add:**

```env
# Grievance Tool Configuration
GRIEVANCE_ENABLED=true
```

### 6. NEW: `tests/test_grievance_flow.py`

**Purpose:** End-to-end tests for grievance detection and handling

---

## .env Toggle Implementation

### Current State Analysis

| Integration        | Env Variable                                      | How Disabled    |
| ------------------ | ------------------------------------------------- | --------------- |
| Zapier             | `MCP_ENABLED=false` OR no `ZAPIER_MCP_SERVER_URL` | Auto-disabled   |
| MongoDB            | No `MONGODB_MCP_CONNECTION_STRING`                | Auto-disabled   |
| Redis              | No `REDIS_MCP_URL`                                | Auto-disabled   |
| Web Search         | No API keys                                       | Auto-disabled   |
| Language Detection | `LANGUAGE_DETECTION_ENABLED=false`                | Explicit toggle |

### Changes Required

**MongoDB:** Add explicit toggle (currently only auto-detects)

```env
MONGODB_ENABLED=true  # NEW - can disable even with connection string
```

**Redis:** Add explicit toggle (currently only auto-detects)

```env
REDIS_ENABLED=true  # NEW - can disable even with URL
```

**Zapier:** Already has `MCP_ENABLED` toggle ✅

**Grievance:** New toggle

```env
GRIEVANCE_ENABLED=true  # NEW
```

---

## Implementation Details

### GrievanceAgent System Prompt

```
You are a grievance parameter extractor for District Magistrates in India.

Extract these parameters from the user's grievance description:

REQUIRED (ask if missing):
- category*: PDS/Revenue/Police/Health/Education/Public Works/Social Welfare/Sanitation/Others
- location*: Block/Tehsil/Village/Ward
- description*: The actual grievance text

OPTIONAL (infer or leave empty):
- sub_category: Specific issue within category
- priority: Critical/High/Medium/Low (infer from urgency words)
- complainant_type: Individual/Group/Organization
- expected_resolution: What the complainant wants done

Rules:
1. If ANY required field is missing, set needs_clarification=true
2. Generate a clear clarification question for missing fields
3. For optional fields, infer if possible, otherwise omit
4. Return JSON only

Example input: "People in ward 5 are not getting ration for 3 months"
Example output:
{
  "category": "PDS",
  "sub_category": "Ration not distributed",
  "location": "Ward 5",
  "description": "People not getting ration for 3 months",
  "priority": "High",
  "complainant_type": "Group",
  "needs_clarification": false
}

Example input: "There's a road problem"
Example output:
{
  "needs_clarification": true,
  "missing_fields": ["location", "description"],
  "clarification_message": "Please provide: 1) Which area/village/ward has this road problem? 2) What exactly is the issue (potholes, no road, flooding)?"
}
```

### Logging Requirements

The GrievanceAgent must log:

1. Input natural language text
2. LLM response (raw)
3. Extracted parameters (structured)
4. Missing fields (if any)
5. Clarification message (if any)

---

## Test Cases

### Test File: `tests/test_grievance_flow.py`

#### Test 1: Grievance Tool Should Be Triggered

```python
query = "Log a complaint: People in Sector 15 Lucknow are not getting drinking water for 2 weeks"
# Expected: tools_to_use should include "grievance"
# Expected: GrievanceAgent extracts category=Public Works, location=Sector 15 Lucknow, etc.
```

#### Test 2: Grievance Tool Should NOT Be Triggered

```python
query = "What's the weather in Lucknow today?"
# Expected: tools_to_use should NOT include "grievance"
# Expected: web_search tool used instead
```

#### Test 3: Clarification Needed

```python
query = "There's a problem with the school"
# Expected: grievance tool triggered
# Expected: needs_clarification=True
# Expected: clarification_message asking for location and specific issue
```

#### Test 4: Complete Grievance

```python
query = "Register complaint: Ration dealer in Village Rampur Block Bakshi is giving less wheat, only 3kg instead of 5kg"
# Expected: All params extracted
# Expected: needs_clarification=False
# Expected: category=PDS, sub_category=Dealer misconduct, location=Village Rampur Block Bakshi
```

---

## Implementation Order

1. **Create `grievance_agent.py`** - Core logic with logging
2. **Update `config.py`** - Add all enable/disable toggles
3. **Update `tools.py`** - Add grievance to ToolManager
4. **Update `optimized_agent.py`** - Add grievance to prompts
5. **Update `.env`** - Add toggle variables
6. **Create `test_grievance_flow.py`** - Test all scenarios

---

## Summary of .env Variables

```env
# ============================================================================
# INTEGRATION TOGGLES
# ============================================================================

# MCP/Zapier Integration
ZAPIER_ENABLED=true
MCP_ENABLED=true

# MongoDB Integration
MONGODB_ENABLED=true
MONGODB_MCP_CONNECTION_STRING=mongodb+srv://...

# Redis Integration
REDIS_ENABLED=true
REDIS_MCP_URL=redis://...

# Grievance Tool
GRIEVANCE_ENABLED=true

# Language Detection
LANGUAGE_DETECTION_ENABLED=true
```

---

## Files Created/Modified

### NEW FILES:

1. `core/mcp/grievance_agent.py` - GrievanceAgent with LLM-based param extraction
2. `tests/test_grievance_flow.py` - Test file with unit and integration tests
3. `docs/GRIEVANCE_TOOL_IMPLEMENTATION_PLAN.md` - This plan document

### MODIFIED FILES:

1. `.env` - Added GRIEVANCE_ENABLED, MONGODB_ENABLED, REDIS_ENABLED, ZAPIER_ENABLED
2. `core/tools.py` - Added grievance to ToolManager
3. `core/optimized_agent.py` - Added grievance to prompts and availability tracking
4. `api/chat.py` - Added grievance initialization in lifespan

---

## No Over-Engineering Principles

1. **Reuse existing patterns** - GrievanceAgent follows QueryAgent pattern exactly
2. **Simple LLM prompt** - No complex parsing, just JSON extraction
3. **Standard logging** - Use existing logger, no custom log handlers
4. **Minimal config changes** - Just boolean toggles
5. **Test what matters** - Focus on detection and param extraction
