"""
Grievance Agent - Updated for Backend Integration
==================================================

Extracts ALL required fields for backend payload submission.
"""

import os
import json
import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

from core.llm_client import LLMClient

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class LocationParams:
    """Location structure for backend"""
    state: Optional[str] = None
    district: Optional[str] = None
    ward: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def is_complete(self) -> bool:
        return all([self.state, self.district, self.ward])


@dataclass
class SubmittedByParams:
    """Submitter information for backend"""
    userId: Optional[str] = None
    name: Optional[str] = None
    contactType: Optional[str] = None  # PHONE, EMAIL
    contactDetails: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def is_complete(self) -> bool:
        return all([self.userId, self.name, self.contactType, self.contactDetails])


@dataclass
class GrievanceParams:
    """Complete grievance parameters matching backend payload"""
    # Required fields
    category: Optional[str] = None
    sub_category: Optional[str] = None
    location: Optional[LocationParams] = None
    priority: Optional[str] = None  # HIGH, MEDIUM, LOW, CRITICAL
    submittedBy: Optional[SubmittedByParams] = None
    complainantType: Optional[str] = None  # INDIVIDUAL, GROUP, ORGANIZATION
    expectedResolution: Optional[str] = None
    description: Optional[str] = None  # Internal field for context
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary matching backend format"""
        result = {}
        if self.category:
            result['category'] = self.category
        if self.sub_category:
            result['sub_category'] = self.sub_category
        if self.location:
            result['location'] = self.location.to_dict()
        if self.priority:
            result['priority'] = self.priority
        if self.submittedBy:
            result['submittedBy'] = self.submittedBy.to_dict()
        if self.complainantType:
            result['complainantType'] = self.complainantType
        if self.expectedResolution:
            result['expectedResolution'] = self.expectedResolution
        return result
    
    def get_missing_required(self) -> List[str]:
        """Get list of missing required fields"""
        missing = []
        if not self.category:
            missing.append('category')
        if not self.sub_category:
            missing.append('sub_category')
        if not self.location or not self.location.is_complete():
            if not self.location:
                missing.extend(['location.state', 'location.district', 'location.ward'])
            else:
                if not self.location.state:
                    missing.append('location.state')
                if not self.location.district:
                    missing.append('location.district')
                if not self.location.ward:
                    missing.append('location.ward')
        if not self.priority:
            missing.append('priority')
        if not self.submittedBy or not self.submittedBy.is_complete():
            if not self.submittedBy:
                missing.extend(['submittedBy.name', 'submittedBy.contactType', 'submittedBy.contactDetails'])
            else:
                if not self.submittedBy.name:
                    missing.append('submittedBy.name')
                if not self.submittedBy.contactType:
                    missing.append('submittedBy.contactType')
                if not self.submittedBy.contactDetails:
                    missing.append('submittedBy.contactDetails')
        if not self.complainantType:
            missing.append('complainantType')
        if not self.expectedResolution:
            missing.append('expectedResolution')
        return missing


@dataclass
class GrievanceResult:
    """Result from grievance extraction"""
    success: bool
    params: Optional[GrievanceParams] = None
    error: Optional[str] = None
    llm_response: Optional[str] = None
    needs_clarification: bool = False
    clarification_message: Optional[str] = None
    missing_fields: Optional[List[str]] = None
    raw_input: Optional[str] = None


@dataclass
class LLMConfig:
    """LLM configuration for grievance agent"""
    provider: str
    model: str
    api_key: str
    max_tokens: int = 2048
    temperature: float = 0.1
    base_url: Optional[str] = None


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

GRIEVANCE_SYSTEM_PROMPT = """You are a grievance parameter extractor for District Magistrates (DM) in India.

Your job is to extract ALL required structured parameters from natural language grievance descriptions to submit to the backend system.

PARAMETER DEFINITIONS:

REQUIRED PARAMETERS (you MUST extract or ask for ALL of these):

1. category*: Which government department/service?
   Valid values: Water Supply, PDS (Ration), Revenue (Land Records), Police (Law & Order), 
   Health, Education, Public Works (Roads/Electricity), Panchayati Raj, Social Welfare, Sanitation, Others

2. sub_category*: Specific issue type within category
   Examples for Water Supply: "No Water Connection", "Water Quality Issue", "Irregular Supply"
   Examples for PDS: "Ration Card Not Issued", "Dealer Misconduct", "Wrong Quantity", "Quality Issues"
   Examples for Health: "Hospital Staff Absence", "Medicine Shortage", "Ambulance Delay"
   Examples for Education: "Teacher Absence", "Infrastructure Issue", "Mid-Day Meal Problem"

3. location*: Complete location details (ALL three required)
   - state*: Full state name (e.g., "Maharashtra", "Uttar Pradesh")
   - district*: District name (e.g., "Pune", "Lucknow")
   - ward*: Ward/Village/Block/Tehsil name or number (e.g., "12", "Ward 5", "Rampur")

4. priority*: Severity level (MUST be one of these exact values)
   - CRITICAL: Life-threatening, major public safety, requires immediate action
   - HIGH: Affects many people, ongoing hardship, urgent resolution needed
   - MEDIUM: Individual cases, moderate impact, regular timeline
   - LOW: Suggestions, minor issues, non-urgent

5. submittedBy*: Person filing the complaint (ALL fields required)
   - userId: Auto-generate as "USR-" + random 4 digits (e.g., "USR-1001")
   - name*: Full name of the person
   - contactType*: Either "PHONE" or "EMAIL"
   - contactDetails*: Phone number (with +91) or email address

6. complainantType*: Type of complainant (MUST be exact value)
   - INDIVIDUAL: Single person filing
   - GROUP: Community, multiple people, residents
   - ORGANIZATION: NGO, institution, official body

7. expectedResolution*: What the complainant wants done (be specific)
   Example: "Restore regular water supply within 24 hours"
   Example: "Issue ration card within 7 days"
   Example: "Fix the road potholes before monsoon"

8. description: Internal field - The actual grievance text (extract the core problem)

EXTRACTION RULES:
1. If ANY required field cannot be extracted from the input, set needs_clarification=true
2. Generate a CLEAR, HELPFUL clarification question in Hindi-English mix for ALL missing fields
3. Priority can often be inferred from urgency words like "urgent", "emergency", "dying", "3 months"
4. Always infer complainantType from context (single person = INDIVIDUAL, "log", "people" = GROUP)
5. For submittedBy.userId, always auto-generate as "USR-XXXX" where XXXX is random 4 digits
6. If contact info not provided, ASK for it - we need either phone or email
7. If expectedResolution not mentioned, infer a reasonable one based on the grievance type
8. All enum values (priority, contactType, complainantType) must be UPPERCASE

RESPONSE FORMAT (JSON only, no markdown):

If ALL required parameters are extractable or can be reasonably inferred:
{
  "category": "exact category from list",
  "sub_category": "specific issue type",
  "location": {
    "state": "full state name",
    "district": "district name",
    "ward": "ward/village/block"
  },
  "priority": "CRITICAL|HIGH|MEDIUM|LOW",
  "submittedBy": {
    "userId": "USR-XXXX",
    "name": "person's name",
    "contactType": "PHONE|EMAIL",
    "contactDetails": "+91XXXXXXXXXX or email"
  },
  "complainantType": "INDIVIDUAL|GROUP|ORGANIZATION",
  "expectedResolution": "specific action requested",
  "description": "core grievance text",
  "needs_clarification": false
}

If ANY required parameter is MISSING or CANNOT be inferred:
{
  "category": "extracted or null",
  "sub_category": "extracted or null",
  "location": {
    "state": "extracted or null",
    "district": "extracted or null",
    "ward": "extracted or null"
  },
  "priority": "inferred or null",
  "submittedBy": {
    "name": "extracted or null",
    "contactType": "inferred or null",
    "contactDetails": "extracted or null"
  },
  "complainantType": "inferred or null",
  "expectedResolution": "inferred or null",
  "description": "extracted or null",
  "needs_clarification": true,
  "missing_fields": ["field1", "field2.subfield"],
  "clarification_message": "Kripya batayein: 1) [missing info 1]? 2) [missing info 2]?"
}

EXAMPLES:

Input: "Main Ravi Kumar hun, Ward 5 Lucknow mein log 3 mahine se ration nahi mil raha, dealer ghar pe nahi milta. Mera number +919876543210 hai. Jaldi kuch karo"
Output:
{
  "category": "PDS (Ration)",
  "sub_category": "Dealer Misconduct",
  "location": {
    "state": "Uttar Pradesh",
    "district": "Lucknow",
    "ward": "Ward 5"
  },
  "priority": "HIGH",
  "submittedBy": {
    "userId": "USR-1523",
    "name": "Ravi Kumar",
    "contactType": "PHONE",
    "contactDetails": "+919876543210"
  },
  "complainantType": "GROUP",
  "expectedResolution": "Ensure regular ration distribution and dealer accountability within 7 days",
  "description": "People not getting ration for 3 months, dealer not available at home",
  "needs_clarification": false
}

Input: "Pune ke Ward 12 mein paani nahi aa raha 2 hafton se"
Output:
{
  "category": "Water Supply",
  "sub_category": "Irregular Supply",
  "location": {
    "state": "Maharashtra",
    "district": "Pune",
    "ward": "Ward 12"
  },
  "priority": "HIGH",
  "description": "No water supply for 2 weeks",
  "needs_clarification": true,
  "missing_fields": ["submittedBy.name", "submittedBy.contactDetails"],
  "clarification_message": "Kripya batayein: 1) Aapka naam kya hai? 2) Aapka phone number ya email kya hai?"
}

Input: "School mein problem hai"
Output:
{
  "category": "Education",
  "needs_clarification": true,
  "missing_fields": ["sub_category", "location.state", "location.district", "location.ward", "submittedBy.name", "submittedBy.contactDetails", "description"],
  "clarification_message": "Kripya batayein: 1) Kaun sa school aur kahan hai (state, district, ward)? 2) Exactly kya problem hai? 3) Aapka naam aur phone number/email?"
}
"""


# =============================================================================
# GRIEVANCE AGENT CLASS
# =============================================================================

class GrievanceAgent:
    """
    Grievance Parameter Extraction Agent - Backend Integration Ready.
    
    Extracts ALL required fields for backend payload submission.
    """
    
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        llm_client: Optional[LLMClient] = None
    ):
        """Initialize Grievance Agent"""
        if llm_client is not None:
            self.llm_client = llm_client
            self.llm_config = getattr(llm_client, "config", None)
            self._owns_client = False
        else:
            if llm_config is None:
                self.llm_config = self._load_config_from_env()
            else:
                self.llm_config = llm_config
            self.llm_client = LLMClient(self.llm_config)
            self._owns_client = True
        
        logger.info("✅ GrievanceAgent initialized (Backend-ready)")
    
    async def close(self) -> None:
        """Close resources"""
        if self._owns_client and self.llm_client:
            await self.llm_client.close_session()
    
    def _load_config_from_env(self) -> LLMConfig:
        """Load LLM config from environment variables"""
        provider = os.getenv("HEART_LLM_PROVIDER", "openrouter")
        model = os.getenv("HEART_LLM_MODEL", "meta-llama/llama-4-maverick")
        api_key = os.getenv("HEART_LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        
        if not api_key:
            raise ValueError("No API key found. Set HEART_LLM_API_KEY or OPENROUTER_API_KEY in .env")

        base_url: Optional[str] = None
        if provider == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
        elif provider == "groq":
            base_url = "https://api.groq.com/openai/v1"
        elif provider == "deepseek":
            base_url = "https://api.deepseek.com/v1"

        return LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            max_tokens=2048,
            temperature=0.1,
            base_url=base_url,
        )
    
    async def execute(self, instruction: str) -> GrievanceResult:
        """
        Execute grievance parameter extraction from natural language.
        
        Args:
            instruction: Natural language grievance description
            
        Returns:
            GrievanceResult with extracted params or clarification request
        """
        logger.info(f"🔍 GrievanceAgent processing: {instruction[:100]}...")
        
        try:
            # Step 1: Build prompt
            user_prompt = f"""Extract ALL grievance parameters from this complaint:

"{instruction}"

Return JSON only, no other text. Make sure to include ALL required fields."""

            # Step 2: Call LLM
            llm_response = await self._call_llm(user_prompt)
            logger.debug(f"📄 LLM Raw Response: {llm_response}")
            
            # Step 3: Parse response
            parsed = self._parse_response(llm_response)
            
            # Step 4: Check if clarification is needed
            if parsed.get("needs_clarification"):
                missing = parsed.get("missing_fields", [])
                clarification = parsed.get("clarification_message", "Please provide more details.")
                
                logger.info(f"❓ Clarification needed - Missing: {missing}")
                
                return GrievanceResult(
                    success=False,
                    needs_clarification=True,
                    clarification_message=clarification,
                    missing_fields=missing,
                    llm_response=llm_response,
                    raw_input=instruction
                )
            
            # Step 5: Build structured params
            location = None
            loc_data = parsed.get("location", {})
            if isinstance(loc_data, dict):
                location = LocationParams(
                    state=loc_data.get("state"),
                    district=loc_data.get("district"),
                    ward=loc_data.get("ward")
                )
            
            submitted_by = None
            sub_data = parsed.get("submittedBy", {})
            if isinstance(sub_data, dict):
                submitted_by = SubmittedByParams(
                    userId=sub_data.get("userId"),
                    name=sub_data.get("name"),
                    contactType=sub_data.get("contactType"),
                    contactDetails=sub_data.get("contactDetails")
                )
            
            params = GrievanceParams(
                category=parsed.get("category"),
                sub_category=parsed.get("sub_category"),
                location=location,
                priority=parsed.get("priority"),
                submittedBy=submitted_by,
                complainantType=parsed.get("complainantType"),
                expectedResolution=parsed.get("expectedResolution"),
                description=parsed.get("description")
            )
            
            # Step 6: Validate required fields
            missing_required = params.get_missing_required()
            if missing_required:
                logger.warning(f"⚠️ Missing required fields: {missing_required}")
                return GrievanceResult(
                    success=False,
                    needs_clarification=True,
                    clarification_message=f"Kripya batayein: {', '.join(missing_required)}",
                    missing_fields=missing_required,
                    llm_response=llm_response,
                    raw_input=instruction
                )
            
            logger.info(f"✅ Grievance extracted successfully")
            logger.info(f"   Category: {params.category}")
            logger.info(f"   Location: {params.location.district}, {params.location.state}")
            logger.info(f"   Priority: {params.priority}")
            logger.info(f"   Submitter: {params.submittedBy.name}")
            
            return GrievanceResult(
                success=True,
                params=params,
                llm_response=llm_response,
                raw_input=instruction
            )
            
        except Exception as e:
            logger.error(f"❌ GrievanceAgent error: {e}")
            return GrievanceResult(
                success=False,
                error=str(e),
                raw_input=instruction
            )
    
    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM for parameter extraction"""
        if not self.llm_client:
            raise RuntimeError("LLM client not configured")

        temperature = getattr(self.llm_config, "temperature", 0.1) if self.llm_config else 0.1
        max_tokens = getattr(self.llm_config, "max_tokens", 2048) if self.llm_config else 2048

        response = await self.llm_client.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=GRIEVANCE_SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract grievance parameters"""
        if not response or not response.strip():
            return {"needs_clarification": True, "missing_fields": ["all"], 
                    "clarification_message": "Could not parse grievance. Please describe clearly."}
        
        response = response.strip()
        
        # Try direct JSON parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Extract from markdown code block
        json_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response)
        if json_block_match:
            try:
                return json.loads(json_block_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Find JSON object in text
        brace_start = response.find('{')
        if brace_start != -1:
            depth = 0
            in_string = False
            escape_next = False
            
            for i, char in enumerate(response[brace_start:], brace_start):
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\':
                    escape_next = True
                    continue
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if not in_string:
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0:
                            json_str = response[brace_start:i + 1]
                            try:
                                return json.loads(json_str)
                            except json.JSONDecodeError:
                                break
        
        return {"needs_clarification": True, "missing_fields": ["parsing_failed"],
                "clarification_message": "Could not understand. Please describe: problem, location (state/district/ward), your name and contact."}


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

async def extract_grievance(instruction: str, llm_client: Optional[LLMClient] = None) -> GrievanceResult:
    """
    Convenience function to extract grievance parameters.
    
    Args:
        instruction: Natural language grievance description
        llm_client: Optional shared LLM client
        
    Returns:
        GrievanceResult with extracted params or clarification request
    """
    agent = GrievanceAgent(llm_client=llm_client)
    try:
        return await agent.execute(instruction)
    finally:
        await agent.close()