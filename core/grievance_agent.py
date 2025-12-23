"""
Grievance Agent for DM Office
==============================

Simplified agent for District Magistrate office grievance handling.
Only uses RAG and Grievance tools. No business detection, no sales logic.

Toggle via .env: GRIEVANCE_AGENT_ENABLED=true
"""

import json
import logging
import asyncio
import re
import aiohttp
from typing import Dict, List, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
from os import getenv
# from mem0 import AsyncMemory
from functools import partial
from .config import AddBackgroundTask, memory_config, SARVAM_SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)


class GrievanceAgent:
    """
    Simplified agent for DM Office grievance handling.
    Only 2 tools: RAG and Grievance
    Only 3 prompts: Language Detection, Analysis, Response Generation
    """
    
    def __init__(self, llm, tool_manager, language_detector_llm=None, indic_llm=None):
        """
        Initialize GrievanceAgent.
        
        Args:
            llm: Single LLM for analysis and response (meta-llama/llama-3.3-70b-instruct)
            tool_manager: ToolManager with RAG and Grievance tools
            language_detector_llm: Optional LLM for language detection
            indic_llm: Optional LLM for Indic language responses
        """
        self.llm = llm
        self.indic_llm = indic_llm if indic_llm else llm
        self.language_detector_llm = language_detector_llm
        self.language_detection_enabled = language_detector_llm is not None
        self.tool_manager = tool_manager
        
        # Only RAG and Grievance tools
        self.available_tools = ["rag", "grievance"]
        
        # Memory for context
        # self.memory = AsyncMemory(memory_config)
        self.task_queue: asyncio.Queue["AddBackgroundTask"] = asyncio.Queue()
        self._worker_started = False
        
        # Check tool availability
        self._grievance_available = tool_manager.grievance_available
        
        # Backend URL for posting grievances
        self.backend_url = getenv("GRIEVANCE_BACKEND_URL")
        self.resolve_ward_url = getenv("RESOLVE_WARD")
        
        logger.info(f"✅ GrievanceAgent initialized")
        logger.info(f"   Available tools: {self.available_tools}")
        logger.info(f"   Language Detection: {'ENABLED ✅' if self.language_detection_enabled else 'DISABLED ⚠️'}")
        logger.info(f"   Grievance Tool: {'ENABLED ✅' if self._grievance_available else 'DISABLED ⚠️'}")
        logger.info(f"   Backend URL: {self.backend_url}")
    
    async def process_query(self, query: str, chat_history: List[Dict] = None, user_id: str = None, mode: str = None, source: Optional[str] = None) -> Dict[str, Any]:
        """Process query through GrievanceAgent pipeline"""
        self._start_worker_if_needed()
        logger.info(f"🏛️ GRIEVANCE AGENT PROCESSING: '{query}'")
        start_time = datetime.now()
        
        # Initialize variables
        detected_language = "english"
        english_query = query
        original_query = query
        grievance_id = None
        
        try:
            # STEP 1: Language Detection (if enabled)
            if self.language_detection_enabled:
                logger.info(f"🌍 LANGUAGE DETECTION LAYER...")
                lang_result = await self._detect_and_translate(query)
                detected_language = lang_result["detected_language"]
                english_query = lang_result["english_translation"]
                original_query = lang_result["original_query"]
                
                logger.info(f"   Detected: {detected_language}")
                logger.info(f"   English: {english_query}")
            
            processing_query = english_query
            
            # STEP 2: Get memories
            # memory_results = await self.memory.search(processing_query[:100], user_id=user_id, limit=5)
            # memories = "\n".join([
            #     f"- {item['memory']}" 
            #     for item in memory_results.get("results", []) 
            #     if item.get("memory")
            # ]) or "No previous context."
            memories = ""
            # logger.info(f"📝 Retrieved memories: {len(memories)} chars")
            
            # STEP 3: Analysis
            analysis_start = datetime.now()
            analysis = await self._grievance_analysis(processing_query, chat_history, memories)
            analysis_time = (datetime.now() - analysis_start).total_seconds()
            logger.info(f"📊 Analysis completed in {analysis_time:.2f}s")
            logger.info(f"   Intent: {analysis.get('semantic_intent', 'Unknown')}")
            logger.info(f"   Tools: {analysis.get('tools_to_use', [])}")
            
            # STEP 4: Execute tools
            tools_to_use = analysis.get('tools_to_use', [])
            tool_start = datetime.now()
            tool_results = await self._execute_tools(tools_to_use, processing_query, analysis, user_id)
            tool_time = (datetime.now() - tool_start).total_seconds()
            logger.info(f"🔧 Tools executed in {tool_time:.2f}s")
            
            # Post to backend if grievance successful
            # Find any grievance result (could be 'grievance' or 'grievance_0', etc.)
            grievance_result = None
            for key, val in tool_results.items():
                if key == 'grievance' or (key.startswith('grievance_') and key.split('_')[-1].isdigit()):
                    grievance_result = val
                    break
            
            if (grievance_result and 
                grievance_result.get("success") and 
                not grievance_result.get("needs_clarification")):
                params = grievance_result.get("params")
                if params:
                    grievance_id = await self._post_grievance_to_backend(params, user_id)
            
            # STEP 5: Generate response
            response_start = datetime.now()
            final_response = await self._generate_response(
                processing_query,
                analysis,
                tool_results,
                chat_history,
                memories=memories,
                detected_language=detected_language,
                original_query=original_query,
                grievance_id=grievance_id
            )
            response_time = (datetime.now() - response_start).total_seconds()
            logger.info(f"💬 Response generated in {response_time:.2f}s")
            
            # Add to memory in background
            # await self.task_queue.put(
            #     AddBackgroundTask(
            #         func=partial(self.memory.add),
            #         params=(
            #             [{"role": "user", "content": original_query}, {"role": "assistant", "content": final_response}],
            #             user_id,
            #         ),
            #     )
            # )
            
            total_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"⏱️ TOTAL: {total_time:.2f}s")
            
            return {
                "success": True,
                "response": final_response,
                "analysis": analysis,
                "tool_results": tool_results,
                "tools_used": tools_to_use,
                "grievance_id": grievance_id,
                "processing_time": {
                    "analysis": analysis_time,
                    "tools": tool_time,
                    "response": response_time,
                    "total": total_time
                }
            }
            
        except Exception as e:
            logger.error(f"❌ GrievanceAgent error: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "Maaf kijiye, kuch gadbad ho gayi. Kripya dobara koshish karein."
            }
    
    async def background_task_worker(self) -> None:
        """Background worker for memory operations"""
        while True:
            task: AddBackgroundTask = await self.task_queue.get()
            try:
                func_name = getattr(task.func, "func", task.func).__name__ if hasattr(task.func, "__name__") else repr(task.func)
                logger.info(f"Executing background task: {func_name}")
                messages, user_id = task.params
                await task.func(messages, user_id=user_id)
                logger.info(f"✅ Background task completed: {func_name}")
            except Exception as e:
                logger.error(f"❌ Background task failed: {e}")
            finally:
                self.task_queue.task_done()
    
    def _start_worker_if_needed(self):
        """Start background worker once"""
        if not self._worker_started:
            asyncio.create_task(self.background_task_worker())
            self._worker_started = True
            logger.info("🔄 Background worker started")
    
    async def _detect_and_translate(self, query: str) -> Dict[str, str]:
        """Detect language and translate to English if needed"""
        
        detection_prompt = f"""Analyze this query and identify its language, then translate if needed.

QUERY: "{query}"

YOUR TASK:
1. Identify what language this query is written in
2. Be specific with your language detection:
   - If it's Roman/Latin script with Hindi vocabulary → "hinglish"
   - If it's Devanagari script → "hindi"
   - If it's pure English → "english"
   - For other languages, identify accurately (malayalam, tamil, telugu, etc.)
   - If romanized script of any Indian language → add "_romanized" (e.g., "malayalam_romanized")

3. If the query is NOT in English, translate it to English while preserving the exact meaning and intent
4. If already in English, keep it as is

Return ONLY valid JSON:
{{
  "detected_language": "<language name or language_romanized>",
  "english_translation": "<English version or original if already English>"
}}

Examples:
- "kya kiya aaj?" → {{"detected_language": "hinglish", "english_translation": "what did you do today?"}}
- "what's the weather?" → {{"detected_language": "english", "english_translation": "what's the weather?"}}
- "क्या हाल है?" → {{"detected_language": "hindi", "english_translation": "how are you?"}}
"""
        
        try:
            response = await self.language_detector_llm.generate(
                messages=[{"role": "user", "content": detection_prompt}],
                system_prompt="You are a language detection expert. Analyze queries and return JSON only.",
                temperature=0.1,
                max_tokens=200
            )
            
            json_str = self._extract_json(response)
            result = json.loads(json_str)
            
            return {
                "detected_language": result.get('detected_language', 'english'),
                "english_translation": result.get('english_translation', query),
                "original_query": query
            }
            
        except Exception as e:
            logger.error(f"❌ Language detection failed: {e}")
            return {
                "detected_language": "english",
                "english_translation": query,
                "original_query": query
            }
    
    async def _grievance_analysis(self, query: str, chat_history: List[Dict] = None, memories: str = "") -> Dict[str, Any]:
        """Analyze query for DM office grievance handling with multi-dimensional reasoning"""
        
        context = chat_history[-4:] if chat_history else []
        current_date = datetime.now().strftime("%B %d, %Y")
        
        analysis_prompt = f"""You are analyzing queries for a grievance system.
    This system helps citizens register and track complaints with government departments.

    DATE: {current_date}

    USER'S QUERY: "{query}"

    CONVERSATION HISTORY: {context}

    Available tools:
    - rag: Knowledge base retrieval (government policies, procedures, schemes, previous info)
    - grievance: Citizen complaints registration to DM office
    Use for: complaint registration, shikayat, grievance reporting, samasyaa darj karna
    The tool extracts category, location, description and asks for clarification if needed

    YOUR TASK: Deeply analyze the query and select appropriate tools.

    STEP 1: MULTI-TASK DETECTION
    Think naturally: Does this query ask for ONE thing or MULTIPLE things?

    Examples:
    - "Water supply problem in Gomti Nagar" → 1 task (just grievance)
    - "Water supply problem in Gomti Nagar and tell me about water schemes" → 2 tasks (grievance + info)
    - "Complaint about PDS ration, also what's the complaint status process" → 2 tasks (grievance + process info)

    If multiple tasks detected:
    - Set multi_task_detected = true
    - List each task clearly in sub_tasks array

    STEP 2: UNDERSTAND OVERALL INTENT
    What does the user ultimately want to achieve?
    - If filing grievance + asking about process → "User wants to file complaint and understand the system"
    - If asking about multiple departments → "User needs information about various government services"

    Synthesize the big picture, don't just repeat the query.

    STEP 3: MULTI-DIMENSIONAL TOOL SELECTION

    Think about EACH sub-task independently:

    For GRIEVANCE tasks:
    - Select `grievance` when user wants to register complaint
    - Keywords: complaint, shikayat, grievance, problem, issue, report, darj karna, pareshani

    GRIEVANCE CATEGORIES (use ONE grievance per CATEGORY, not per issue):
    - PUBLIC_WORKS: Roads, water supply, drains, sewage, street lights, bridges, footpaths
    - POLICE: Theft, crime, FIR, law & order, police inaction, missing person
    - ELECTRICITY: Power supply, power cut, electric connection, transformer, meter
    - REVENUE: Land records, property, tehsil, patwari, registry, mutation
    - HEALTH: Hospital, PHC, medicines, ambulance, sanitation, epidemic
    - EDUCATION: School, college, teacher, scholarship, MDM
    - PDS: Ration card, ration shop, kerosene, food supply
    - MUNICIPAL: Garbage, cleanliness, encroachment, building permission
    
    MULTI-GRIEVANCE RULE:
    - If ALL issues belong to SAME category → use ONE grievance tool (combine issues)
    - If issues belong to DIFFERENT categories → use SEPARATE grievance tools
    
    Examples:
    - "No water + broken road" → SAME category (PUBLIC_WORKS) → 1 grievance
    - "No water + theft" → DIFFERENT categories (PUBLIC_WORKS + POLICE) → 2 grievances
    - "Power cut + transformer issue" → SAME category (ELECTRICITY) → 1 grievance
    - "Ration not received + police not helping" → DIFFERENT (PDS + POLICE) → 2 grievances

    For INFORMATION tasks:
    - Select `rag` for EACH distinct information need
    - Don't assume one rag search covers everything
    - If multiple info needs, add rag multiple times to tools_to_use list

    CRITICAL THINKING QUESTION:
    "If I only do 1 rag search for this query, will I get complete information?"
    - If NO → create multiple rag searches, one per information dimension
    - User asks about schemes AND process → needs 2 separate rag searches
    - User asks about policy in education AND health → needs 2 separate rag searches

    TOOL NAMING:
    - Use simple names: rag, grievance (no numbers/indexes)
    - If multiple rag searches needed, add "rag" multiple times in tools_to_use list
    - Order in the list = execution order for sequential mode
    - Example: ["rag", "rag", "grievance"] means 2 rag searches then 1 grievance

    For NO tools:
    - Greetings, casual conversation
    - Simple thank you or acknowledgment
    - Questions answerable from conversation context

    STEP 4: EXECUTION PLANNING

    Default to PARALLEL execution (tools run simultaneously)
    - Most grievance queries have independent information needs
    - Example: Filing complaint + asking about schemes → parallel

    Use SEQUENTIAL only if:
    - One tool NEEDS results from another tool to work
    - Example: "Check my complaint status then tell me next steps" → sequential (rag first, then use that info)
    
    For SEQUENTIAL: tools_to_use order = execution order. First tool runs first, etc.

    STEP 5: QUERY OPTIMIZATION

    For each selected tool, write a COMPLETE, SELF-CONTAINED query:

    RAG queries:
    - Specific, targeted to that information need
    - Include key concepts from the query
    - CRITICAL FOR FOLLOW-UP QUESTIONS: If user's query is vague/incomplete (like "what documents?", "how to apply?", "tell me more"), 
      use CONVERSATION HISTORY to understand WHAT TOPIC they're asking about
    - Add the topic/subject from conversation context to make query complete
    
    RAG Examples:
    Previous conversation about PMMVY scheme, then user asks "what documents needed?"
    → RAG query: "Pradhan Mantri Matru Vandana Yojana PMMVY required documents"
    NOT: "required documents DM office complaint" (WRONG - ignores conversation topic!)
    
    Previous conversation about electricity complaint, then user asks "how to track status?"
    → RAG query: "electricity complaint status tracking process"

    Grievance queries - CRITICAL RULES:
    - Each grievance query must be COMPLETE and SELF-CONTAINED
    - Include ALL shared context in EVERY grievance query:
      * User's name (if mentioned anywhere in original query)
      * Contact info - phone/email (if mentioned anywhere)
      * Common location (applies to all grievances from same area)
      * Any identity/address info that applies to multiple complaints
    - Include the specific issue details for that grievance
    - Do NOT split shared info - DUPLICATE it in each query
    - Do NOT assume or guess missing details

    CORRECT Example of multi-grievance splitting:
    User: "My name is Rahul, phone 9876543210. I live in Gomti Nagar. No water for 2 days and police not taking action on theft"
    → Water = PUBLIC_WORKS, Police inaction = POLICE (DIFFERENT categories)
    tools_to_use: ["grievance", "grievance"]
    enhanced_queries: [
      "My name is Rahul, phone 9876543210, Gomti Nagar. No water supply for 2 days",
      "My name is Rahul, phone 9876543210, Gomti Nagar. Police not taking action on theft"
    ]

    CORRECT Example - SAME category, ONE grievance:
    User: "My name is Rahul, phone 9876543210, Gomti Nagar. No water for 2 days and road is also broken"
    → Water + Road = BOTH PUBLIC_WORKS (SAME category)
    tools_to_use: ["grievance"]
    enhanced_queries: [
      "My name is Rahul, phone 9876543210, Gomti Nagar. No water supply for 2 days and road is broken"
    ]
    ↑ Combined into ONE grievance because same department handles both.

    WRONG Example (DO NOT DO THIS):
    User: "No water and road broken in Gomti Nagar"
    tools_to_use: ["grievance", "grievance"]  ← WRONG! Same category, should be 1 grievance
    enhanced_queries: ["No water in Gomti Nagar", "Road broken in Gomti Nagar"]

    Single grievance example:
    User: "Mera naam Priya hai, 8765432109. Bijli nahi aa rahi 3 din se, Lalbagh mein"
    tools_to_use: ["grievance"]
    enhanced_queries: ["Mera naam Priya hai, 8765432109, Lalbagh. Bijli nahi aa rahi 3 din se"]

    STEP 6: ANALYZE SENTIMENT & STRATEGY

    User's emotional state:
    - frustrated (high/medium/low)
    - urgent (high/medium/low)
    - worried (high/medium/low)
    - calm
    - confused

    Response approach:
    - empathetic (for frustrated users)
    - reassuring (for worried users)
    - helpful (for confused users)
    - informative (for calm queries)

    Return ONLY valid JSON:
    {{
    "multi_task_analysis": {{
        "multi_task_detected": true or false,
        "sub_tasks": ["task 1 description", "task 2 description"]
    }},
    "semantic_intent": "overall user goal synthesized from all sub-tasks",
    "is_grievance_related": true or false,
    "tools_to_use": ["tool1", "tool2", "tool3"],
    "execution_mode": "parallel or sequential",
    "tool_execution": {{
        "mode": "parallel or sequential",
        "dependency_reason": "reason if sequential, empty if parallel"
    }},
    "enhanced_queries": ["query for first tool", "query for second tool", "...one query per tool in order"],
    "tool_reasoning": "why these tools selected and why this many",
    "sentiment": {{
        "primary_emotion": "frustrated|worried|urgent|calm|confused",
        "intensity": "low|medium|high"
    }},
    "response_strategy": {{
        "tone": "empathetic|helpful|informative|reassuring",
        "length": "short|medium|detailed"
    }},
    "key_points": ["point1", "point2"]
    }}

    FINAL CHECK:
    - Did I identify ALL distinct tasks in the query?
    - Did I create separate tool calls for each information dimension?
    - Are my queries focused and specific?
    - Did I synthesize a clear semantic_intent?
    - CRITICAL: Did I include name/contact/location in EACH grievance query? (Don't lose shared context!)

    Now analyze the query above."""

        try:
            messages = chat_history[-4:] if chat_history else []
            messages.append({"role": "user", "content": analysis_prompt})
            
            response = await self.llm.generate(
                messages,
                system_prompt=f"You analyze queries for DM office grievance system. Date: {current_date}. Return valid JSON only.",
                temperature=0.1,
                max_tokens=3000  # Increased for multi-dimensional analysis
            )
            
            json_str = self._extract_json(response)
            result = json.loads(json_str)
            
            logger.info(f"✅ Analysis complete: {result.get('semantic_intent', 'N/A')[:100]}")
            logger.info(f"   Multi-task: {result.get('multi_task_analysis', {}).get('multi_task_detected', False)}")
            logger.info(f"   Sub-tasks: {result.get('multi_task_analysis', {}).get('sub_tasks', [])}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Analysis JSON parse error: {e}")
            return self._get_fallback_analysis(query)
    
    def _get_fallback_analysis(self, query: str) -> Dict[str, Any]:
        """Fallback analysis when parsing fails"""
        return {
            "semantic_intent": query,
            "is_grievance_related": False,
            "tools_to_use": [],
            "execution_mode": "parallel",
            "enhanced_queries": [],
            "tool_reasoning": "Fallback - direct response",
            "sentiment": {"primary_emotion": "calm", "intensity": "medium"},
            "response_strategy": {"tone": "helpful", "length": "medium"},
            "key_points": []
        }
    
    async def _execute_tools(self, tools: List[str], query: str, analysis: Dict, user_id: str = None) -> Dict[str, Any]:
        """Execute tools - simplified for RAG and Grievance only
        
        tools_to_use and enhanced_queries are both lists in same order.
        Example: tools=["rag", "rag", "grievance"], queries=["query1", "query2", "query3"]
        """
        
        if not tools:
            return {}
        
        results = {}
        enhanced_queries = analysis.get('enhanced_queries', [])
        execution_mode = analysis.get('execution_mode', 'parallel')
        
        # Handle both list and dict format for backward compatibility
        if isinstance(enhanced_queries, dict):
            enhanced_queries = list(enhanced_queries.values())
        
        # Filter tools and keep corresponding queries aligned
        valid_tools_with_queries = []
        for i, tool in enumerate(tools):
            if tool in self.available_tools:
                tool_query = enhanced_queries[i] if i < len(enhanced_queries) else query
                valid_tools_with_queries.append((tool, tool_query))
        
        if not valid_tools_with_queries:
            return {}
        
        logger.info(f"🔧 Executing {len(valid_tools_with_queries)} tools (mode: {execution_mode})")
        
        if execution_mode == 'sequential':
            # Execute in order - list order is execution order
            for i, (tool, tool_query) in enumerate(valid_tools_with_queries):
                logger.info(f"   → [{i}] {tool}: '{tool_query[:80]}...'")
                
                try:
                    result = await self.tool_manager.execute_tool(
                        tool, 
                        query=tool_query, 
                        user_id=user_id
                    )
                    # Use index to allow multiple same-tool results
                    result_key = f"{tool}_{i}" if valid_tools_with_queries.count((tool, tool_query)) > 1 or sum(1 for t, _ in valid_tools_with_queries if t == tool) > 1 else tool
                    results[result_key] = result
                    logger.info(f"   ✅ {tool} completed")
                except Exception as e:
                    logger.error(f"   ❌ {tool} failed: {e}")
                    results[f"{tool}_{i}"] = {"error": str(e)}
        else:
            # Execute in parallel - order doesn't matter
            tasks = []
            for i, (tool, tool_query) in enumerate(valid_tools_with_queries):
                logger.info(f"   → [{i}] {tool}: '{tool_query[:80]}...'")
                
                task = self.tool_manager.execute_tool(
                    tool,
                    query=tool_query,
                    user_id=user_id
                )
                tasks.append((i, tool, task))
            
            for i, tool_name, task in tasks:
                try:
                    result = await task
                    # Use index to allow multiple same-tool results
                    result_key = f"{tool_name}_{i}" if sum(1 for _, t, _ in tasks if t == tool_name) > 1 else tool_name
                    results[result_key] = result
                    logger.info(f"   ✅ {tool_name} completed")
                except Exception as e:
                    logger.error(f"   ❌ {tool_name} failed: {e}")
                    results[tool_name] = {"error": str(e)}
        
        return results

    async def _resolve_ward_from_location(self, address: str, city: str) -> Optional[str]:
        """Resolve ward number from location using RAG tool"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {"address": address, "city": city}
                async with session.post(self.resolve_ward_url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        ward = data.get("ward_name")
                        logger.info(f"✅ Resolved ward: {ward} for location: {address}")
                        return ward
                    else:
                        logger.error(f"❌ Ward resolution failed: {response.status}")
            return None
        except Exception as e:
            logger.error(f"❌ Ward resolution failed: {e}")
            return None
       
    async def _post_grievance_to_backend(self, params: dict, user_id: str) -> Optional[str]:
        """Post successful grievance data to backend and return grievance ID if available"""
        if not self.backend_url:
            logger.warning("⚠️ GRIEVANCE_BACKEND_URL not set, skipping backend posting")
            return None
        
                
        logger.info(f"params before posting: {params}")
        location = params.get("location", "")
        city = location.get("city", "")
        district, ward = location.get("district", ""), location.get("ward", "")
        if district and ward:
            address = f"{ward}, {district}"
            ward = await self._resolve_ward_from_location(address, city)
            params['location']['ward'] = ward
  
        try:
            payload = {
                "user_id": user_id,
                **params
            }
            logger.info(f"📤 Posting grievance to backend: {self.backend_url}")
            
            logger.info(f"   Payload: {payload}")
            async with aiohttp.ClientSession() as session:
                async with session.post(self.backend_url, json=payload) as response:
                    if response.status in [200, 201]:
                        response_data = await response.json()
                        logger.info("✅ Grievance posted to backend successfully")
                        logger.info(f"   Backend Response: {response_data}")
                        
                        # Extract and log grievance ID
                        grievance_id = response_data.get('_id')
                        if grievance_id:
                            logger.info(f"📋 Generated Grievance ID: {grievance_id}")
                            return grievance_id
                        else:
                            logger.warning("⚠️ No 'grievanceId' found in backend response")
                            return None
                    else:
                        response_text = await response.text()
                        logger.error(f"❌ Failed to post grievance: {response.status} {response_text}")
                        logger.error(f"   Payload sent: {payload}")
                        return None
        except Exception as e:
            logger.error(f"❌ Error posting grievance to backend: {e}")
            logger.error(f"   Payload was: {payload}")
            return None
    
    async def _generate_response(self, query: str, analysis: Dict, tool_results: Dict, 
                                  chat_history: List[Dict], memories: str = "",
                                  detected_language: str = "english", 
                                  original_query: str = None, grievance_id: Optional[str] = None) -> str:
        """Generate response for DM office context"""
        
        if original_query is None:
            original_query = query
        
        intent = analysis.get('semantic_intent', '')
        sentiment = analysis.get('sentiment', {})
        strategy = analysis.get('response_strategy', {})
        is_grievance = analysis.get('is_grievance_related', False)
        
        # Format tool results
        tool_data = self._format_tool_results(tool_results)
        
        context = chat_history[-4:] if chat_history else []
        
        response_prompt = f"""You are a helpful assistant for the District Magistrate (DM) Office grievance system.

Your role is to:
- Help citizens register and track grievances
- Provide information about government services and policies
- Guide users through the complaint process
- Be empathetic and supportive

CRITICAL - LANGUAGE RULE:
User's detected language: {detected_language}
You MUST respond ONLY in {detected_language}. Match the exact script:
- If hinglish → use Roman letters: "aapka", "hai", "kripya", "shikayat"
- If hindi → use Devanagari: क, ख, ग, आपका, है
- If english → use English only
- If tamil/telugu/other → use that language's script

This language rule overrides everything else.

UNDERSTOOD INTENT: {intent}

TOOL DATA:
{tool_data}

GRIEVANCE ID: {grievance_id if grievance_id else "Not available"}

CONVERSATION CONTEXT:
- User Emotion: {sentiment.get('primary_emotion', 'calm')} ({sentiment.get('intensity', 'medium')})
- Grievance Related: {is_grievance}

RESPONSE GUIDELINES:

FOR GRIEVANCE REGISTRATION:
- If grievance tool SUCCESS:
  * Confirm complaint is registered
  * Mention the category and location extracted
  * Provide reassurance that action will be taken
  * Include the GRIEVANCE ID in the response if available
  * DO NOT ask follow-up questions after successful registration

- If grievance tool NEEDS CLARIFICATION:
  * Use the clarification message from the tool
  * Ask politely for missing information
  * Be patient

FOR INFORMATION QUERIES (RAG):
- Provide clear, accurate information from knowledge base
- If info not available, say so honestly

FOR GENERAL QUERIES:
- Be warm and welcoming
- Guide users on how to file complaints if appropriate

TONE EXAMPLES:
- Empathetic: "Hum samajh sakte hain aapki pareshani..."
- Reassuring: "Aapki shikayat darj ho gayi hai, karyawahi ki jayegi..."
- Helpful: "Main aapki madad kar sakta hoon..."

CRITICAL RULES:
1. NEVER repeat or restate what the user said
2. Start directly with your response - no preamble
3. NEVER say "Let me check..." or "I found..."
4. Match user's emotional energy
5. Keep response focused and actionable
6. Respond in {detected_language} ONLY

Now respond in {detected_language}:"""

        try:
            max_tokens = {
                "short": 300,
                "medium": 500,
                "detailed": 700
            }.get(strategy.get('length', 'medium'), 500)
            
            messages = chat_history[-4:] if chat_history else []
            messages.append({"role": "user", "content": response_prompt})
            
            language = detected_language.lower()
            
            # Use appropriate LLM based on language
            if language in SARVAM_SUPPORTED_LANGUAGES:
                response = await self.indic_llm.generate(
                    messages,
                    temperature=0.4,
                    max_tokens=max_tokens,
                    system_prompt=f"Respond in {detected_language} only. Use the same script as the user."
                )
            else:
                response = await self.llm.generate(
                    messages,
                    temperature=0.4,
                    max_tokens=max_tokens,
                    system_prompt=f"Respond in {detected_language} only. Use the same script as the user."
                )
            
            response = self._clean_response(response)
            logger.info(f"💬 Generated Response: {response}")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Response generation failed: {e}")
            return "Maaf kijiye, response generate karne mein samasya aayi. Kripya dobara koshish karein."
    
    def _format_tool_results(self, tool_results: Dict) -> str:
        """Format tool results for response generation"""
        if not tool_results:
            return "No tool data available."
        
        formatted = []
        
        for tool_name, result in tool_results.items():
            # Handle both 'rag' and 'rag_0', 'rag_1' etc.
            base_tool = tool_name.split('_')[0] if '_' in tool_name and tool_name.split('_')[-1].isdigit() else tool_name
            
            if base_tool == 'rag':
                if isinstance(result, dict) and result.get('success'):
                    formatted.append(f"RAG RESULTS:\n{result.get('retrieved', 'No data')}")
                elif isinstance(result, dict) and result.get('error'):
                    formatted.append(f"RAG: Error - {result.get('error')}")
            
            elif base_tool == 'grievance':
                if isinstance(result, dict):
                    if result.get('success'):
                        params = result.get('params', {})
                        if hasattr(params, 'to_dict'):
                            params = params.to_dict()
                        formatted.append(f"GRIEVANCE REGISTERED:\n- Category: {params.get('category', 'N/A')}\n- Location: {params.get('location', 'N/A')}\n- Description: {params.get('description', 'N/A')}\n- Priority: {params.get('priority', 'N/A')}")
                    elif result.get('needs_clarification'):
                        formatted.append(f"GRIEVANCE NEEDS CLARIFICATION:\n{result.get('clarification_message', 'Please provide more details.')}\nMissing: {result.get('missing_fields', [])}")
                    elif result.get('error'):
                        formatted.append(f"GRIEVANCE ERROR: {result.get('error')}")
        
        return "\n\n".join(formatted) if formatted else "No tool data available."
    
    def _extract_json(self, response: str) -> str:
        """Extract JSON from LLM response"""
        response = response.strip()
        
        # Remove thinking tags if present
        if '<think>' in response:
            end_idx = response.find('</think>')
            if end_idx != -1:
                response = response[end_idx + 8:].strip()
        
        # Remove markdown code blocks
        if response.startswith('```'):
            lines = response.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            response = '\n'.join(lines)
        
        # Find JSON boundaries
        json_start = response.find('{')
        json_end = response.rfind('}')
        
        if json_start != -1 and json_end != -1 and json_end > json_start:
            return response[json_start:json_end+1]
        
        return response
    
    def _clean_response(self, response: str) -> str:
        """Clean final response for display"""
        response = response.strip()
        
        # Remove thinking tags
        if '<think>' in response:
            end_idx = response.find('</think>')
            if end_idx != -1:
                response = response[end_idx + 8:].strip()
        
        # Remove markdown blocks
        if response.startswith('```'):
            lines = response.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            response = '\n'.join(lines)
        
        return response.strip()
