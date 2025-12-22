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
from mem0 import AsyncMemory
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
        self.memory = AsyncMemory(memory_config)
        self.task_queue: asyncio.Queue["AddBackgroundTask"] = asyncio.Queue()
        self._worker_started = False
        
        # Check tool availability
        self._grievance_available = tool_manager.grievance_available
        
        # Backend URL for posting grievances
        self.backend_url = getenv("GRIEVANCE_BACKEND_URL")
        
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
            memory_results = await self.memory.search(processing_query[:100], user_id=user_id, limit=5)
            memories = "\n".join([
                f"- {item['memory']}" 
                for item in memory_results.get("results", []) 
                if item.get("memory")
            ]) or "No previous context."
            logger.info(f"📝 Retrieved memories: {len(memories)} chars")
            
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
            if ("grievance" in tool_results and 
                tool_results["grievance"].get("success") and 
                not tool_results["grievance"].get("needs_clarification")):
                params = tool_results["grievance"].get("params")
                if params:
                    await self._post_grievance_to_backend(params, user_id)
            
            # STEP 5: Generate response
            response_start = datetime.now()
            final_response = await self._generate_response(
                processing_query,
                analysis,
                tool_results,
                chat_history,
                memories=memories,
                detected_language=detected_language,
                original_query=original_query
            )
            response_time = (datetime.now() - response_start).total_seconds()
            logger.info(f"💬 Response generated in {response_time:.2f}s")
            
            # Add to memory in background
            await self.task_queue.put(
                AddBackgroundTask(
                    func=partial(self.memory.add),
                    params=(
                        [{"role": "user", "content": original_query}, {"role": "assistant", "content": final_response}],
                        user_id,
                    ),
                )
            )
            
            total_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"⏱️ TOTAL: {total_time:.2f}s")
            
            return {
                "success": True,
                "response": final_response,
                "analysis": analysis,
                "tool_results": tool_results,
                "tools_used": tools_to_use,
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
        """Analyze query for DM office grievance handling"""
        
        context = chat_history[-4:] if chat_history else []
        current_date = datetime.now().strftime("%B %d, %Y")
        
        analysis_prompt = f"""You are analyzing queries for a District Magistrate (DM) Office grievance system.
This system helps citizens register and track complaints with government departments.

DATE: {current_date}

USER'S QUERY: "{query}"

CONVERSATION HISTORY: {context}

LONG-TERM CONTEXT (Memories): {memories}

Available tools:
- rag: Knowledge base retrieval (government policies, procedures, schemes, previous info)
- grievance: Citizen complaints registration to DM office
  Use for: complaint registration, shikayat, grievance reporting, samasyaa darj karna
  The tool extracts category, location, description and asks for clarification if needed

YOUR TASK: Analyze the query and select appropriate tools.

TOOL SELECTION RULES:

1. SELECT `grievance` when user:
   - Wants to register a complaint (shikayat, samasyaa, problem report)
   - Reports an issue with government services
   - Mentions department problems (PDS, Revenue, Police, Health, Education, Public Works, Water, Electricity)
   - Uses words like: complaint, shikayat, grievance, problem, issue, report, darj karna, pareshani

2. SELECT `rag` when user:
   - Asks about government policies or procedures
   - Wants information about schemes or services
   - Asks about complaint status or tracking
   - Needs reference information

3. SELECT BOTH when:
   - Registering a complaint that might need policy context
   
4. SELECT NO tools when:
   - Greetings, casual conversation
   - Simple thank you or acknowledgment
   - Questions answerable from conversation context

EXECUTION ORDER:
- If both tools needed, list them in order of execution (e.g., ["rag", "grievance"] means rag first, then grievance)
- Tools will be executed in the order you provide

IMPORTANT FOR GRIEVANCE TOOL:
- Provide the user's complaint in natural language as the query
- Include all details the user mentioned (location, department, issue)
- Do NOT add any information the user did not provide
- Do NOT assume or guess missing details

Return ONLY valid JSON:
{{
  "semantic_intent": "what user wants to achieve",
  "is_grievance_related": true or false,
  "tools_to_use": ["tool1", "tool2"],
  "execution_mode": "parallel or sequential",
  "enhanced_queries": {{
    "rag": "query for knowledge base if rag selected",
    "grievance": "the user's complaint in natural language if grievance selected"
  }},
  "tool_reasoning": "why these tools selected",
  "sentiment": {{
    "primary_emotion": "frustrated|worried|urgent|calm|confused",
    "intensity": "low|medium|high"
  }},
  "response_strategy": {{
    "tone": "empathetic|helpful|informative|reassuring",
    "length": "short|medium|detailed"
  }},
  "key_points": ["point1", "point2"]
}}"""
        
        try:
            messages = chat_history[-4:] if chat_history else []
            messages.append({"role": "user", "content": analysis_prompt})
            
            response = await self.llm.generate(
                messages,
                system_prompt=f"You analyze queries for DM office grievance system. Date: {current_date}. Return valid JSON only.",
                temperature=0.1,
                max_tokens=2000
            )
            
            json_str = self._extract_json(response)
            result = json.loads(json_str)
            
            logger.info(f"✅ Analysis complete: {result.get('semantic_intent', 'N/A')[:100]}")
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
            "enhanced_queries": {},
            "tool_reasoning": "Fallback - direct response",
            "sentiment": {"primary_emotion": "calm", "intensity": "medium"},
            "response_strategy": {"tone": "helpful", "length": "medium"},
            "key_points": []
        }
    
    async def _execute_tools(self, tools: List[str], query: str, analysis: Dict, user_id: str = None) -> Dict[str, Any]:
        """Execute tools - simplified for RAG and Grievance only"""
        
        if not tools:
            return {}
        
        results = {}
        enhanced_queries = analysis.get('enhanced_queries', {})
        execution_mode = analysis.get('execution_mode', 'parallel')
        
        # Filter to only available tools
        valid_tools = [t for t in tools if t in self.available_tools]
        
        if not valid_tools:
            return {}
        
        logger.info(f"🔧 Executing tools: {valid_tools} (mode: {execution_mode})")
        
        if execution_mode == 'sequential':
            # Execute in order provided by LLM
            for tool in valid_tools:
                tool_query = enhanced_queries.get(tool, query)
                logger.info(f"   → {tool}: '{tool_query[:80]}...'")
                
                try:
                    result = await self.tool_manager.execute_tool(
                        tool, 
                        query=tool_query, 
                        user_id=user_id
                    )
                    results[tool] = result
                    logger.info(f"   ✅ {tool} completed")
                except Exception as e:
                    logger.error(f"   ❌ {tool} failed: {e}")
                    results[tool] = {"error": str(e)}
        else:
            # Execute in parallel
            tasks = []
            for tool in valid_tools:
                tool_query = enhanced_queries.get(tool, query)
                logger.info(f"   → {tool}: '{tool_query[:80]}...'")
                
                task = self.tool_manager.execute_tool(
                    tool,
                    query=tool_query,
                    user_id=user_id
                )
                tasks.append((tool, task))
            
            for tool_name, task in tasks:
                try:
                    result = await task
                    results[tool_name] = result
                    logger.info(f"   ✅ {tool_name} completed")
                except Exception as e:
                    logger.error(f"   ❌ {tool_name} failed: {e}")
                    results[tool_name] = {"error": str(e)}
        
        return results
    
    async def _post_grievance_to_backend(self, params: dict, user_id: str):
        """Post successful grievance data to backend"""
        if not self.backend_url:
            logger.warning("⚠️ GRIEVANCE_BACKEND_URL not set, skipping backend posting")
            return
            
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
                        logger.info("✅ Grievance posted to backend successfully")
                    else:
                        response_text = await response.text()
                        logger.error(f"❌ Failed to post grievance: {response.status} {response_text}")
                        logger.error(f"   Payload sent: {payload}")
        except Exception as e:
            logger.error(f"❌ Error posting grievance to backend: {e}")
            logger.error(f"   Payload was: {payload}")
    
    async def _generate_response(self, query: str, analysis: Dict, tool_results: Dict, 
                                  chat_history: List[Dict], memories: str = "",
                                  detected_language: str = "english", 
                                  original_query: str = None) -> str:
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

CONVERSATION CONTEXT:
- User Emotion: {sentiment.get('primary_emotion', 'calm')} ({sentiment.get('intensity', 'medium')})
- Grievance Related: {is_grievance}

MEMORIES (if relevant): {memories}

RESPONSE GUIDELINES:

FOR GRIEVANCE REGISTRATION:
- If grievance tool SUCCESS:
  * Confirm complaint is registered
  * Mention the category and location extracted
  * Provide reassurance that action will be taken
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
            logger.info(f"💬 Response: {len(response)} chars")
            
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
            if tool_name == 'rag':
                if isinstance(result, dict) and result.get('success'):
                    formatted.append(f"RAG RESULTS:\n{result.get('retrieved', 'No data')}")
                elif isinstance(result, dict) and result.get('error'):
                    formatted.append(f"RAG: Error - {result.get('error')}")
            
            elif tool_name == 'grievance':
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
