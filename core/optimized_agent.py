"""
Optimized Single-Pass Agent System
Combines semantic analysis, tool execution, and response generation in minimal LLM calls
Now supports sequential tool execution with middleware for dependent tools
WITH REDIS CACHING for queries and formatted tool data
WITH SCRAPING CONFIRMATION FLOW for user consent on intensive scraping
"""

import json
import logging
import asyncio
import uuid
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
from os import getenv
from mem0 import AsyncMemory
from functools import partial
from mem0.configs.base import MemoryConfig
from .config import (
    AddBackgroundTask, 
    RedisCacheManager,
    ENABLE_SCRAPING_CONFIRMATION,
    SCRAPING_CONFIRMATION_THRESHOLD,
    ESTIMATED_TIME_PER_PAGE,
    ENABLE_SUPERSEDE_ON_NEW_QUERY,
    ENABLE_LLM_CONFIRMATION_DETECTION,
    LLM_CONFIRMATION_CONFIDENCE_THRESHOLD,
    ENABLE_CONFIRMATION_REGEX_FALLBACK
)

config = MemoryConfig(
    graph_store={
        "provider": "neo4j",
        "config": {
            "url": getenv('NEO4J_URL'),
            "username": getenv('NEO4J_USER'),
            "password": getenv('NEO4J_PASSWORD')
        }
    },
    vector_store={
        "provider": "chroma",
        "config": {
            "collection_name": "mem0_collection",
            "path": ".chromadb"
        }
    }
)

logger = logging.getLogger(__name__)

# SYSTEM_PROMPT="""You are an expert analyst. Analyze queries using multi-signal intelligence covering semantics,
# business opportunities, tool needs, and communication strategy. 

# Your analysis must also demonstrate contextual intelligence — infer user intent, constraints, and emotional tone,
# even if not explicitly stated.

# Expand reasoning scope intelligently:
# - Identify the user’s true problem or need beyond the literal query.
# - Infer constraints such as budget, environment, urgency, or effort tolerance.
# - Suggest adjacent or alternative solution paths when relevant. 
#   (Example: If a user asks for "best AC on a tight budget", include air coolers or energy-efficient models.)
# - When presenting solutions, prioritize durability, accessibility, and practicality.
# - Briefly explain the reasoning or trade-off behind your choices, maintaining a natural and helpful tone.

# Behavioral principles:
# 1. Curate, not list — provide reasoned, context-fitting answers rather than raw data.
# 2. Maintain an empathetic, peer-like tone — guide the user thoughtfully, not mechanically.
# 3. Adapt dynamically — balance between precision and user-centered flexibility.
# 4. Never invent irrelevant information, but always consider meaningful adjacent options.
# 5. Your reasoning framework:
#    - What is the user's real goal or constraint?
#    - What related domains or solutions could address it better?
#    - How do I present this insight clearly, within structured output?

# Output Format:
# - Return valid JSON only.

# Your mission: combine analytical precision with contextual empathy — deliver responses that are not only correct,
# but *relevant, human, and insightful*.
# """



class OptimizedAgent:
    """Single-pass agent that minimizes LLM calls while maintaining all functionality"""
    
    def __init__(self, brain_llm, heart_llm, tool_manager):
        self.brain_llm = brain_llm
        self.heart_llm = heart_llm
        self.tool_manager = tool_manager
        self.available_tools = tool_manager.get_available_tools()
        self.memory = AsyncMemory(config)
        self.task_queue: asyncio.Queue["AddBackgroundTask"] = asyncio.Queue()
        self._worker_started = False
        
        # Initialize Redis cache manager
        self.cache_manager = RedisCacheManager()
        
        logger.info(f"OptimizedAgent initialized with tools: {self.available_tools}")
        logger.info(f"Redis caching: {'ENABLED ✅' if self.cache_manager.enabled else 'DISABLED ⚠️'}")
        logger.info(f"Scraping confirmation: {'ENABLED ✅' if ENABLE_SCRAPING_CONFIRMATION else 'DISABLED ⚠️'}")
    
    def _is_confirmation_reply(self, query: str) -> Optional[str]:
        """
        Detect if query is a yes/no confirmation reply
        
        Returns:
            'yes', 'no', or None if not a confirmation
        """
        query_lower = query.lower().strip()
        
        # Yes patterns
        yes_patterns = [
            r'^yes$', r'^yeah$', r'^yep$', r'^yup$', r'^sure$', 
            r'^ok$', r'^okay$', r'^continue$', r'^proceed$', r'^go ahead$',
            r'^y$', r'^hai$', r'^han$', r'^haan$'  # Hindi: yes
        ]
        
        # No patterns
        no_patterns = [
            r'^no$', r'^nah$', r'^nope$', r'^cancel$', r'^stop$',
            r'^n$', r'^nahi$', r'^nai$', r'^mat karo$'  # Hindi: no
        ]
        
        for pattern in yes_patterns:
            if re.match(pattern, query_lower):
                return 'yes'
        
        for pattern in no_patterns:
            if re.match(pattern, query_lower):
                return 'no'
        
        return None
    
    def _estimate_scraping_time(self, scraping_guidance: Dict) -> int:
        """
        Estimate total scraping time in seconds
        
        Args:
            scraping_guidance: Dict of tool -> {scraping_count, ...}
        
        Returns:
            Estimated time in seconds
        """
        total_pages = 0
        for tool_key, guidance in scraping_guidance.items():
            if 'web_search' in tool_key:
                total_pages += guidance.get('scraping_count', 0)
        
        # Add base time for search + LLM processing
        base_time = 5
        scraping_time = total_pages * ESTIMATED_TIME_PER_PAGE
        
        return base_time + scraping_time
    
    def _needs_confirmation(self, scraping_guidance: Dict) -> bool:
        """
        Check if any scraping guidance requires user confirmation
        
        Args:
            scraping_guidance: Dict of tool -> {scraping_count, ...}
        
        Returns:
            True if confirmation needed
        """
        if not ENABLE_SCRAPING_CONFIRMATION:
            return False
        
        for tool_key, guidance in scraping_guidance.items():
            if 'web_search' in tool_key:
                scraping_count = guidance.get('scraping_count', 0)
                if scraping_count >= SCRAPING_CONFIRMATION_THRESHOLD:
                    return True
        
        return False
    
    async def process_query(self, query: str, chat_history: List[Dict] = None, user_id: str = None) -> Dict[str, Any]:
        """Process query with minimal LLM calls, Redis caching, and scraping confirmation"""
        self._start_worker_if_needed()
        logger.info(f" PROCESSING QUERY: '{query}'")
        start_time = datetime.now()
        logger.info(f" DEBUG CHAT HISTORY:")
        logger.info(f"   Type: {type(chat_history)}")
        logger.info(f"   Length: {len(chat_history) if chat_history else 0}")
        logger.info(f"   Content: {chat_history}")
        logger.info(f"   User ID: {user_id}")
        logger.info(f"   Is None?: {chat_history is None}")
        
        # Initialize variables that are used later in all code paths
        cached_analysis = None
        analysis = None
        analysis_time = 0.0
        
        try:
            # STEP 1: Check if pending confirmation exists for this user
            pending = None
            if user_id and ENABLE_SCRAPING_CONFIRMATION:
                pending = await self.cache_manager.get_pending_confirmation_for_user(user_id)
            
            # STEP 2: Determine if this is a confirmation reply using LLM (if pending exists)
            if pending and ENABLE_LLM_CONFIRMATION_DETECTION:
                logger.info(f"⏳ Pending confirmation found - analyzing intent with LLM")
                
                # Retrieve memories for context
                memory_results = await self.memory.search(query, user_id=user_id, limit=5)
                
                # Detailed mem0 logging
                logger.info(f"🧠 MEM0 SEARCH RESULTS (Pending Confirmation Path):")
                logger.info(f"   Query: '{query[:50]}...'")
                logger.info(f"   User ID: {user_id}")
                logger.info(f"   Raw results type: {type(memory_results)}")
                logger.info(f"   Results keys: {memory_results.keys() if isinstance(memory_results, dict) else 'N/A'}")
                logger.info(f"   Total results count: {len(memory_results.get('results', [])) if isinstance(memory_results, dict) else 0}")
                
                # Log each individual memory
                if isinstance(memory_results, dict) and 'results' in memory_results:
                    for idx, item in enumerate(memory_results.get('results', [])):
                        logger.info(f"   Memory {idx + 1}:")
                        logger.info(f"      Content: {item.get('memory', 'N/A')}")
                        logger.info(f"      Score: {item.get('score', 'N/A')}")
                        logger.info(f"      Metadata: {item.get('metadata', {})}")
                else:
                    logger.info(f"   ⚠️ No results or unexpected format")
                
                memories = "\n".join([
                    f"- {item['memory']}" 
                    for item in memory_results.get("results", []) 
                    if item.get("memory")
                ]) or "No previous context."
                
                # Call Brain LLM with pending confirmation context
                analysis = await self._comprehensive_analysis(
                    query=query,
                    chat_history=chat_history,
                    memories=memories,
                    pending_confirmation=pending
                )
                
                # Extract confirmation response from LLM
                confirmation_response = analysis.get('confirmation_response', {})
                has_pending = confirmation_response.get('has_pending', False)
                user_intent = confirmation_response.get('user_intent', 'new_query')
                confidence = confirmation_response.get('confidence', 0)
                reasoning = confirmation_response.get('reasoning', 'No reasoning provided')
                
                logger.info(f"🎯 LLM Confirmation Analysis:")
                logger.info(f"   Has Pending: {has_pending}")
                logger.info(f"   User Intent: {user_intent}")
                logger.info(f"   Confidence: {confidence}%")
                logger.info(f"   Reasoning: {reasoning}")
                
                # Route based on LLM semantic analysis
                if has_pending and user_intent == 'approve':
                    # User approved - proceed with full scraping
                    logger.info(f"✅ User approved (confidence: {confidence}%) - proceeding with full scraping")
                    token = pending.get('token')
                    await self.cache_manager.delete_pending_confirmation(token)
                    
                    return await self._resume_with_confirmation(
                        pending=pending,
                        user_decision='yes',
                        start_time=start_time
                    )
                
                elif has_pending and user_intent == 'decline':
                    # User declined - downgrade scraping
                    logger.info(f"� LLM detected decline (confidence: {confidence}%) - downgrading to minimal scraping")
                    token = pending.get('token')
                    await self.cache_manager.delete_pending_confirmation(token)
                    
                    return await self._resume_with_confirmation(
                        pending=pending,
                        user_decision='no',
                        start_time=start_time
                    )
                
                elif user_intent == 'new_query' or user_intent == 'ambiguous':
                    # User changed topic or intent unclear - cancel pending and process new query
                    logger.info(f"🔄 User intent: {user_intent} (confidence: {confidence}%) - cancelling pending and processing as new query")
                    await self.cache_manager.cancel_all_pending_confirmations_for_user(user_id)
                    # Fall through to normal processing with existing analysis
                
                # If we reached here, we have analysis from LLM - use it
                logger.info(f"📊 Using LLM analysis from confirmation check")
                analysis_time = 0.0  # Already analyzed above
                
            else:
                # No pending confirmation - proceed normally
                analysis = None
            
            # STEP 3: If no analysis yet (no pending or didn't use LLM path), check cache or analyze
            if analysis is None:
                cached_analysis = await self.cache_manager.get_cached_query(query, user_id)
                
                if cached_analysis:
                    logger.info(f"🎯 USING CACHED ANALYSIS - Skipping Brain LLM call")
                    analysis = cached_analysis
                    analysis_time = 0.0  # Cache hit = instant
                else:
                    # Retrieve memories
                    memory_results = await self.memory.search(query, user_id=user_id, limit=5)
                    
                    # Detailed mem0 logging
                    logger.info(f"🧠 MEM0 SEARCH RESULTS (Normal Query Path):")
                    logger.info(f"   Query: '{query[:50]}...'")
                    logger.info(f"   User ID: {user_id}")
                    logger.info(f"   Raw results type: {type(memory_results)}")
                    logger.info(f"   Results keys: {memory_results.keys() if isinstance(memory_results, dict) else 'N/A'}")
                    logger.info(f"   Total results count: {len(memory_results.get('results', [])) if isinstance(memory_results, dict) else 0}")
                    
                    # Log each individual memory
                    if isinstance(memory_results, dict) and 'results' in memory_results:
                        for idx, item in enumerate(memory_results.get('results', [])):
                            logger.info(f"   Memory {idx + 1}:")
                            logger.info(f"      Content: {item.get('memory', 'N/A')}")
                            logger.info(f"      Score: {item.get('score', 'N/A')}")
                            logger.info(f"      Metadata: {item.get('metadata', {})}")
                    else:
                        logger.info(f"   ⚠️ No results or unexpected format")
                    
                    memories = "\n".join([
                        f"- {item['memory']}" 
                        for item in memory_results.get("results", []) 
                        if item.get("memory")
                    ]) or "No previous context."

                    logger.info(f" Retrieved memories: {memories}")
                    analysis_start = datetime.now()
                    
                    # Perform analysis
                    analysis = await self._comprehensive_analysis(query, chat_history, memories)
                    analysis_time = (datetime.now() - analysis_start).total_seconds()
                    logger.info(f" Analysis completed in {analysis_time:.2f}s")
                    
                    # Cache the analysis
                    await self.cache_manager.cache_query(query, analysis, user_id, ttl=3600)
            
            # LOG: Enhanced analysis results
            logger.info(f" ANALYSIS RESULTS:")
            logger.info(f"   Intent: {analysis.get('semantic_intent', 'Unknown')}")
            business_opp = analysis.get('business_opportunity', {})
            logger.info(f"   Business Confidence: {business_opp.get('composite_confidence', 0)}/100")
            logger.info(f"   Engagement Level: {business_opp.get('engagement_level', 'none')}")
            logger.info(f"   Signal Breakdown: {business_opp.get('signal_breakdown', {})}")
            logger.info(f"   Tools Selected: {analysis.get('tools_to_use', [])}")
            logger.info(f"   Response Strategy: {analysis.get('response_strategy', {}).get('personality', 'Unknown')}")
            
            # LOG: Tool execution mode
            tool_execution = analysis.get('tool_execution', {})
            execution_mode = tool_execution.get('mode', 'parallel')
            logger.info(f"   Execution Mode: {execution_mode}")
            if execution_mode == 'sequential':
                logger.info(f"   Execution Order: {tool_execution.get('order', [])}")
                logger.info(f"   Dependency Reason: {tool_execution.get('dependency_reason', 'N/A')}")
            
            # Extract scraping guidance if web_search is used
            scraping_guidance = analysis.get('scraping_guidance', {})
            if scraping_guidance:
                logger.info(f"   Scraping Guidance: {scraping_guidance}")
                for tool_key, guidance in scraping_guidance.items():
                    logger.info(f"      {tool_key}: {guidance.get('scraping_level')} "
                              f"({guidance.get('scraping_count')} pages) - {guidance.get('scraping_reason')}")
            
            # STEP 2: Extract tools_to_use FIRST (needed for confirmation payload)
            tools_to_use = analysis.get('tools_to_use', [])
            
            # STEP 3: Check if scraping needs confirmation (AFTER tools_to_use is defined)
            if ENABLE_SCRAPING_CONFIRMATION and scraping_guidance and self._needs_confirmation(scraping_guidance):
                estimated_time = self._estimate_scraping_time(scraping_guidance)
                
                # Count total pages correctly - check tool_key names, not dict values
                total_pages = sum(
                    g.get('scraping_count', 0) 
                    for tool_key, g in scraping_guidance.items()
                    if 'web_search' in tool_key
                )
                
                logger.info(f"⚠️ High scraping detected ({total_pages} pages) - requesting user confirmation")
                
                # Generate confirmation token
                token = str(uuid.uuid4())
                
                # Build payload for pending confirmation
                pending_payload = {
                    "query": query,
                    "analysis": analysis,
                    "tools_to_use": tools_to_use,
                    "scraping_guidance": scraping_guidance,
                    "estimated_time_secs": estimated_time,
                    "chat_history": chat_history
                }
                
                # Store pending confirmation
                stored = await self.cache_manager.set_pending_confirmation(
                    token=token,
                    payload=pending_payload,
                    user_id=user_id
                )
                
                if stored:
                    # Return confirmation request to user
                    confirmation_message = (
                        f"This query requires scraping {total_pages} pages, which will take approximately "
                        f"{estimated_time} seconds. Would you like to continue? (Reply 'yes' to proceed or 'no' "
                        f"for faster minimal scraping)"
                    )
                    
                    logger.info(f"💬 Asking user for confirmation: {confirmation_message}")
                    
                    # Save original query and confirmation message to memory BEFORE returning
                    # This ensures conversation is preserved even if user changes topic
                    await self.task_queue.put(
                        AddBackgroundTask(
                            func=partial(self.memory.add),
                            params=(
                                [{"role": "user", "content": query}, {"role": "assistant", "content": confirmation_message}],
                                user_id,
                            ),
                        )
                    )
                    logger.info(f"💾 Saved confirmation exchange to memory")
                    
                    return {
                        "success": True,
                        "needs_confirmation": True,
                        "confirmation_token": token,
                        "estimated_time_secs": estimated_time,
                        "total_pages": total_pages,
                        "response": confirmation_message,
                        "message": confirmation_message  # For compatibility
                    }
                else:
                    logger.warning(f"⚠️ Failed to store confirmation - proceeding without confirmation")
                    # Fall through to normal execution
            
            # STEP 4: Check for cached tool results first
            cached_tool_results = None
            tools_cache_hit = False
            
            if tools_to_use:
                cached_tool_results = await self.cache_manager.get_cached_tool_results(
                    query, tools_to_use, user_id, scraping_guidance
                )
                if cached_tool_results:
                    tools_cache_hit = True
                    logger.info(f"🎯 USING CACHED TOOL RESULTS - Skipping tool execution")
            
            if tools_cache_hit and cached_tool_results:
                tool_results = cached_tool_results
                tool_time = 0.0  # Cache hit = instant
            else:
                # Execute tools if needed (may include middleware LLM call for sequential)
                tool_start = datetime.now()
                tool_results = await self._execute_tools(
                    tools_to_use,
                    query,
                    analysis,
                    user_id
                )
                tool_time = (datetime.now() - tool_start).total_seconds()
                logger.info(f" Tools executed in {tool_time:.2f}s")
                
                # Cache the tool results
                if tool_results:
                    await self.cache_manager.cache_tool_results(
                        query, tools_to_use, tool_results, user_id, scraping_guidance, ttl=3600
                    )
            
            if tool_results:
                logger.info(f" TOOL RESULTS SUMMARY:")
                for tool_name, result in tool_results.items():
                    if isinstance(result, dict) and result.get('success'):
                        logger.info(f"   {tool_name}: SUCCESS - {len(str(result))} chars of data")
                    elif isinstance(result, dict) and 'error' in result:
                        logger.info(f"   {tool_name}: ERROR - {result.get('error', 'Unknown')}")
                    else:
                        logger.info(f"   {tool_name}: RESULT - {type(result)} returned")
            else:
                logger.info(f" NO TOOLS EXECUTED - Conversational response only")
            
            response_start = datetime.now()
            logger.info(f" PASSING TO RESPONSE GENERATOR:")
            logger.info(f"   Analysis data: {len(str(analysis))} chars")
            logger.info(f"   Tool data: {len(str(tool_results))} chars")
            logger.info(f"   Strategy: {analysis.get('response_strategy', {})}")
            
            # Get memories for response generation if not cached
            if not cached_analysis:
                memory_results = await self.memory.search(query, user_id=user_id, limit=5)
                
                # Detailed mem0 logging
                logger.info(f"🧠 MEM0 SEARCH RESULTS (Response Generation Path):")
                logger.info(f"   Query: '{query[:50]}...'")
                logger.info(f"   User ID: {user_id}")
                logger.info(f"   Raw results type: {type(memory_results)}")
                logger.info(f"   Results keys: {memory_results.keys() if isinstance(memory_results, dict) else 'N/A'}")
                logger.info(f"   Total results count: {len(memory_results.get('results', [])) if isinstance(memory_results, dict) else 0}")
                
                # Log each individual memory
                if isinstance(memory_results, dict) and 'results' in memory_results:
                    for idx, item in enumerate(memory_results.get('results', [])):
                        logger.info(f"   Memory {idx + 1}:")
                        logger.info(f"      Content: {item.get('memory', 'N/A')}")
                        logger.info(f"      Score: {item.get('score', 'N/A')}")
                        logger.info(f"      Metadata: {item.get('metadata', {})}")
                else:
                    logger.info(f"   ⚠️ No results or unexpected format")
                
                memories = "\n".join([
                    f"- {item['memory']}" 
                    for item in memory_results.get("results", []) 
                    if item.get("memory")
                ]) or "No previous context."
            else:
                memories = "No previous context."
            
            final_response = await self._generate_response(
                query,
                analysis,
                tool_results,
                chat_history,
                memories=memories
            )
            
            await self.task_queue.put(
                AddBackgroundTask(
                    func=partial(self.memory.add),
                    params=(
                        [{"role": "user", "content": query}, {"role": "assistant", "content": final_response}],
                        user_id,
                    ),
                )
            )
            response_time = (datetime.now() - response_start).total_seconds()
            logger.info(f" Response generated in {response_time:.2f}s")
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            # Count actual LLM calls
            llm_calls = 2 if not cached_analysis else 1  # Brain skipped if cached + Heart
            if execution_mode == 'sequential' and not cached_analysis:
                llm_calls += 1  # Middleware
            
            logger.info(f" TOTAL PROCESSING TIME: {total_time:.2f}s ({llm_calls} LLM calls)")
            logger.info(f" ANALYSIS CACHE: {'HIT ✅' if cached_analysis else 'MISS ❌'}")
            logger.info(f" TOOL CACHE: {'HIT ✅' if tools_cache_hit else 'MISS ❌'}")
            
            return {
                "success": True,
                "response": final_response,
                "analysis": analysis,
                "tool_results": tool_results,
                "tools_used": analysis.get('tools_to_use', []),
                "execution_mode": execution_mode,
                "business_opportunity": analysis.get('business_opportunity', {}),
                "analysis_cache_hit": bool(cached_analysis),
                "tools_cache_hit": tools_cache_hit,
                "processing_time": {
                    "analysis": analysis_time,
                    "tools": tool_time,
                    "response": response_time,
                    "total": total_time
                }
            }
            
        except Exception as e:
            logger.error(f" Processing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": "I apologize, but I encountered an error. Please try again."
            }
    
    async def _resume_with_confirmation(self, pending: Dict, user_decision: str, start_time: datetime) -> Dict[str, Any]:
        """
        Resume processing after user confirmation
        
        Args:
            pending: Pending confirmation payload
            user_decision: 'yes' or 'no'
            start_time: Original start time for timing
        
        Returns:
            Processing result dict
        """
        query = pending.get('query')
        analysis = pending.get('analysis')
        tools_to_use = pending.get('tools_to_use', [])
        scraping_guidance = pending.get('scraping_guidance', {})
        chat_history = pending.get('chat_history', [])
        user_id = pending.get('user_id')
        
        logger.info(f"📝 Resuming query: '{query}' with decision: {user_decision}")
        
        # If user said no, downgrade scraping to low (1 page)
        if user_decision == 'no':
            logger.info(f"⬇️ Downgrading scraping to low (1 page) for all web_search tools")
            for tool_key in scraping_guidance:
                if 'web_search' in tool_key:
                    scraping_guidance[tool_key] = {
                        "scraping_level": "low",
                        "scraping_count": 1,
                        "scraping_reason": "User declined high scraping"
                    }
            
            # CRITICAL: Update analysis object with modified scraping_guidance
            # because _execute_tools reads from analysis, not from parameters
            analysis['scraping_guidance'] = scraping_guidance
            logger.info(f"✅ Updated analysis with downgraded scraping guidance")
        
        # Check cache first (use modified scraping_guidance for cache key)
        cached_tool_results = await self.cache_manager.get_cached_tool_results(
            query, tools_to_use, user_id, scraping_guidance
        )
        
        if cached_tool_results:
            tool_results = cached_tool_results
            tools_cache_hit = True
            tool_time = 0.0
            logger.info(f"🎯 Cache HIT for resumed query")
        else:
            # Execute tools
            tool_start = datetime.now()
            tool_results = await self._execute_tools(
                tools_to_use,
                query,
                analysis,
                user_id
            )
            tool_time = (datetime.now() - tool_start).total_seconds()
            tools_cache_hit = False
            
            # Cache the results
            if tool_results:
                await self.cache_manager.cache_tool_results(
                    query, tools_to_use, tool_results, user_id, scraping_guidance, ttl=3600
                )
        
        # Log results
        if tool_results:
            logger.info(f" TOOL RESULTS SUMMARY:")
            for tool_name, result in tool_results.items():
                if isinstance(result, dict) and result.get('success'):
                    logger.info(f"   {tool_name}: SUCCESS - {len(str(result))} chars of data")
        
        # Get memories for response generation
        memory_results = await self.memory.search(query, user_id=user_id, limit=5)
        
        # Detailed mem0 logging
        logger.info(f"🧠 MEM0 SEARCH RESULTS (Resume Confirmation Path):")
        logger.info(f"   Query: '{query[:50]}...'")
        logger.info(f"   User ID: {user_id}")
        logger.info(f"   Raw results type: {type(memory_results)}")
        logger.info(f"   Results keys: {memory_results.keys() if isinstance(memory_results, dict) else 'N/A'}")
        logger.info(f"   Total results count: {len(memory_results.get('results', [])) if isinstance(memory_results, dict) else 0}")
        
        # Log each individual memory
        if isinstance(memory_results, dict) and 'results' in memory_results:
            for idx, item in enumerate(memory_results.get('results', [])):
                logger.info(f"   Memory {idx + 1}:")
                logger.info(f"      Content: {item.get('memory', 'N/A')}")
                logger.info(f"      Score: {item.get('score', 'N/A')}")
                logger.info(f"      Metadata: {item.get('metadata', {})}")
        else:
            logger.info(f"   ⚠️ No results or unexpected format")
        
        memories = "\n".join([
            f"- {item['memory']}" 
            for item in memory_results.get("results", []) 
            if item.get("memory")
        ]) or "No previous context."
        
        # Generate response
        response_start = datetime.now()
        final_response = await self._generate_response(
            query,
            analysis,
            tool_results,
            chat_history,
            memories=memories
        )
        
        # Add to memory
        await self.task_queue.put(
            AddBackgroundTask(
                func=partial(self.memory.add),
                params=(
                    [{"role": "user", "content": query}, {"role": "assistant", "content": final_response}],
                    user_id,
                ),
            )
        )
        
        response_time = (datetime.now() - response_start).total_seconds()
        total_time = (datetime.now() - start_time).total_seconds()
        
        execution_mode = analysis.get('tool_execution', {}).get('mode', 'parallel')
        
        return {
            "success": True,
            "response": final_response,
            "analysis": analysis,
            "tool_results": tool_results,
            "tools_used": tools_to_use,
            "execution_mode": execution_mode,
            "business_opportunity": analysis.get('business_opportunity', {}),
            "analysis_cache_hit": True,  # Analysis was cached from pending
            "tools_cache_hit": tools_cache_hit,
            "confirmation_decision": user_decision,
            "processing_time": {
                "analysis": 0.0,  # Cached from pending
                "tools": tool_time,
                "response": response_time,
                "total": total_time
            }
        }
            
    async def background_task_worker(self) -> None:
        while True:
            task: AddBackgroundTask = await self.task_queue.get()
            try:
    
                func_name = getattr(task.func, "func", task.func).__name__ if hasattr(task.func, "__name__") else repr(task.func)
                logger.info(f"Executing background task: {func_name}")
                messages,user_id = task.params
                logger.info(f" Background task params: messages length={len(messages)}, user_id={user_id}")
                await task.func(messages=messages, user_id=user_id)

            except asyncio.CancelledError:
    
                break
            except Exception as e:
                logger.error(f"Error executing background task: {e}")
            finally:
                self.task_queue.task_done()

                
    def _start_worker_if_needed(self):
        """Start background worker once, on first use"""
        if not self._worker_started:
            asyncio.create_task(self.background_task_worker())
            self._worker_started = True
            logging.info("✅ OptimizedAgent background worker started")
    
    def _build_sentiment_language_guide(self, sentiment: Dict) -> str:
        """Build sentiment-driven language guidance"""
        emotion = sentiment.get('primary_emotion', 'casual')
        intensity = sentiment.get('intensity', 'medium')
        
        guides = {
            'frustrated': {
                'high': "User is highly frustrated - use very empathetic, understanding language. Be supportive and understanding", 
                'medium': "User is frustrated - be supportive and understanding",
                'low': "User is mildly frustrated - be gentle and reassuring"
            },
            'excited': {
                'high': "User is very excited - match their energy! Be enthusiastic and positive",
                'medium': "User is excited - be upbeat and encouraging",
                'low': "User is mildly excited - be positive and supportive"
            },
            'confused': {
                'high': "User is very confused - be extra patient and clear. Use simple language",
                'medium': "User is confused - be helpful and explanatory",
                'low': "User is slightly confused - be clarifying but not condescending"
            },
            'urgent': {
                'high': "User needs immediate help - be direct but supportive. Focus on solutions",
                'medium': "User has some urgency - be helpful and action-focused",
                'low': "User has mild urgency - be responsive and solution-oriented"
            },
            'casual': {
                'high': "User is very relaxed - be friendly and conversational",
                'medium': "User is casual - be warm and natural",
                'low': "User is somewhat casual - be friendly but focused"
            }
        }
        
        return guides.get(emotion, {}).get(intensity, "Be naturally helpful and friendly")

    async def _comprehensive_analysis(self, query: str, chat_history: List[Dict] = None, memories:str = "", pending_confirmation: Optional[Dict] = None) -> Dict[str, Any]:
        """Single LLM call for ALL analysis needs"""
        logger.info(f" ANALYSIS DEBUG:")
        logger.info(f"   Chat History Type: {type(chat_history)}")
        logger.info(f"   Chat History Content: {chat_history}")
        logger.info(f"   Chat History Length: {len(chat_history) if chat_history else 0}")
        
        # Build context from chat history
        context = chat_history[-2:] if chat_history else []
        logger.info(f"   Built Context: '{context}'")
        
        # Extract pending confirmation data if exists (for context section)
        pending_info = None
        if pending_confirmation:
            pending_info = {
                'query': pending_confirmation.get('payload', {}).get('query', ''),
                'estimated_time': pending_confirmation.get('payload', {}).get('estimated_time_secs', 0),
                'total_pages': pending_confirmation.get('payload', {}).get('total_pages', 0)
            }
            logger.info(f"📋 Pending confirmation detected: {pending_info['query'][:50]}...")
        
        # Build pending context string for main prompt
        pending_context = ""
        if pending_info:
            pending_context = f"""
PENDING ACTION AWAITING USER RESPONSE:
- Original Query: "{pending_info['query']}"
- Action Required: Scrape {pending_info['total_pages']} pages (~{pending_info['estimated_time']} seconds)
- System Asked: "Would you like to continue? (yes/no)"
"""
        
        # Create comprehensive prompt that does everything in one shot
        analysis_prompt = f"""You are analyzing queries for Mochan-D - an AI chatbot solution that:
- Automates customer support and sales (24/7 availability)
- Works across multiple platforms (WhatsApp, Facebook, Instagram, etc.)
- Uses RAG + Web Search for intelligent responses
- Serves businesses of all sizes needing to scale customer communication

LONG-TERM CONTEXT (Memories): {memories}
RECENT CONVERSATION: {context}
{pending_context}
USER QUERY: {query}

AVAILABLE TOOLS:
- web_search: Search internet for current information
- rag: Retrieve from uploaded knowledge base  
- calculator: Perform calculations and analysis

Perform ALL of the following analyses in ONE response:

1. MULTI-TASK DETECTION & DECOMPOSITION:
   - Analyze the user query to identify if it contains multiple distinct, actionable tasks or questions.
   - Look for:
     * Multiple questions separated by "and", "also", "plus", or similar connectors
     * Different types of information requests (e.g., weather + recommendations, prices + comparisons)
     * Sequential tasks where one leads to another
     * Independent tasks that can be handled separately
   
   - If 2 or more distinct tasks are found:
     * Set `multi_task_detected` to `true`
     * List each task clearly in the `sub_tasks` array
     * Determine if tasks are dependent (sequential) or independent (parallel)
   
   - If only one task is found, set `multi_task_detected` to `false`
   
   - Examples:
     * "What's the weather in Lucknow and what should I wear?" → 2 tasks: [weather query, clothing recommendation]
     * "iPhone 16 price and Samsung S24 price" → 2 tasks: [iPhone pricing, Samsung pricing]
     * "Compare our product with competitors" → 1 task: [product comparison]

2. SEMANTIC INTENT ANALYSIS:
   
   A. Understand the user's TRUE goal:
      - What do they actually want to achieve?
      - Consider their emotional state (urgent? frustrated? casual?)
      - What's the REAL intent behind their words?
      
   B. Synthesize understanding:
      - Based on decomposed tasks (from step 1), what is the user's ultimate goal?
      - Include every specific number, measurement, name, date, and technical detail from the query
      - Consider emotional cues, urgency signals, and conversation context

3. MOCHAN-D PRODUCT OPPORTUNITY ANALYSIS:
   Does the user's query relate to problems that Mochan-D's AI chatbot solution can solve?


   MOCHAN-D-SPECIFIC TRIGGERS (check for these pain points):
   - Customer support automation needs
   - High customer service costs or staff burden 
   - Need for 24/7 customer availability
   - Multiple messaging platform management difficulties (WhatsApp, Facebook, Instagram)
   - Repetitive customer query handling
   - Customer engagement/response time issues
   - Integration needs with CRM/payment systems for customer communication
   - Scaling customer communication challenges

   CONTEXTUAL TRIGGERS (Score: 50-70):
    - Mentions competitors
    - Asks "how to improve..." business processes
    - Growth/scaling discussions
    - Team efficiency concerns
    
   EMOTIONAL CUES (Score: 40-60):
   - Frustration → Empathy + solution
   - Celebration → Join joy, suggest growth
   - Worry → Reassurance + clarity
   
   Set business_opportunity.detected = true if query shows ANY of:
   - User states a current problem/challenge
   - User is actively seeking/evaluating solutions
   - User expresses dissatisfaction with current situation
   - User mentions "need", "looking for", "considering", "want to improve"


   CONFIDENCE SCORING:
   composite_confidence = (work_context + emotional_distress + solution_seeking + scale_scope) / 4
   
   - work_context: 0-100 (Business vs personal)
   - emotional_distress: 0-100 (Frustration/stress level)
   - solution_seeking: 0-100 (Actively looking for help?)
   - scale_scope: 0-100 (Size/urgency of problem)
   
   Score Bands:
   0-30: No business context → pure_empathy
   31-50: Ambiguous → empathetic_probing
   51-70: Possible → gentle_suggestion
   71-85: Clear pain → soft_pitch
   86-100: Hot lead → direct_consultation


   DO NOT trigger business_opportunity.detected = true for:
   - Pure research/comparison without context ("Compare X vs Y")
   - Definition questions ("What is X")
   - General knowledge inquiries
   - Personal health, relationships, entertainment
   - Weather, jokes, casual chat (unless leads to business context)
   - Pet problems, family issues


   If business opportunity detected:
   - Set business_opportunity.detected = true
   - Add "rag" to tools_to_use (fetch Mochan-D product docs)


   If query is about other business areas (accounting, inventory, website, etc.):
   - Set business_opportunity.detected = false

4. TOOL SELECTION FOR MULTI-TASK QUERIES:

   For EACH sub-task identified in step 1, select the most appropriate tool:
   
   RAG SELECTION (STRICT RULE):
   Select `rag` for a sub-task if EITHER:
   1. The sub-task is directly ABOUT Mochan-D (mentions "Mochan-D", "our/your product", "this chatbot", etc.)
   2. OR business_opportunity.detected = true for that sub-task

   GENERAL TOOL SELECTION:
   - `web_search`: For current information, prices, comparisons, weather, news, etc.
   - `calculator`: For mathematical calculations, statistical operations
   
   IMPORTANT: The `tools_to_use` array should contain one tool for each sub-task.
   - If you have 2 sub-tasks needing web_search, include ["web_search", "web_search"]
   - If you have 1 sub-task needing web_search and 1 needing calculator, include ["web_search", "calculator"]
   
   Use NO tools for:
   - Greetings, casual chat
   - General knowledge questions that don't require current information

5. SENTIMENT & PERSONALITY:
   - User's emotional state (frustrated/excited/casual/urgent/confused)
   - Best response personality (empathetic_friend/excited_buddy/helpful_dost/urgent_solver/patient_guide)

6. RESPONSE STRATEGY:
   - Response length (micro/short/medium/detailed)
   - Language style (hinglish/english/professional/casual)

7. WEB SCRAPING GUIDANCE (FOR WEB_SEARCH TOOL):

   When web_search is selected, determine appropriate scraping intensity:
   
   SCRAPING LEVELS:
   - "low" (1 page): Simple factual queries, quick lookups, single-source answers
     Examples: "what is capital of France", "current time", "definition of X"
   
   - "medium" (3 pages): Comparison queries, multi-source verification, moderate depth
     Examples: "compare iPhone vs Samsung", "best restaurants in Lucknow", "product reviews"
   
   - "high" (5 pages): Complex research, comprehensive analysis, multi-faceted queries
     Examples: "analyze market trends", "competitive landscape", "in-depth technical comparison"
   
   DECISION RULES:
   1. Query complexity: Simple fact → low, Comparison → medium, Research → high
   2. Expected answer breadth: Single point → low, Multiple points → medium, Comprehensive → high
   3. Verification needs: No verification → low, Cross-check → medium, Thorough validation → high
   
   For EACH web_search tool in tools_to_use, provide scraping guidance:
   - Set `scraping_level`: "low", "medium", or "high"
   - Set `scraping_count`: corresponding number (1, 3, or 5)
   - Include brief `scraping_reason`: why this level is appropriate
   
   If NO web_search tool is used, omit scraping_guidance entirely.
   
   IMPORTANT: For indexed tools (web_search_0, web_search_1), provide guidance for EACH:
   {{
     "scraping_guidance": {{
       "web_search_0": {{
         "scraping_level": "medium",
         "scraping_count": 3,
         "scraping_reason": "Comparison query requires multiple sources"
       }},
       "web_search_1": {{
         "scraping_level": "low",
         "scraping_count": 1,
         "scraping_reason": "Simple factual lookup"
       }}
     }}
   }}

8. DEPENDENCY & EXECUTION PLANNING FOR MULTI-TASK QUERIES:

    Step 1: Analyze task dependencies
    
    DEFAULT = PARALLEL (tasks are independent)
    
    For each sub-task, ask: "Does this task need information from another task to complete?"
    
    Common dependency patterns:
    - Weather + clothing recommendation → SEQUENTIAL (clothing depends on weather data)
    - Product features + competitor comparison → SEQUENTIAL (comparison needs product info)
    - iPhone price + Samsung price → PARALLEL (completely independent)
    - Math calculation + web search for formula → SEQUENTIAL (calculation needs formula)
    
    Step 2: Create execution plan with indexed tool names
    
    CRITICAL: Create unique indexed names for each tool execution:
    - Format: `tool_name_index` (e.g., `web_search_0`, `web_search_1`, `rag_0`)
    - The `order` array must contain these indexed names
    - The `enhanced_queries` object keys must EXACTLY match the indexed names in `order`
    
    Step 3: Generate queries for each indexed tool
    
    For PARALLEL mode:
    - Each indexed tool gets its own specific query based on its corresponding sub-task
    - Example: `web_search_0`: "iPhone 16 price", `web_search_1`: "Samsung S24 price"
    
    For SEQUENTIAL mode:
    - ONLY the first indexed tool (position 0) gets a real query
    - ALL subsequent tools get "WAIT_FOR_PREVIOUS"
    - Example: `rag_0`: "Mochan-D features", `web_search_0`: "WAIT_FOR_PREVIOUS"
    
    Query optimization rules:
    - RAG: "Mochan-D" + [specific topic from sub-task]
    - Calculator: Extract numbers from sub-task, create valid Python expression
    - Web_search: Transform sub-task into focused search query, preserve qualifiers (when, how much, what type), add "2025" if time-sensitive

9. CONFIRMATION RESPONSE ANALYSIS:

   Check if there is a PENDING ACTION in the context above.
   
   If PENDING ACTION EXISTS:
   
   The system asked: "Would you like to continue? (yes/no)"
   
   Identify the SUBJECT of the user's message:
   
   What is the user's message ABOUT?
   What is the user REFERRING to?
   
   If the message is ABOUT the pending action (answering the yes/no question):
   → Classify based on their answer:
      - Affirmative answer → "approve"
      - Negative answer or urgency for fast alternative → "decline"
   
   If the message is NOT about the pending action:
   → Classify as "new_query"
   → This includes: statements about other things, new questions, comments not related to the pending action
   
   Set confidence 0-100 based on clarity of subject identification.
   
   Reasoning: State what the subject of the message is and whether it refers to the pending action.
   
   If NO PENDING ACTION EXISTS:
   - has_pending: false
   - user_intent: "new_query"
   - confidence: 100

    EXAMPLES OF MULTI-TASK HANDLING:

    Example 1 - Multi-task Parallel (Independent tasks):
    Query: "What's today's weather in Lucknow and iPhone 16 price"
    Multi-task Analysis: 2 independent tasks → parallel
    Output:
    {{
    "multi_task_analysis": {{
        "multi_task_detected": true,
        "sub_tasks": ["Get today's weather for Lucknow", "Find iPhone 16 price"]
    }},
    "tools_to_use": ["web_search", "web_search"],
    "tool_execution": {{
        "mode": "parallel",
        "order": ["web_search_0", "web_search_1"],
        "dependency_reason": "Weather and phone pricing are independent queries"
    }},
    "enhanced_queries": {{
        "web_search_0": "today weather Lucknow",
        "web_search_1": "iPhone 16 price 2025"
    }}
    }}

    Example 2 - Multi-task Sequential (Dependent tasks):
    Query: "What's today's weather in Lucknow and what should I wear?"
    Multi-task Analysis: 2 dependent tasks → sequential
    Output:
    {{
    "multi_task_analysis": {{
        "multi_task_detected": true,
        "sub_tasks": ["Get today's weather for Lucknow", "Get clothing recommendation based on weather"]
    }},
    "tools_to_use": ["web_search", "web_search"],
    "tool_execution": {{
        "mode": "sequential",
        "order": ["web_search_0", "web_search_1"],
        "dependency_reason": "Clothing recommendation depends on weather data"
    }},
    "enhanced_queries": {{
        "web_search_0": "today weather Lucknow temperature conditions",
        "web_search_1": "WAIT_FOR_PREVIOUS"
    }}
    }}

    Example 3 - Single task:
    Query: "what is my product?"
    Multi-task Analysis: 1 task → single execution
    Output:
    {{
    "multi_task_analysis": {{
        "multi_task_detected": false,
        "sub_tasks": ["Get information about Mochan-D product"]
    }},
    "tools_to_use": ["rag"],
    "tool_execution": {{
        "mode": "parallel",
        "order": ["rag_0"],
        "dependency_reason": ""
    }},
    "enhanced_queries": {{
        "rag_0": "Mochan-D product information features"
    }}
    }}

Return ONLY valid JSON:
{{
    "multi_task_analysis": {{
        "multi_task_detected": true/false,
        "sub_tasks": ["description of task 1", "description of task 2"]
    }},
    "semantic_intent": "clear description of overall user goal",
    "confirmation_response": {{
        "has_pending": true/false,
        "user_intent": "approve|decline|new_query|ambiguous",
        "confidence": 0-100,
        "reasoning": "Brief explanation based on semantic analysis of user's TRUE intent"
    }},
    "business_opportunity": {{
        "detected": true/false,
        "composite_confidence": 0-100,
        "engagement_level": "direct_consultation|gentle_suggestion|empathetic_probing|pure_empathy",
        "signal_breakdown": {{
            "work_context": 0-100,
            "emotional_distress": 0-100,
            "solution_seeking": 0-100,
            "scale_scope": 0-100
        }},
        "pain_points": ["specific problems"],
        "solution_areas": ["how Mochan-D helps 1", "solution 2"],
        "recommended_approach": "empathy_first|solution_focused|consultation_ready"
    }},
    "tools_to_use": ["tool1", "tool2"],
    "tool_execution": {{
        "mode": "sequential|parallel",
        "order": ["tool1", "tool2"],
        "dependency_reason": "why sequential is needed"
    }},
    "enhanced_queries": {{
        "web_search": "optimized search query or WAIT_FOR_PREVIOUS",
        "rag": "optimized rag query",
        "calculator": "clear calculation"
    }},
    "scraping_guidance": {{
        "web_search_0": {{
            "scraping_level": "low|medium|high",
            "scraping_count": 1|3|5,
            "scraping_reason": "why this level"
        }}
    }},
    "tool_reasoning": "why these tools",
    "sentiment": {{
        "primary_emotion": "frustrated|excited|casual|urgent|confused",
        "intensity": "low|medium|high"
    }},
    "response_strategy": {{
        "personality": "empathetic_friend|excited_buddy|helpful_dost|urgent_solver|patient_guide",
        "length": "micro|short|medium|detailed",
        "language": "hinglish|english|professional|casual",
        "tone": "friendly|professional|empathetic|excited"
    }},
    "key_points_to_address": ["point1", "point2"]
}}"""

        try:
            messages = []
            messages.append({"role": "user", "content": analysis_prompt})
            
            
            logger.info(f" CALLING BRAIN LLM for analysis...")
            response = await self.brain_llm.generate(
                messages,
                temperature=0.1,
                system_prompt="You are an expert analyst. Analyze queries using multi-signal intelligence covering semantics, business opportunities, tool needs, and communication strategy. Return valid JSON only." 
            )
            
            # LOG: Raw LLM response
            logger.info(f" BRAIN LLM RAW RESPONSE: {len(response)} chars")
            logger.info(f" First 200 chars: {response[:200]}...")
            
            # Clean response
            cleaned = self._clean_json_response(response)
            logger.info(f" CLEANED RESPONSE: {len(cleaned)} chars")
            
            analysis = json.loads(cleaned)
            
            # Ensure tool_execution exists with defaults
            if 'tool_execution' not in analysis:
                analysis['tool_execution'] = {
                    'mode': 'parallel',
                    'order': [],
                    'dependency_reason': ''
                }
            
            # LOG: Parsed analysis details
            logger.info(f" Analysis complete: intent={analysis.get('semantic_intent')}, "
                       f"business={analysis.get('business_opportunity', {}).get('detected')}, "
                       f"confidence={analysis.get('business_opportunity', {}).get('composite_confidence', 0)}, "
                       f"tools={analysis.get('tools_to_use', [])}")
            
            logger.info(f" FULL ANALYSIS GENERATED:")
            logger.info(f"   Semantic Intent: {analysis.get('semantic_intent', 'Unknown')}")
            logger.info(f"   Business Opportunity: {analysis.get('business_opportunity', {})}")
            logger.info(f"   Tools to Use: {analysis.get('tools_to_use', [])}")
            logger.info(f"   Tool Reasoning: {analysis.get('tool_reasoning', 'None')}")
            logger.info(f"   Sentiment: {analysis.get('sentiment', {})}")
            logger.info(f"   Response Strategy: {analysis.get('response_strategy', {})}")
            logger.info(f"   Enhanced Queries: {analysis.get('enhanced_queries', {})}")
            logger.info(f"   Tool Execution: {analysis.get('tool_execution', {})}")
            return analysis
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            # Return safe defaults
            return {
                "semantic_intent": "general_inquiry",
                "business_opportunity": {
                    "detected": False, 
                    "composite_confidence": 0,
                    "engagement_level": "pure_empathy",
                    "signal_breakdown": {
                        "work_context": 0,
                        "emotional_distress": 0,
                        "solution_seeking": 0,
                        "scale_scope": 0
                    },
                    "recommended_approach": "empathy_first"
                },
                "tools_to_use": [],
                "tool_execution": {
                    "mode": "parallel",
                    "order": [],
                    "dependency_reason": ""
                },
                "enhanced_queries": {},
                "sentiment": {"primary_emotion": "casual", "intensity": "medium"},
                "response_strategy": {
                    "personality": "helpful_dost",
                    "length": "medium",
                    "language": "hinglish",
                    "tone": "friendly"
                }
            }
    
    async def _execute_tools(self, tools: List[str], query: str, analysis: Dict, user_id: str = None) -> Dict[str, Any]:
        """Execute tools with smart parallel/sequential handling based on dependencies"""
        
        if not tools:
            return {}
        
        # Check execution mode from analysis
        tool_execution = analysis.get('tool_execution', {})
        execution_mode = tool_execution.get('mode', 'parallel')
        
        # Route to appropriate execution method
        if execution_mode == 'sequential' and len(tools) > 1:
            logger.info(f" SEQUENTIAL EXECUTION MODE")
            return await self._execute_sequential(tools, query, analysis, user_id)
        else:
            logger.info(f" PARALLEL EXECUTION MODE")
            return await self._execute_parallel(tools, query, analysis, user_id)
    
    async def _execute_parallel(self, tools: List[str], query: str, analysis: Dict, user_id: str = None) -> Dict[str, Any]:
        """Execute tools in parallel (handles duplicate tool names)"""
        results = {}
        enhanced_queries = analysis.get('enhanced_queries', {})
        scraping_guidance = analysis.get('scraping_guidance', {})
        
        logger.info(f"Enhanced queries for parallel execution: {enhanced_queries}")
        logger.info(f"Scraping guidance for parallel execution: {scraping_guidance}")
        
        # Execute tools in parallel for speed
        tasks = []
        tool_counter = {}  # Track occurrences of each tool type
        
        for i, tool in enumerate(tools):
            if tool in self.available_tools:
                # Count tool occurrences for unique keys
                count = tool_counter.get(tool, 0)
                tool_counter[tool] = count + 1
                
                # Try indexed key first, then fall back to non-indexed
                indexed_key = f"{tool}_{i}"
                tool_query = enhanced_queries.get(indexed_key) or enhanced_queries.get(tool, query)
                
                logger.info(f"🔧 {tool.upper()} #{i} ENHANCED QUERY: '{tool_query}'")
                
                # Get scraping params for web_search tools
                scrape_count = None
                if tool == 'web_search' and indexed_key in scraping_guidance:
                    guidance = scraping_guidance[indexed_key]
                    scrape_count = guidance.get('scraping_count', 3)
                    scraping_level = guidance.get('scraping_level', 'medium')
                    logger.info(f"   📊 Scraping: {scraping_level} level ({scrape_count} pages)")
                    logger.info(f"   📋 Reason: {guidance.get('scraping_reason', 'N/A')}")
                
                # Store results with unique keys
                result_key = f"{tool}_{i}" if count > 0 else tool
                
                # Build kwargs with scraping params if applicable
                tool_kwargs = {"query": tool_query, "user_id": user_id}
                if scrape_count is not None:
                    tool_kwargs["scrape_top"] = scrape_count
                
                task = self.tool_manager.execute_tool(tool, **tool_kwargs)
                tasks.append((result_key, task))
        
        if tasks:
            # Gather all results in parallel
            for tool_name, task in tasks:
                try:
                    result = await task
                    results[tool_name] = result
                    logger.info(f" Tool {tool_name} executed successfully")
                except Exception as e:
                    logger.error(f" Tool {tool_name} failed: {e}")
                    results[tool_name] = {"error": str(e)}
        
        return results
    
    async def _execute_sequential(self, tools: List[str], query: str, analysis: Dict, user_id: str = None) -> Dict[str, Any]:
        """Execute tools sequentially with middleware for dependent queries"""
        results = {}
        enhanced_queries = analysis.get('enhanced_queries', {})
        scraping_guidance = analysis.get('scraping_guidance', {})
        tool_execution = analysis.get('tool_execution', {})
        order = tool_execution.get('order', tools)
        
        logger.info(f"   Execution order: {order}")
        logger.info(f"   Reason: {tool_execution.get('dependency_reason', 'N/A')}")
        logger.info(f"   Scraping guidance: {scraping_guidance}")
        
        # Execute first tool
        first_tool_key = order[0]  # e.g., 'web_search_0'
        first_tool_name = first_tool_key.rsplit('_', 1)[0] if '_' in first_tool_key and first_tool_key.split('_')[-1].isdigit() else first_tool_key
        # ^ Strips index: 'web_search_0' -> 'web_search'
        
        first_query = enhanced_queries.get(first_tool_key, query)
        logger.info(f"   → Step 1: Executing {first_tool_key.upper()} with query: '{first_query}'")
        
        # Get scraping params for first tool if it's web_search
        first_tool_kwargs = {"query": first_query, "user_id": user_id}
        if first_tool_name == 'web_search' and first_tool_key in scraping_guidance:
            guidance = scraping_guidance[first_tool_key]
            scrape_count = guidance.get('scraping_count', 3)
            scraping_level = guidance.get('scraping_level', 'medium')
            first_tool_kwargs["scrape_top"] = scrape_count
            logger.info(f"      Scraping: {scraping_level} level ({scrape_count} pages)")
        
        try:
            results[first_tool_key] = await self.tool_manager.execute_tool(first_tool_name, **first_tool_kwargs)
            logger.info(f"   ✅ {first_tool_key} completed")
        except Exception as e:
            logger.error(f"   ❌ {first_tool_key} failed: {e}")
            results[first_tool_key] = {"error": str(e)}
            return results
        
        # Execute remaining tools with middleware
        for i in range(1, len(order)):
            current_tool_key = order[i]  # e.g., 'web_search_1'
            current_tool_name = current_tool_key.rsplit('_', 1)[0] if '_' in current_tool_key and current_tool_key.split('_')[-1].isdigit() else current_tool_key
            # ^ Strips index: 'web_search_1' -> 'web_search'
            
            # Check if this tool needs middleware
            if enhanced_queries.get(current_tool_key) == "WAIT_FOR_PREVIOUS":
                logger.info(f"   → Step 2: Middleware generating enhanced query for {current_tool_key}...")
                
                # Use middleware to generate enhanced query from previous results
                enhanced_query = await self._middleware_summarizer(
                    previous_results=results,
                    original_query=query,
                    next_tool=current_tool_name  # Pass base name, not indexed name
                )
                logger.info(f"   → Middleware output: '{enhanced_query}'")
            else:
                enhanced_query = enhanced_queries.get(current_tool_key, query)
            
            # Execute current tool
            logger.info(f"   → Step {i+2}: Executing {current_tool_key.upper()} with query: '{enhanced_query}'")
            
            # Get scraping params for current tool if it's web_search
            current_tool_kwargs = {"query": enhanced_query, "user_id": user_id}
            if current_tool_name == 'web_search' and current_tool_key in scraping_guidance:
                guidance = scraping_guidance[current_tool_key]
                scrape_count = guidance.get('scraping_count', 3)
                scraping_level = guidance.get('scraping_level', 'medium')
                current_tool_kwargs["scrape_top"] = scrape_count
                logger.info(f"      Scraping: {scraping_level} level ({scrape_count} pages)")
            
            try:
                results[current_tool_key] = await self.tool_manager.execute_tool(current_tool_name, **current_tool_kwargs)
                logger.info(f"   ✅ {current_tool_key} completed")
            except Exception as e:
                logger.error(f"   ❌ {current_tool_key} failed: {e}")
                results[current_tool_key] = {"error": str(e)}
        
        return results
    
    async def _middleware_summarizer(self, previous_results: Dict, original_query: str, next_tool: str) -> str:
        """Middleware: Extract key info from previous tool results and generate enhanced query"""
        
        # Format previous results
        previous_data = []
        for tool_name, result in previous_results.items():
            if isinstance(result, dict):
                if 'retrieved' in result:
                    previous_data.append(f"{tool_name.upper()} found: {result['retrieved'][:1000]}")
                elif 'results' in result and isinstance(result['results'], list):
                    for item in result['results'][:3]:
                        if 'snippet' in item:
                            previous_data.append(f"{tool_name.upper()}: {item['snippet']}")
        
        previous_summary = "\n".join(previous_data) if previous_data else "No data from previous tools"
        
        # SPECIAL HANDLING FOR CALCULATOR
        if next_tool == "calculator":
            middleware_prompt = f"""Extract numbers from data and create a math expression.

        ORIGINAL USER QUERY: {original_query}

        DATA FROM PREVIOUS TOOLS:
        {previous_summary}

        YOUR TASK:
        1. Find all numbers in the data above
        2. Understand what calculation the user wants from their query
        3. Create a valid Python math expression

        RULES:
        - Extract numbers only (remove ₹, $, %, commas)
        - Use operators: + - * / ( )
        - Match the calculation to user's query intent:
        * "total" or "sum" → add numbers
        * "difference" or "compare" → subtract
        * "multiply" or "times" → multiply
        * "percentage" or "discount" → multiply by decimal (15% = 0.15)
        * Complex queries → use parentheses for order

        EXAMPLES:
        Query: "compare prices", Data: "Item A: $2000, Item B: $1500"
        → "2000 - 1500"

        Query: "calculate 15% of 5000", Data: none needed
        → "5000 * 0.15"

        Query: "total cost for 3 items at 500 each", Data: "Price: ₹500"
        → "500 * 3"

        Query: "trip cost", Data: "Bus ₹600, Hotel ₹1000/night for 7 days, Food ₹200/day"
        → "600 + (1000*7) + (200*7)"

        Return ONLY a valid math expression. If you cannot determine what to calculate, return "SKIP"."""

        
        else:
            middleware_prompt = f"""You are a query generator. Analyze the previous results and create the NEXT search query.

                ORIGINAL USER QUERY: {original_query}

                PREVIOUS TOOL RESULTS:
                {previous_summary}

                INSTRUCTIONS:
                1. Read the previous results carefully
                2. Determine what the user wants next based on their original query
                3. If the query has conditional logic (if/then/else), evaluate the condition using the previous results
                4. Generate a specific, focused search query for what comes next

                CONDITIONAL QUERY RULES:
                - If query says "if weather is good/clear/sunny → suggest OUTDOOR"
                - If query says "if weather is bad/rainy/cloudy → suggest INDOOR"
                - Check the previous weather data to determine which condition is true
                - Weather indicators:
                * GOOD/OUTDOOR: "sunny", "clear", "75°F or higher", "0% rain", "no precipitation"
                * BAD/INDOOR: "rain", "storm", "cloudy", "cold", "high precipitation"

                COMPARISON QUERY RULES:
                - Extract the category/technology from previous results (NOT brand names)
                - Add "competitors" or "alternatives" or "comparison"
                - Example: "customer support chatbot" → "customer support chatbot competitors 2025"

                EXAMPLES:

                Example 1 (Weather Conditional):
                Query: "Check weather in Lucknow. If clear suggest outdoor events, else indoor events"
                Previous: "Lucknow: 85°F, Sunny, 0% rain, Clear skies"
                Analysis: Weather is CLEAR (sunny, 0% rain, 85°F) → User wants OUTDOOR
                Output: outdoor events activities Lucknow 2025

                Example 2 (Weather Conditional - Bad Weather):
                Query: "Check weather in Lucknow. If clear suggest outdoor events, else indoor events"
                Previous: "Lucknow: 65°F, Heavy rain, 90% precipitation"
                Analysis: Weather is BAD (rain, 90% precipitation) → User wants INDOOR
                Output: indoor events activities Lucknow 2025

                Example 3 (Comparison):
                Query: "compare competitors"
                Previous: "Mochan-D is a customer support chatbot for WhatsApp"
                Analysis: User wants competitors of customer support chatbots
                Output: customer support chatbot WhatsApp competitors 2025

                Example 4 (Product Info):
                Query: "compare pricing"
                Previous: "ProductX is an AI chatbot builder platform"
                Analysis: User wants pricing comparison for AI chatbot builders
                Output: AI chatbot builder pricing comparison 2025

                YOUR TASK:
                Generate the next search query based on the analysis above.
                Return ONLY the search query (max 10 words). No explanations."""
        
        try:
            logger.info(f"🔄 Calling middleware LLM...")
            
            response = await self.brain_llm.generate(
                [{"role": "user", "content": middleware_prompt}],
                temperature=0.4,
                max_tokens=100
            )
            
            enhanced_query = response.strip()
            logger.info(f" Middleware generated: '{enhanced_query}'")
            
            return enhanced_query
            
        except Exception as e:
            logger.error(f"Middleware failed: {e}")
            return original_query

    
    async def _generate_response(self, query: str, analysis: Dict, tool_results: Dict, chat_history: List[Dict], memories:str="") -> str:
        """Generate response with simple business mode switching like old system"""
        
        # Extract key elements
        intent = analysis.get('semantic_intent', '')
        business_opp = analysis.get('business_opportunity', {})
        sentiment = analysis.get('sentiment', {})
        strategy = analysis.get('response_strategy', {})
        
        # Simple binary business mode logic (like your old system)
        business_detected = business_opp.get('detected', False)
        conversation_mode = "Smart Business Friend" if business_detected else "Casual Dost"
        
        # Actually USE the sentiment guide you built
        sentiment_guidance = self._build_sentiment_language_guide(sentiment)
        
        # Enhanced logging
        logger.info(f"  RESPONSE GENERATION INPUTS:")
        logger.info(f"   Intent: {intent}")
        logger.info(f"   Business Opportunity Detected: {business_detected}")
        logger.info(f"   Conversation Mode: {conversation_mode}")
        logger.info(f"   User Emotion: {sentiment.get('primary_emotion', 'casual')}")
        logger.info(f"   Sentiment Guidance: {sentiment_guidance}")
        logger.info(f"   Response Personality: {strategy.get('personality', 'helpful_dost')}")
        logger.info(f"   Response Length: {strategy.get('length', 'medium')}")
        logger.info(f"   Language Style: {strategy.get('language', 'hinglish')}")
        
        # Format tool results
        tool_data = self._format_tool_results(tool_results)
        logger.info(f" FORMATTED TOOL DATA: {len(tool_data)} chars")
        
        # Build memory context to avoid repetition
        recent_phrases = self._extract_recent_phrases(chat_history)
        logger.info(f" RECENT PHRASES TO AVOID: {recent_phrases}")
        
        # Simple, clean prompt (like your old system)
        response_prompt = f"""You are Mochan-D (Mochand Dost) - an AI companion who's equal parts:
- Helpful friend (dost) who genuinely cares
- Smart business consultant who spots opportunities  
- Natural conversationalist who builds relationships
- Clever sales agent who never feels pushy

YOUR PERSONALITY:

Base Mode (Casual Dost): Warm, friendly Hinglish, picks up emotional cues, conversational not robotic

Business Mode (Smart Consultant): Maintains friendly tone + strategic depth, spots pain points, connects to solutions naturally (NEVER forced)

        CURRENT CONVERSATION CONTEXT:
        - User Intent: {intent}
        - Business Status: {business_detected}
            {f"- Confidence: {business_opp.get('composite_confidence', 0)}/100" if business_detected else ""}
            {f"- Pain Points: {business_opp.get('pain_points', [])}" if business_detected else ""}
            {f"- Solutions: {business_opp.get('solution_areas', [])}" if business_detected else ""}
        - Conversation Mode: {conversation_mode}
        - User Emotion: {sentiment.get('primary_emotion', 'casual')} ({sentiment.get('intensity', 'medium')})
        - User Sentiment Guide: {sentiment_guidance}

        DATA AUTHORITY CONTEXT:

        When data is presented as WEB_SEARCH or RAG results:
        - This represents CURRENT REALITY (not training memory)
        - This is what exists in the world RIGHT NOW
        - Your training knowledge is a backup reference only

        Build your response using the tool data as your source of truth
        
        AVAILABLE DATA TO USE NATURALLY:
        {tool_data}

        CONVERSATION MEMORY:
        {memories}

        RESPONSE REQUIREMENTS:
        - Personality: {strategy.get('personality', 'helpful_dost')}
        - Length: {strategy.get('length', 'medium')} 
        - Language: {strategy.get('language', 'hinglish')}
        - Tone: {strategy.get('tone', 'friendly')}

        🎯 RESPONSE RULES:

        CORE PRINCIPLES:
        1. NEVER echo/restate what user said
        2. NEVER announce tool usage ("Let me search...", "I found...")
        3. Match emotional energy PRECISELY using sentiment guide
        4. Stay in character as their dost

        BUSINESS OPPORTUNITY HANDLING:

        NO Opportunity (0-30): Pure friend mode, NO sales, just helpful

        LOW Opportunity (31-50): Empathetic probing - address query, then ONE gentle exploratory question

        MEDIUM Opportunity (51-70): Gentle suggestion - solve query fully, acknowledge challenge, drop subtle hint, ask ONE question
        Example: "Manual processes are tough. We help businesses with exactly this. What's your biggest bottleneck?"

        HIGH Opportunity (71-85): Soft pitch - solve query, naturally connect pain to Mochan-D, share ONE capability, invite to learn more
        Example: "That ticket chaos is real, yaar. Mochan-D automates these 24/7 while staying personal. Want to see how it works for businesses like yours?"

        VERY HIGH Opportunity (86-100): Direct consultation - address pain immediately, clear value prop, focus on their ROI, create urgency through value, clear CTA
        Example: "Losing deals to faster competitors - that's money on the table, bhai. Mochan-D gives 24/7 sales with AI that learns YOUR business. Should I show you the setup?"

        SALES TECHNIQUES:
        - Empathy Hook: "Sounds like..." / "That's rough, yaar..."
        - Correlation Weave: Natural segue from their world to solution
        - Social Proof: "A lot of startups face this..."
        - ROI Translator: Features → their specific benefits
        - Assumptive Consultant: "How many touchpoints juggling?"

        CLOSING:
        - Casual query → End naturally, no pitch
        - Warm lead → Ask ONE question, plant seed
        - Hot lead → Clear CTA but conversational
        - MIX IT UP: Don't always ask questions

        CRITICAL DON'TS:
        ❌ Repeat user's words
        ❌ Corporate jargon
        ❌ Sound desperate/pushy
        ❌ Force Mochan-D if no opportunity
        ❌ Multiple questions (1 max)
        
        ✅ DO: Sound like smart friend who knows solutions, build relationships, use data invisibly, match communication style, create value even if no sale today

        USER QUERY: {query}

        NOW RESPOND as Mochand Dost in {conversation_mode} mode. Be natural, helpful, strategic, human. If business opportunity exists, weave it like a skilled storyteller - make them see value without feeling sold to. If casual chat, be the best dost ever.
        Remember: You're building relationships that could turn into business. Play it smart, smooth, genuine."""
        try:
            # Determine max tokens based on length strategy
            max_tokens = {
                "micro": 150,
                "short": 300,
                "medium": 500,
                "detailed": 700
            }.get(strategy.get('length', 'medium'), 500)
            
            logger.info(f" CALLING HEART LLM for response generation...")
            logger.info(f" Max tokens: {max_tokens}, Temperature: 0.4")
            
            # messages = chat_history if chat_history else []
            messages = []
            messages.append({"role": "user", "content": response_prompt})
            
            response = await self.heart_llm.generate(
                messages,
                temperature=0.4,
                max_tokens=4000,
                system_prompt="You are Mochand Dost - conversational AI. If business opportunity exists, naturally pitch Mochan-D. Always natural conversation - NEVER JSON/analysis/structured data. Warm and helpful."
            )
            
            # LOG: Raw response from Heart LLM
            logger.info(f" HEART LLM RAW RESPONSE: {len(response)} chars")
            logger.info(f" First 200 chars: {response[:200]}...")
            
            # Clean and format
            response = self._clean_response(response)
            logger.info(f" FINAL CLEANED RESPONSE: {len(response)} chars")
            logger.info(f" FINAL RESPONSE: {response}")
            
            logger.info(f" Response generated: {len(response)} chars")
            return response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "I apologize, but I had trouble generating a response. Could you please try again?"
    
    def _build_context(self, chat_history: List[Dict]) -> str:
        """Build context from chat history"""
        if not chat_history:
            return "No previous context"
        
        recent = chat_history
        context_parts = []
        for item in recent:
            if 'query' in item:
                context_parts.append(f"User: {item['query']}")
            elif 'content' in item:
                context_parts.append(f"{item.get('role', 'user')}: {item['content']}")
        
        return " \n ".join(context_parts) if context_parts else "No previous context"
    
    def _format_tool_results(self, tool_results: dict) -> str:
        """Format tool results for response generation, handling different tool structures with Redis caching."""
        if not tool_results:
            return "No external data available"
        
        logger.info(f" RAW TOOL RESULTS DEBUG:")
        for tool_name, result in tool_results.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"TOOL: {tool_name.upper()}")
            logger.info(f"{'='*60}")
            
            if tool_name == 'web_search' and isinstance(result, dict):
                logger.info(f"Web Search Query: {result.get('query', 'N/A')}")
                logger.info(f"Success: {result.get('success', False)}")
                logger.info(f"Scraped Count: {result.get('scraped_count', 0)}") 
                
                if 'results' in result and isinstance(result['results'], list):
                    logger.info(f"Number of results: {len(result['results'])}")
                    
                    for idx, item in enumerate(result['results'][:5]):
                        logger.info(f"\n--- Result {idx+1} ---")
                        logger.info(f"Title: {item.get('title', 'No title')}")
                        logger.info(f"Snippet: {item.get('snippet', 'No snippet')}")
                        logger.info(f"Link: {item.get('link', 'No link')}")
                        
                        # scraped content
                        if 'scraped_content' in item:
                            scraped = item['scraped_content']
                            if scraped and not scraped.startswith("["):
                                logger.info(f"Scraped: {len(scraped)} chars")
                                logger.debug(f"Preview: {scraped[:200]}...")
                            else:
                                logger.info(f"Scraped: {scraped}")
        
        logger.info(f"\n{'='*60}\n")
        
        formatted = []
        
        for tool, result in tool_results.items():
            if isinstance(result, dict) and 'error' not in result:
                # Handle RAG-style result
                if 'success' in result and result['success']:
                    if 'retrieved' in result:
                        retrieved = result.get('retrieved', '')
                        chunks = result.get('chunks', [])
                        formatted.append(f"{tool.upper()} RETRIEVED TEXT:\n{retrieved}\n")
                        if chunks:
                            formatted.append(f"{tool.upper()} CHUNKS:\n" + "\n---\n".join(chunks))
                    
                    # Handle web search-style results
                    elif 'results' in result and isinstance(result['results'], list):
                        formatted.append(f"{tool.upper()} SEARCH RESULTS for query: {result.get('query', '')}\n")
                        for item in result['results']:
                            title = item.get('title', 'No title')
                            snippet = item.get('snippet', '')
                            link = item.get('link', '')
                            
                            # scraped content
                            if 'scraped_content' in item and item['scraped_content']:
                                scraped = item['scraped_content']
                                if not scraped.startswith("["):
                                    # UNIVERSAL CLEANUP - no char limit
                                    lines = scraped.split('\n')
                                    cleaned_lines = []
                                    
                                    for line in lines:
                                        line = line.strip()
                                        if len(line) < 40:  # Skip short lines (nav/menus)
                                            continue
                                        if line.count('http') > 2:  # Skip link lists
                                            continue
                                        if line.startswith('![') or line.startswith('Image'):  # Skip images
                                            continue
                                        cleaned_lines.append(line)
                                    
                                    # Join all cleaned lines
                                    cleaned = '\n'.join(cleaned_lines)
                                    
                                    formatted.append(f"- {title}\n  Content:\n{cleaned}\n  Link: {link}")
                                else:
                                    formatted.append(f"- {title}\n  {snippet}\n  Link: {link}")
                            else:
                                formatted.append(f"- {title}\n  {snippet}\n  Link: {link}")
                    
                    # Generic fallback for other data/result keys
                    elif 'data' in result:
                        formatted.append(f"{tool.upper()} DATA:\n{result['data']}")
                    elif 'result' in result:
                        formatted.append(f"{tool.upper()} RESULT:\n{result['result']}")
                    
                    else:
                        formatted.append(f"{tool.upper()}: Success but no recognizable content")
                else:
                    formatted.append(f"{tool.upper()}: No data retrieved or request failed")
            
            elif isinstance(result, str):
                formatted.append(f"{tool.upper()}: {result}")
        
        final_formatted = "\n\n".join(formatted) if formatted else "No usable tool data"
        
        # Cache the formatted tool data (fire and forget - don't wait)
        asyncio.create_task(self.cache_manager.cache_tool_data(tool_results, final_formatted, ttl=7200))
        
        return final_formatted


    def _extract_recent_phrases(self, chat_history: List[Dict]) -> List[str]:
        """Extract recent phrases to avoid repetition"""
        if not chat_history:
            return []
        
        phrases = []
        for msg in chat_history[-4:]:
            if msg.get('role') == 'assistant' or 'response' in msg:
                content = msg.get('content', msg.get('response', ''))
                # Get first 30 chars of each sentence
                sentences = content.split('.')
                for sentence in sentences[:2]:
                    if sentence.strip():
                        phrases.append(sentence.strip()[:30])
        
        return phrases[-5:] if phrases else []
    
    def _clean_json_response(self, response: str) -> str:
        """Clean LLM response for JSON parsing"""
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
        
        return response
    
    def _clean_response(self, response: str) -> str:
        """Clean final response for display"""
        # Remove any system thinking
        response = self._clean_json_response(response)
        
        # Fix formatting for display
        response = response.replace('- ', '-')
        response = response.strip()
        
        return response
