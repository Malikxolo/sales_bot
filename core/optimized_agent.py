"""
Optimized Single-Pass Agent System
Combines semantic analysis, tool execution, and response generation in minimal LLM calls
Now supports sequential tool execution with middleware for dependent tools
WITH REDIS CACHING for queries and formatted tool data
"""

import json
import logging
import asyncio
import uuid
import re
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
from os import getenv
from mem0 import AsyncMemory
import time
from functools import partial
from mem0.configs.base import MemoryConfig
from .config import AddBackgroundTask
from .redis_manager import RedisCacheManager

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



class OptimizedAgent:
    """Single-pass agent that minimizes LLM calls while maintaining all functionality"""
    
    def __init__(self, brain_llm, heart_llm, tool_manager, router_llm=None):
        self.brain_llm = brain_llm
        self.heart_llm = heart_llm
        self.router_llm = router_llm if router_llm else heart_llm
        self.tool_manager = tool_manager
        self.available_tools = tool_manager.get_available_tools()
        self.memory = AsyncMemory(config)
        self.task_queue: asyncio.Queue["AddBackgroundTask"] = asyncio.Queue()
        self._worker_started = False
        
        # Initialize Redis cache manager
        self.cache_manager = RedisCacheManager()
        
        logger.info(f"OptimizedAgent initialized with tools: {self.available_tools}")
        logger.info(f"Router LLM: {'DEDICATED ✅' if router_llm else 'SHARED (heart_llm) ⚠️'}")
        logger.info(f"Redis caching: {'ENABLED ✅' if self.cache_manager.enabled else 'DISABLED ⚠️'}")
    
    async def process_query(self, query: str, chat_history: List[Dict] = None, user_id: str = None, mode:str = None, source: str = "whatsapp") -> Dict[str, Any]:
        """Process query with minimal LLM calls and Redis caching"""
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
        needs_cot = None  # Track which analysis path was taken
        
        try:
            # STEP 1: Check cache or analyze
            cached_analysis = await self.cache_manager.get_cached_query(query, user_id)
            
            if cached_analysis:
                logger.info(f"🎯 USING CACHED ANALYSIS - Skipping Brain LLM call")
                analysis = cached_analysis
                analysis_time = 0.0  # Cache hit = instant
                needs_cot = None  # Unknown for cached analysis
            else:
                # Retrieve memories
                eli = time.time()
                memory_results = await self.memory.search(query[:100], user_id=user_id, limit=5)
                logger.info(f" Memory retrieval took {time.time() - eli:.2f}s")
                # Detailed mem0 logging
                logger.info(f"🧠 MEM0 SEARCH RESULTS:")
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
                
                # ROUTING LAYER: Decide which analysis path to take
                routing_decision = await self._route_query(query, chat_history, memories)
                needs_cot = routing_decision.get('needs_cot', True)
                
                # Route to appropriate analysis function
                if needs_cot:
                    logger.info(f"💰 COST PATH: COMPLEX (Qwen CoT) - Deep reasoning required")
                    analysis = await self._comprehensive_analysis(query, chat_history, memories)
                else:
                    logger.info(f"💰 COST PATH: SIMPLE (Llama Fast) - Straightforward query")
                    analysis = await self._simple_analysis(query, chat_history, memories)
                
                analysis_time = (datetime.now() - analysis_start).total_seconds()
                logger.info(f" Analysis completed in {analysis_time:.2f}s")
                
                # Cache the analysis
                await self.cache_manager.cache_query(query, analysis, user_id, ttl=3600)
            
            # LOG: Enhanced analysis results
            logger.info(f" ANALYSIS RESULTS:")
            logger.info(f"   Intent: {analysis.get('semantic_intent', 'Unknown')}")
            
            # LOG: Reasoning about tool selection
            expansion_reasoning = analysis.get('expansion_reasoning', '')
            if expansion_reasoning:
                logger.info(f"   🧠 Model Reasoning: {expansion_reasoning}")
            
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
            
            # STEP 2: Extract tools_to_use
            tools_to_use = analysis.get('tools_to_use', [])
            
            # STEP 3: Execute tools
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
                    query, tools_to_use, tool_results, user_id, ttl=3600
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
                memories=memories,
                mode=mode,
                source=source
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
            if cached_analysis:
                llm_calls = 1  # Only Heart (response generation)
                analysis_path = "CACHED"
            else:
                llm_calls = 1  # Router
                llm_calls += 1  # Analysis (either CoT or Simple)
                llm_calls += 1  # Heart (response generation)
                if execution_mode == 'sequential':
                    llm_calls += 1  # Middleware for sequential tools
                analysis_path = "COT" if needs_cot else "SIMPLE"
            
            logger.info(f" TOTAL PROCESSING TIME: {total_time:.2f}s ({llm_calls} LLM calls)")
            logger.info(f" ANALYSIS CACHE: {'HIT ✅' if cached_analysis else 'MISS ❌'}")
            logger.info(f" ANALYSIS PATH: {analysis_path}")
            
            return {
                "success": True,
                "response": final_response,
                "analysis": analysis,
                "tool_results": tool_results,
                "tools_used": analysis.get('tools_to_use', []),
                "execution_mode": execution_mode,
                "business_opportunity": analysis.get('business_opportunity', {}),
                "analysis_cache_hit": bool(cached_analysis),
                "analysis_path": analysis_path,
                "tools_cache_hit": False,  # Tools are always executed fresh
                "processing_time": {
                    "analysis": analysis_time,
                    "tools": tool_time,
                    "response": response_time,
                    "total": total_time
                },
                "llm_calls": llm_calls
            }
            
        except Exception as e:
            logger.error(f" Processing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": "I apologize, but I encountered an error. Please try again."
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

    async def _route_query(self, query: str, chat_history: List[Dict] = None, memories: str = "") -> Dict[str, Any]:
        """
        Router layer: Decide if query needs Chain-of-Thought reasoning (Qwen) or simple analysis (Llama)
        Uses Meta Llama 3.3 70B for fast, cost-effective routing decision
        """
        context = chat_history[-2:] if chat_history else []
        
        routing_prompt = f"""Analyze this query and decide if it needs deep Chain-of-Thought reasoning or simple analysis.

USER QUERY: {query}
LONG-TERM CONTEXT (Memories): {memories}
CONVERSATION HISTORY: {context}

Your task: Determine query complexity level.

NEEDS DEEP REASONING (Chain-of-Thought) when query involves:
- Multiple dimensions requiring expansion thinking
- Conditional logic with dependencies (if/then/else scenarios)
- Multi-task decomposition with sequential dependencies
- Ambiguous intent requiring deep semantic analysis
- Complex business opportunity assessment with nuanced signals
- Queries asking for comparisons, alternatives, or multi-angle exploration
- Strategic thinking or planning required

SIMPLE ANALYSIS when query involves:
- Greetings, casual conversation
- Single direct question with clear intent
- Simple fact retrieval or definition
- Obvious single tool selection
- Straightforward information request
- Already clear context, no ambiguity

Think about:
1. Does this query have hidden dimensions or is it straightforward?
2. Does answering this require exploring multiple angles?
3. Is the intent crystal clear or does it need interpretation?
4. Will simple pattern matching suffice or is reasoning needed?

Return ONLY valid JSON:
{{
  "needs_cot": true or false,
  "reasoning": "brief explanation of complexity assessment"
}}"""

        try:
            logger.info(f"🧭 ROUTING QUERY: '{query[:60]}...'")
            
            response = await self.router_llm.generate(
                messages=[{"role": "user", "content": routing_prompt}],
                system_prompt="You assess query complexity for routing. Return JSON only.",
                temperature=0.1,
                max_tokens=200
            )
            
            json_str = self._extract_json(response)
            routing_decision = json.loads(json_str)
            
            needs_cot = routing_decision.get('needs_cot', True)  # Default to safe path
            reasoning = routing_decision.get('reasoning', 'Routing decision made')
            
            logger.info(f"🧭 ROUTING DECISION: needs_cot={needs_cot}")
            logger.info(f"   Reason: {reasoning}")
            logger.info(f"   Path: {'COMPLEX (Qwen CoT)' if needs_cot else 'SIMPLE (Llama Fast)'}")
            
            return {
                "needs_cot": needs_cot,
                "reasoning": reasoning
            }
            
        except Exception as e:
            logger.error(f"❌ Routing failed: {e}, defaulting to CoT (safe path)")
            return {
                "needs_cot": True,  # Safe default
                "reasoning": f"Routing error: {str(e)}"
            }
    
    async def _simple_analysis(self, query: str, chat_history: List[Dict] = None, memories: str = "") -> Dict[str, Any]:
        """
        Fast-path analysis for simple queries using Meta Llama
        Returns same JSON structure as comprehensive analysis for compatibility
        """
        from datetime import datetime
        
        context = chat_history[-2:] if chat_history else []
        current_date = datetime.now().strftime("%B %d, %Y")
        
        analysis_prompt = f"""Analyze this query for Mochan-D - an AI chatbot that automates customer support across WhatsApp, Facebook, Instagram with RAG and web search capabilities.

USER QUERY: {query}
DATE: {current_date}
LONG-TERM CONTEXT (Memories): {memories}
CONVERSATION HISTORY: {context}

Available tools:
- web_search: Current internet information
- rag: Knowledge base retrieval
- calculator: Math operations

Your task: Provide structured analysis for this query.

ANALYZE:

1. What is the user asking for? (semantic intent)

2. Is this related to business communication challenges that AI chatbots solve?
   - Customer support automation needs
   - 24/7 availability requirements
   - Multi-platform management difficulties
   - Scaling communication challenges
   
   If yes, set business_opportunity.detected = true and calculate confidence (0-100)
   Consider: work context, emotional distress, solution-seeking behavior, scale/urgency

3. What tools are needed?
   - Use rag if: query about Mochan-D OR business opportunity detected
   - Use web_search if: needs current data, prices, comparisons, weather
   - Use calculator if: math operations needed
   - Use no tools if: greetings, casual chat, general knowledge

4. If multiple tools needed, can they run in parallel or must be sequential?
   - Parallel: Independent tasks
   - Sequential: One depends on another's output

5. What's the user's emotional state and how should we respond?
   - Emotion: frustrated/excited/casual/urgent/confused
   - Personality: empathetic_friend/excited_buddy/helpful_dost/urgent_solver/patient_guide
   - Length: micro/short/medium/detailed
   - Language: hinglish/english/professional/casual

6. Is this a follow-up query?
   - Look at conversation history: Does current query build on previous topics?
   - Follow-up = asking for details, clarification, or diving deeper into what was discussed
   - New query = completely different topic or no conversation history

Return ONLY valid JSON:
{{
  "multi_task_analysis": {{
    "multi_task_detected": true or false,
    "sub_tasks": ["task 1", "task 2"]
  }},
  "is_follow_up": true or false,
  "semantic_intent": "what user wants",
  "expansion_reasoning": "kept simple - straightforward query",
  "business_opportunity": {{
    "detected": true or false,
    "composite_confidence": 0-100,
    "engagement_level": "direct_consultation|gentle_suggestion|empathetic_probing|pure_empathy",
    "signal_breakdown": {{
      "work_context": 0-100,
      "emotional_distress": 0-100,
      "solution_seeking": 0-100,
      "scale_scope": 0-100
    }},
    "recommended_approach": "empathy_first|solution_focused|consultation_ready",
    "pain_points": ["problem 1", "problem 2"],
    "solution_areas": ["how Mochan-D helps"]
  }},
  "tools_to_use": ["tool1", "tool2"],
  "tool_execution": {{
    "mode": "sequential|parallel",
    "order": ["tool1_0", "tool2_0"],
    "dependency_reason": "reason if sequential"
  }},
  "enhanced_queries": {{
    "rag_0": "query for rag",
    "web_search_0": "focused search query",
    "calculator_0": "math expression"
  }},
  "tool_reasoning": "why these tools selected",
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
            logger.info(f"💨 SIMPLE ANALYSIS (Llama Fast Path)")
            
            response = await self.router_llm.generate(
                messages=[{"role": "user", "content": analysis_prompt}],
                system_prompt=f"You analyze queries as of {current_date}. Return valid JSON only.",
                temperature=0.1,
                max_tokens=4000
            )
            
            json_str = self._extract_json(response)
            result = json.loads(json_str)
            
            logger.info(f"✅ Simple analysis complete: {result.get('semantic_intent', 'N/A')[:100]}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Simple analysis JSON parse error: {e}")
            return self._get_fallback_analysis(query)
    
    def _get_fallback_analysis(self, query: str) -> Dict[str, Any]:
        """Fallback analysis structure when parsing fails"""
        return {
            "multi_task_analysis": {"multi_task_detected": False, "sub_tasks": []},
            "semantic_intent": query,
            "expansion_reasoning": "Fallback due to parse error",
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
                "recommended_approach": "empathy_first",
                "pain_points": [],
                "solution_areas": []
            },
            "tools_to_use": [],
            "tool_execution": {"mode": "parallel", "order": [], "dependency_reason": ""},
            "enhanced_queries": {},
            "tool_reasoning": "Direct response needed",
            "sentiment": {"primary_emotion": "casual", "intensity": "medium"},
            "response_strategy": {
                "personality": "helpful_dost",
                "length": "medium",
                "language": "hinglish",
                "tone": "friendly"
            }
        }

    async def _comprehensive_analysis(self, query: str, chat_history: List[Dict] = None, memories:str = "") -> Dict[str, Any]:
        """
        Core semantic analysis + tool selection
        Returns structured analysis with tool execution plan
        """
        from datetime import datetime
        
        context = chat_history[-5:] if chat_history else []
        current_date = datetime.now().strftime("%B %d, %Y")
        
        analysis_prompt = f"""You are analyzing a user query for Mochan-D as of {current_date} - an AI chatbot that automates customer support across WhatsApp, Facebook, Instagram with RAG and web search capabilities.

Available tools:
- web_search: Current internet data
- rag: Knowledge base retrieval  
- calculator: Math operations

USER QUERY: {query}

LONG-TERM CONTEXT (Memories): {memories}
CONVERSATION HISTORY: {context}

CRITICAL INSTRUCTION - DATA FRESHNESS:
- Any information that is liable to change, USE web-search to validate that. For standard definitions and facts, use your base data. Based on that, expand on the dimensionality aspect to retrieve all that information at once.
- Think deeply for every possibilities do not leave things by assuming anything
CORE PRINCIPLE: Think like a world-class consultant.
When someone asks for X, you don't just give X. You think: "What else do they need to make X truly successful?"

Your superpower: MULTI-DIMENSIONAL REASONING
- User mentions restaurant recommendations → Think: What about parking? Dietary restrictions? Price range?
- User asks for laptop → Think: What about accessories? Software? Warranty options?
- User wants recipe → Think: What about substitutes? Cooking tips? Storage instructions?

THINK THROUGH THESE QUESTIONS (use your intelligence, not rules):

1. WHAT DOES THE USER REALLY WANT?
   - Look beyond the literal words - what's their actual goal?
   - What emotional state are they in?
   - Is this one request or multiple separate things?

2. INFORMATION QUALITY CHECK - THINK BEYOND THE OBVIOUS
   Ask yourself repeatedly: "What am I missing?"
   
   - If I answer just what they asked, will it be complete?
   - What did the user NOT mention but would obviously need?
   - What alternatives or related options should they consider?
   - What context or background would make this more valuable?
   
   MULTI-DIMENSIONAL THINKING:
   Don't just answer the literal question. Think about:
   - WHAT they asked for (explicit need)
   - WHAT they forgot to ask (implicit need)
   - WHAT alternatives exist (options they should know about)
   - WHAT context matters (surrounding information)
   
   Mental process training:
   User says: "best laptop for video editing"
   Your thinking: "They said video editing... but they'll also need: storage solutions (external drives),
   editing software recommendations, color-accurate monitors, backup strategies. That's 5 dimensions:
   laptop specs + storage + software + display + backup. Each needs separate focused research."
   
   Use this expansion mindset for EVERY query.

3. IS THIS A BUSINESS PROBLEM?
   Think naturally: Does this query relate to challenges that an AI chatbot could solve?
   - Customer communication problems?
   - Need for automation or always-available support?
   - Managing multiple platforms or scaling interactions?
   
   If yes → this is a business context (you should include rag to provide Mochan-D context)
   If no → just answer the query directly

4. MULTI-DIMENSIONAL TASK BREAKDOWN - FIND ALL THE HIDDEN ANGLES
   
   Your job: Identify EVERY dimension of this query, including what user didn't explicitly say.
   
   CRITICAL MINDSET: When you think you have enough searches, DOUBLE IT.
   Most people under-search. You're smarter than that.
   
   Step 1: What did they LITERALLY ask for?
   Step 2: What did they IMPLY but not say?
   Step 3: What ALTERNATIVES should they know about?
   Step 4: What RELATED INFORMATION would be valuable?
   Step 5: What would a world-class expert include that others miss?
   
   Mental exercise for EVERY query:
   - If they mention ONE audience, are there OTHER audiences? (Create separate search for EACH)
   - If they ask for ONE thing, what RELATED things do they need? (Separate search for EACH)
   - If they want X, should they also know about Y and Z? (Separate search for EACH)
   - What examples would make this concrete? (Separate search)
   - What data would make this credible? (Separate search)
   - What best practices exist? (Separate search)
   - What alternatives or comparisons? (Separate search)
   
   RULE: Create a SEPARATE search for EACH dimension you discover.
   Don't merge dimensions - keep each one focused and distinct.
   If you're generating less than 5 searches for a complex query, you're missing dimensions.

5. HOW TO FORMAT YOUR QUERIES (CRITICAL):
   
   For web_search queries:
   - Write like you're typing into Google: SHORT, keyword-focused
   - Keep it under 6-8 words maximum
   - Focus on core terms only
   - Include year (2025) for time-sensitive topics
   
   For rag queries:
   - Natural language is OK: "product features value proposition"
   - You're searching internal documents

6. TOOL ORCHESTRATION - CAN DIFFERENT TOOLS RUN TOGETHER?
   
   Think about dependencies BETWEEN tool types (not within same tool type):
   
   Ask yourself: "Does one tool type NEED results from another tool type to work properly?"
   
   - Does web_search need rag data first to search effectively? → sequential
   - Does rag need web_search results to query properly? → sequential  
   - Can they work independently with just the user's query? → parallel
   
   Default to PARALLEL unless there's a clear logical dependency.
   
   Note: All web_search queries always run parallel among themselves.
   This is only about cross-tool dependencies (rag ↔ web_search ↔ calculator)

7. HOW SHOULD THE RESPONSE FEEL?
   Based on the user's tone and needs:
   - What personality would work best? (empathetic, professional, casual, excited, urgent)
   - How much detail do they need? (brief, moderate, comprehensive)
   - What language style fits? (formal english, casual english, hinglish)

8. IS THIS A FOLLOW-UP QUERY?
   Look at CONVERSATION HISTORY above:
   - Does the current query build on previous topics discussed?
   - Is user asking for details, clarification, or diving deeper into what was already talked about?
   - Or is this a completely new topic/question?
   Set is_follow_up to true only if genuinely continuing previous conversation.

FINAL CHECK BEFORE YOU OUTPUT:
- Did I find ALL dimensions of this query?
- Am I being generous with search count or conservative? (Be generous!)
- Did I use proper key names? (rag_0, web_search_0, web_search_1, etc.)
- For complex queries: Did I generate at least 5-7 searches?
- Did I keep the EXACT JSON structure below?

OUTPUT THIS EXACT JSON STRUCTURE:

{{
  "multi_task_analysis": {{
    "multi_task_detected": true or false,
    "sub_tasks": ["description of task 1", "description of task 2"]
  }},
  "is_follow_up": true or false,
  "semantic_intent": "clear description of overall user goal",
  "expansion_reasoning": "your thought process why keeping simple OR why adding more searches",
  "business_opportunity": {{
    "detected": true or false,
    "composite_confidence": 0-100,
    "engagement_level": "direct_consultation|gentle_suggestion|empathetic_probing|pure_empathy",
    "signal_breakdown": {{
      "work_context": 0-100,
      "emotional_distress": 0-100,
      "solution_seeking": 0-100,
      "scale_scope": 0-100
    }},
    "recommended_approach": "empathy_first|solution_focused|consultation_ready",
    "pain_points": ["specific problem 1", "specific problem 2"],
    "solution_areas": ["how Mochan-D helps 1", "solution 2"]
  }},
  "tools_to_use": ["tool1", "tool2"],
  "tool_execution": {{
    "mode": "sequential|parallel",
    "order": ["tool1", "tool2"],
    "dependency_reason": "why sequential is needed or empty if parallel"
  }},
  "enhanced_queries": {{
    "rag_0": "query for rag",
    "web_search_0": "first focused search",
    "web_search_1": "second focused search"
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
}}

Now analyze: {query}

Think through each question naturally, then return ONLY the JSON. No other text."""

        try:
            # messages = []
            
            # Simple system prompt for thinking models
            system_prompt = f"""You are analyzing queries as of {current_date}. Think step by step, then output valid JSON only."""
            
            # messages.append({"role": "system", "content": system_prompt})
            # messages.append({"role": "user", "content": analysis_prompt})
            
            response = await self.brain_llm.generate(
                messages=[{"role": "user", "content": analysis_prompt}],
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=16000
            )
            
            json_str = self._extract_json(response)
            result = json.loads(json_str)
            
            logging.info(f"✅ Analysis complete: {result.get('semantic_intent', 'N/A')[:100]}")
            return result
            
        except json.JSONDecodeError as e:
            logging.error(f"❌ JSON parse error: {e}")
            logging.error(f"Response snippet: {response[:500] if response else 'No response'}")
            return self._get_fallback_analysis(query)
 
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
        
        logger.info(f"Enhanced queries for parallel execution: {enhanced_queries}")
        
        # Check if LLMLayer is enabled and merge web_search queries
        llmlayer_enabled = os.getenv('LLMLAYER_ENABLED', 'false').lower() == 'true'
        
        if llmlayer_enabled and 'web_search' in tools:
            # Get all web_search queries
            web_queries = [v for k, v in enhanced_queries.items() if k.startswith("web_search")]
            
            if len(web_queries) > 1:
                # Merge queries with comma separator
                merged_query = ", ".join(web_queries)
                logger.info(f"🔀 LLMLayer enabled: Merging {len(web_queries)} web_search queries")
                logger.info(f"   Combined query: {merged_query[:150]}...")
                
                # Replace all web_search queries with single merged one
                new_queries = {k: v for k, v in enhanced_queries.items() if not k.startswith("web_search")}
                new_queries["web_search_0"] = merged_query
                enhanced_queries = new_queries
        
        # Execute tools in parallel for speed
        tasks = []
        tool_counter = {}  # Track occurrences of each tool type
        
        for i, tool in enumerate(tools):
            if tool in self.available_tools:
                # Count tool occurrences for unique keys
                count = tool_counter.get(tool, 0)
                tool_counter[tool] = count + 1
                
                # FIXED: Use tool-specific counter, not array index
                # This matches how LLM generates indexed keys (web_search_0, web_search_1 per tool type)
                indexed_key = f"{tool}_{count}"
                tool_query = enhanced_queries.get(indexed_key) or enhanced_queries.get(tool, query)
                
                logger.info(f"🔧 {tool.upper()} #{count} ENHANCED QUERY: '{tool_query}'")
                
                # Default scraping for web_search tools (always use 3 pages)
                scrape_count = 3 if tool == 'web_search' else None
                
                # FIXED: Store results with tool-type counter (web_search_0, web_search_1, etc.)
                # Always use indexed key format for consistency
                result_key = indexed_key
                
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
        tool_execution = analysis.get('tool_execution', {})
        order = tool_execution.get('order', tools)
        
        logger.info(f"   Execution order: {order}")
        logger.info(f"   Reason: {tool_execution.get('dependency_reason', 'N/A')}")
        
        # Execute first tool
        first_tool_key = order[0]  # e.g., 'web_search_0'
        first_tool_name = first_tool_key.rsplit('_', 1)[0] if '_' in first_tool_key and first_tool_key.split('_')[-1].isdigit() else first_tool_key
        # ^ Strips index: 'web_search_0' -> 'web_search'
        
        first_query = enhanced_queries.get(first_tool_key, query)
        logger.info(f"   → Step 1: Executing {first_tool_key.upper()} with query: '{first_query}'")
        
        # Default scraping for web_search (always use 3 pages)
        first_tool_kwargs = {"query": first_query, "user_id": user_id}
        if first_tool_name == 'web_search':
            first_tool_kwargs["scrape_top"] = 3
        
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
            
            # Default scraping for web_search (always use 3 pages)
            current_tool_kwargs = {"query": enhanced_query, "user_id": user_id}
            if current_tool_name == 'web_search':
                current_tool_kwargs["scrape_top"] = 3
            
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

    
    async def _generate_response(self, query: str, analysis: Dict, tool_results: Dict, chat_history: List[Dict], memories:str="", mode:str="", source: str = "whatsapp") -> str:
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
        
        if mode == "transformative":
            response_prompt = f"""You are a helpful AI assistant. Your purpose is to provide accurate, comprehensive, and useful responses.

            ORIGINAL USER QUERY: {query}
            UNDERSTOOD AS: {intent}

            (When these differ, pay attention to the original wording - it shows what the user is specifically referring to)

            CRITICAL - DATA USAGE:
            - RAG results = User's uploaded documents (PRIMARY SOURCE - use these fully)
            - WEB_SEARCH results = Current internet information
            - Your training knowledge = Use ONLY as backup when no data provided

            ALWAYS prioritize the provided data as your source of truth.

            AVAILABLE DATA:
            {tool_data}

            TASK HANDLING INSTRUCTIONS:

            When user asks you to SUMMARIZE:
            - Read ALL the provided data carefully
            - Extract every key point, main concept, and important detail
            - Structure your summary logically: Overview → Main Points → Key Takeaways
            - Be comprehensive - cover all major aspects, not just 2-3 lines
            - Don't skip sections or rush through content

            When user asks you to IMPROVE/REVIEW/CRITIQUE:
            - Analyze the ACTUAL content from the data provided
            - Identify what's present and what's missing
            - Give SPECIFIC suggestions with concrete examples
            - Point to exact sections that need changes
            - Structure: Current State → Issues Found → Specific Improvements
            - Don't give generic advice - be actionable and detailed

            When user asks you to ANALYZE/COMPARE:
            - Break down information systematically
            - Compare different aspects when multiple sources exist
            - Provide insights and draw connections, not just facts
            - Highlight patterns, similarities, and differences

            When user asks QUESTIONS:
            - Answer directly and thoroughly using the data
            - Support your answer with evidence from the data
            - Be complete but stay focused on what was asked

            For GENERAL QUERIES:
            - Be helpful, clear, and informative
            - Use available data naturally in your response
            - Match the depth and detail user expects

            RESPONSE REQUIREMENTS:

            LENGTH: Be comprehensive and thorough
            - Don't artificially limit your response
            - If summarizing long documents, give a complete summary
            - If analyzing content, cover all relevant angles
            - If providing suggestions, list everything important

            STRUCTURE: Use clear formatting
            - Headers and subheaders for organization
            - Bullet points for lists and key points
            - Examples and specifics when helpful
            - Logical flow from start to finish

            TONE: Professional yet approachable
            - Clear and easy to understand
            - Friendly but focused
            - Helpful without being condescending

            CRITICAL RULES:
            1. USE ALL relevant data provided (especially RAG content - it's what user uploaded)
            2. DO the complete task (no partial answers or cutting it short)
            3. BE SPECIFIC - cite actual content, give concrete examples
            4. NO selling or pitching anything
            5. DON'T skip content because of length - be thorough

            USER QUERY: {query}

            Provide your comprehensive response now:"""
        else:
        
            response_prompt = f"""You are Mochan-D (Mochand Dost) - an AI companion who's equal parts:
            - Helpful friend (dost) who genuinely cares
            - Smart business consultant who spots opportunities  
            - Natural conversationalist who builds relationships
            - Clever sales agent who never feels pushy
            - Analytical problem-solver who adds real value
            - Structure your responses such that they answer the user's query fully while keeping it short and concise.
            - For complex queries, break down your response into clear sections with headers and bullet points.
            - Keep your response under 200 characters.

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
            
            TASK: When the user's query is an explicit action (summarize, extract, analyze, compare), DO THAT TASK using the available data. Don't ask for clarification on what to do - do it naturally.
            
            AVAILABLE DATA TO USE NATURALLY:
            {tool_data}

            RESPONSE REQUIREMENTS:
            - Personality: {strategy.get('personality', 'helpful_dost')}
            - Length: {strategy.get('length', 'medium')} 
            - Language: {strategy.get('language', 'hinglish')}
            - Tone: {strategy.get('tone', 'friendly')}

            🎯 RESPONSE RULES:

            CORE PRINCIPLES:
            1. Start with value, not preamble. Jump directly into insights without any conversational setup.
            2. NEVER begin your response by restating, echoing, or mentioning what the user asked about. Go straight to the substantive information.
            3. NEVER announce tool usage ("Let me search...", "I found...")
            4. Match emotional energy PRECISELY using sentiment guide
            5. Stay in character as their dost

            OPENING LINE RULES:
            - DO: Start with direct insights, facts, or analysis
            - DON'T: Reference the user's query topic in your first sentence
            - The user knows what they asked - deliver the answer immediately
            
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

            {'WHATSAPP CONTEXT: You are communicating via WhatsApp where brevity is essential for mobile engagement. ' + ('This is a FOLLOW-UP query - user wants depth on previous discussion. Provide 350-450 character response with comprehensive insights, examples, and actionable details. Use the space fully.' if analysis.get('is_follow_up', False) else 'This is an INITIAL query - create engagement spark. Compress response to 200-250 characters maximum - deliver the most critical insight that invites further conversation. Platform constraints override data volume.') if source == 'whatsapp' else ''}

            NOW RESPOND as Mochand Dost in {conversation_mode} mode. Be natural, helpful, strategic, human. If business opportunity exists, weave it like a skilled storyteller - make them see value without feeling sold to. If casual chat, be the best dost ever.
            Remember: You're building relationships that could turn into business. Play it smart, smooth, genuine."""
        
        
        try:
            
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
                max_tokens=4000 if mode == 'transformative' else max_tokens,
                system_prompt="Answer the query based on the provided context and data."
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
                # Check if LLMLayer or Perplexity (pre-formatted responses)
                if result.get('provider') in ['llmlayer', 'perplexity'] and 'llm_response' in result:
                    provider_name = result.get('provider', '').upper()
                    logger.info(f" {provider_name} pre-formatted response detected")
                    formatted.append(f"{tool.upper()} ({provider_name}):\n{result['llm_response']}\n")
                    continue  # Skip scraping logic
                
                # Handle RAG-style result
                if "success" in result and result["success"]:
                    logger.info(f" Formatting result for tool: {result}")
                    if "retrieved" in result:
                        retrieved = result.get("retrieved", "")
                        chunks = result.get("chunks", [])
                        formatted.append(f"{tool.upper()} RETRIEVED TEXT:\n{retrieved}\n")

                        if chunks:
                            # Normalize chunks into readable strings
                            formatted_chunks = []
                            for c in chunks:
                                if isinstance(c, str):
                                    formatted_chunks.append(c)
                                elif isinstance(c, dict):
                                    doc = c.get("document", "")
                                    filename = c.get("metadata", {}).get("filename", "unknown file")
                                    distance = c.get("distance", None)
                                    info_line = f"[{filename}] (distance={distance:.4f})" if distance is not None else f"[{filename}]"
                                    formatted_chunks.append(f"{info_line}\n{doc}")
                                else:
                                    formatted_chunks.append(str(c))  # fallback for unexpected types

                            formatted.append(f"{tool.upper()} CHUNKS:\n" + "\n---\n".join(formatted_chunks))

                    
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
    
    def _extract_json(self, response: str) -> str:
        """Extract JSON from LLM response (handles thinking models)"""
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
