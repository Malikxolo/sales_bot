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
from .config import AddBackgroundTask, memory_config
from .redis_manager import RedisCacheManager

logger = logging.getLogger(__name__)



class CustomerSupportAgent:
    """Simplified agent for customer support with minimal LLM calls"""
    
    def __init__(self, brain_llm, heart_llm, tool_manager):
        self.brain_llm = brain_llm  # For analysis
        self.heart_llm = heart_llm  # For response generation
        self.tool_manager = tool_manager
        self.available_tools = tool_manager.get_available_tools()
        self.memory = AsyncMemory(memory_config)
        self.task_queue: asyncio.Queue["AddBackgroundTask"] = asyncio.Queue()
        self._worker_started = False
        self.cache_manager = RedisCacheManager()
        
        logger.info(f"CustomerSupportAgent initialized with tools: {self.available_tools}")
        logger.info(f"Redis caching: {'ENABLED ✅' if self.cache_manager.enabled else 'DISABLED ⚠️'}")
    
    async def process_query(self, query: str, chat_history: List[Dict] = None, user_id: str = None) -> Dict[str, Any]:
        """Process customer query with minimal LLM calls and caching"""
        self._start_worker_if_needed()
        logger.info(f"🔵 PROCESSING QUERY: '{query}'")
        start_time = datetime.now()
        
        cached_analysis = None
        analysis = None
        analysis_time = 0.0
        
        try:
            # STEP 1: Check cache or analyze
            cached_analysis = await self.cache_manager.get_cached_query(query, user_id)
            
            if cached_analysis:
                logger.info(f"🎯 USING CACHED ANALYSIS")
                analysis = cached_analysis
                analysis_time = 0.0
            else:
                # Retrieve conversation context
                memory_results = await self.memory.search(query[:100], user_id=user_id, limit=5)
                memories = "\n".join([
                    f"- {item['memory']}" 
                    for item in memory_results.get("results", []) 
                    if item.get("memory")
                ]) or "No previous context."

                logger.info(f"🧠 Retrieved memories: {len(memories)} chars")
                
                # Analyze query
                analysis_start = datetime.now()
                analysis = await self._analyze_query(query, chat_history, memories)
                analysis_time = (datetime.now() - analysis_start).total_seconds()
                
                # Cache the analysis
                await self.cache_manager.cache_query(query, analysis, user_id, ttl=3600)
            
            # Log analysis results
            logger.info(f"📊 ANALYSIS RESULTS:")
            logger.info(f"   Intent: {analysis.get('intent', 'Unknown')}")
            logger.info(f"   Sentiment: {analysis.get('sentiment', {}).get('emotion', 'neutral')}")
            logger.info(f"   Tools Selected: {analysis.get('tools_to_use', [])}")
            
            # STEP 2: Execute tools if needed
            tools_to_use = analysis.get('tools_to_use', [])
            tool_start = datetime.now()
            tool_results = await self._execute_tools(tools_to_use, query, analysis, user_id)
            tool_time = (datetime.now() - tool_start).total_seconds()
            
            # STEP 3: Generate response
            response_start = datetime.now()
            if not cached_analysis:
                memory_results = await self.memory.search(query, user_id=user_id, limit=5)
                memories = "\n".join([
                    f"- {item['memory']}" 
                    for item in memory_results.get("results", []) 
                    if item.get("memory")
                ]) or "No previous context."
            
            final_response = await self._generate_response(
                query, analysis, tool_results, chat_history, memories
            )
            
            # Store conversation in memory (background task)
            await self.task_queue.put(
                AddBackgroundTask(
                    func=partial(self.memory.add),
                    params=(
                        [
                            {"role": "user", "content": query}, 
                            {"role": "assistant", "content": final_response}
                        ],
                        user_id,
                    ),
                )
            )
            
            response_time = (datetime.now() - response_start).total_seconds()
            total_time = (datetime.now() - start_time).total_seconds()
            
            # Count LLM calls
            llm_calls = 1 if cached_analysis else 2  # Analysis + Response
            
            logger.info(f"✅ COMPLETED in {total_time:.2f}s ({llm_calls} LLM calls)")
            
            return {
                "success": True,
                "response": final_response,
                "analysis": analysis,
                "tool_results": tool_results,
                "tools_used": tools_to_use,
                "cache_hit": bool(cached_analysis),
                "processing_time": {
                    "analysis": analysis_time,
                    "tools": tool_time,
                    "response": response_time,
                    "total": total_time
                },
                "llm_calls": llm_calls
            }
            
        except Exception as e:
            logger.error(f"❌ Processing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": "I apologize, but I encountered an error. Please try again."
            }
    
    async def _analyze_query(self, query: str, chat_history: List[Dict] = None, memories: str = "") -> Dict[str, Any]:
        """Analyze customer query to understand intent and select appropriate tools"""
        from datetime import datetime
        
        context = chat_history[-5:] if chat_history else []
        current_date = datetime.now().strftime("%B %d, %Y")
        
        analysis_prompt = f"""You are analyzing a customer support query as of {current_date}.

        Available tools:
        - live_information: Search for current information (regarding order status, product availability, etc.)
        - knowledge_base: Search internal documentation and FAQs
        - raise_ticket: Raise a support ticket in the system
        - assign_agent: Assign a human agent for follow-up

        USER QUERY: {query}
        CONVERSATION HISTORY: {context}
        PREVIOUS CONTEXT: {memories}

        Analyze this query and provide:

        1. CUSTOMER INTENT: What does the customer need?
        - Information request
        - Problem resolution
        - Product inquiry
        - Account assistance
        - Complaint/feedback

        2. SENTIMENT ANALYSIS:
        - Emotion: frustrated, satisfied, confused, urgent, neutral
        - Intensity: low, medium, high
        - Urgency level: low, medium, high

        3. TOOL SELECTION:
        Select tools needed to help the customer:
            - live_information: Use FIRST to check user's past order records. Only ask for order ID if information is not found.
            - knowledge_base: Search internal documentation and FAQs
            - raise_ticket: Raise a support ticket in the system
            - assign_agent: ONLY call when BOTH conditions are met:
                a) The damaged good/service is perishable
                b) The user is asking for refund OR replacement
        
        Use NO tools for: greetings, simple acknowledgments, general chitchat

        IMPORTANT: For physically damaged goods, request a picture as proof before proceeding.

        4. QUERY ENHANCEMENT:
        For each selected tool, create a focused query

        5. RESPONSE STRATEGY:
        - Tone: empathetic, professional, friendly, solution-focused
        - Length: brief, moderate, detailed
        - Priority: address emotion first, then solution

        Return ONLY valid JSON:
        {{
        "intent": "what customer needs",
        "is_follow_up": true or false,
        "requires_proof_picture": true or false,
        "sentiment": {{
            "emotion": "frustrated|satisfied|confused|urgent|neutral",
            "intensity": "low|medium|high",
            "urgency": "low|medium|high"
        }},
        "requires_proof_picture": true or false,
        "tools_to_use": ["tool1", "tool2"],
        "enhanced_queries": {{
            "live_information_0": "query for live information",
            "knowledge_base_0": "query for knowledge base",
            "raise_ticket_0": "ticket details",
            "assign_agent_0": "reason for agent assignment"
        }},
        "response_strategy": {{
            "tone": "empathetic|professional|friendly|solution-focused",
            "length": "brief|moderate|detailed",
            "priority": "emotion_first|solution_first|information_first"
        }},
        "key_points": ["point1", "point2"]
        }}"""

        try:
            response = await self.brain_llm.generate(
                messages=[{"role": "user", "content": analysis_prompt}],
                system_prompt=f"You analyze customer queries as of {current_date}. Return JSON only.",
                temperature=0.1,
                max_tokens=2000
            )
            
            json_str = self._extract_json(response)
            result = json.loads(json_str)
            
            logger.info(f"✅ Analysis complete")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parse error: {e}")
            return self._get_fallback_analysis(query)
    
    def _get_fallback_analysis(self, query: str) -> Dict[str, Any]:
        """Fallback analysis when parsing fails"""
        return {
            "intent": query,
            "is_follow_up": False,
            "sentiment": {
                "emotion": "neutral",
                "intensity": "medium",
                "urgency": "medium"
            },
            "tools_to_use": [],
            "enhanced_queries": {},
            "response_strategy": {
                "tone": "professional",
                "length": "moderate",
                "priority": "solution_first"
            },
            "key_points": []
        }

    async def _execute_tools(self, tools: List[str], query: str, analysis: Dict, user_id: str = None) -> Dict[str, Any]:
        """Execute tools in parallel"""
        if not tools:
            return {}
        
        results = {}
        enhanced_queries = analysis.get('enhanced_queries', {})
        
        # Execute all tools in parallel
        tasks = []
        tool_counter = {}
        
        for tool in tools:
            if tool in self.available_tools:
                count = tool_counter.get(tool, 0)
                tool_counter[tool] = count + 1
                
                indexed_key = f"{tool}_{count}"
                tool_query = enhanced_queries.get(indexed_key, query)
                
                logger.info(f"🔧 Executing {indexed_key}: '{tool_query}'")
                
                task = self.tool_manager.execute_tool(tool, query=tool_query, user_id=user_id)
                tasks.append((indexed_key, task))
        
        # Gather results
        for tool_name, task in tasks:
            try:
                result = await task
                results[tool_name] = result
                logger.info(f"✅ {tool_name} completed")
            except Exception as e:
                logger.error(f"❌ {tool_name} failed: {e}")
                results[tool_name] = {"error": str(e)}
        
        return results
    
    async def _generate_response(self, query: str, analysis: Dict, tool_results: Dict, 
                                 chat_history: List[Dict], memories: str = "") -> str:
        """Generate customer support response"""
        
        intent = analysis.get('intent', '')
        sentiment = analysis.get('sentiment', {})
        strategy = analysis.get('response_strategy', {})
        
        # Format tool results
        tool_data = self._format_tool_results(tool_results)
        context = chat_history[-5:] if chat_history else []
        
        response_prompt = f"""You are a helpful customer support assistant.

        CUSTOMER QUERY: {query}
        UNDERSTOOD INTENT: {intent}

        CUSTOMER SENTIMENT:
        - Emotion: {sentiment.get('emotion', 'neutral')}
        - Urgency: {sentiment.get('urgency', 'medium')}

        AVAILABLE INFORMATION:
        {tool_data}

        CONVERSATION HISTORY: {context}
        PREVIOUS CONTEXT: {memories}

        RESPONSE GUIDELINES:

        1. TONE: {strategy.get('tone', 'professional')}
        - Be empathetic and understanding
        - Acknowledge their concern/question
        - Use clear, friendly language

        2. STRUCTURE:
        - Start with acknowledgment (1 sentence max)
        - DO NOT start the sentence with what the customer said
        - Provide clear solution/information (direct and focused)
        - Offer additional help only if necessary (1 sentence max)

        3. SENTIMENT HANDLING:
        - frustrated/urgent: Skip pleasantries, go straight to the solution
        - confused: Give step-by-step guidance, no extra explanations
        - satisfied: Brief, warm acknowledgment
        - neutral: Direct and informative

        4. LENGTH: 
        - Maximum 3-4 sentences total
        - Get to the point immediately
        - Cut all filler words and unnecessary explanations
        - If multiple steps are needed, use a brief numbered list
        

        5. CRITICAL RULES:
        - Use the provided tool data as your source of truth
        - Be accurate - don't make up information
        - If you don't have information, say so in one sentence and suggest next steps
        - Always aim to help and resolve the customer's issue
        - Prioritize clarity over friendliness - be helpful, not chatty

        6. SPECIAL HANDLING BASED ON ANALYSIS:
        
        PROOF OF PICTURE REQUIRED: {analysis.get('requires_proof_picture', False)}
        - If true and no picture provided yet: Ask the customer to provide a picture of the damaged item before proceeding
        - If true and picture already provided: Acknowledge receipt and proceed with solution
        
        ORDER INFORMATION:
        - If live_information was used: The tool_data contains order history - use this information directly
        - If order info is missing from tool_data: Politely ask for the order ID to look up details
        - Never ask for order ID if it's already in the tool_data from live_information
        
        AGENT ASSIGNMENT:
        - If assign_agent tool was triggered: Inform customer that a specialist agent will follow up
        - Only mention agent assignment if it appears in the tools_to_use list
        - For perishable damaged goods with refund/replacement requests: Confirm agent assignment for priority handling

        TICKET RAISED:
        - If raise_ticket tool was used: Provide the ticket reference if available in tool_data
        - Assure customer their issue is being tracked

        CUSTOMER QUERY: {query}

        Provide a concise, direct response now (3-4 sentences maximum):"""

        try:
            max_tokens = {
                "brief": 150,
                "moderate": 300,
                "detailed": 500
            }.get(strategy.get('length', 'moderate'), 300)
            
            messages = chat_history[-4:] if chat_history else []
            messages.append({"role": "user", "content": response_prompt})
            
            response = await self.heart_llm.generate(
                messages,
                temperature=0.4,
                max_tokens=max_tokens,
                system_prompt="You are a helpful customer support assistant."
            )
            
            response = self._clean_response(response)
            logger.info(f"✅ Response generated: {len(response)} chars")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Response generation failed: {e}")
            return "I apologize, but I had trouble generating a response. Please try again."
    
    def _format_tool_results(self, tool_results: dict) -> str:
        """Format tool results for response generation"""
        if not tool_results:
            return "No additional data available"
        
        formatted = []
        
        for tool, result in tool_results.items():
            if isinstance(result, dict) and 'error' not in result:
                if "success" in result and result["success"]:
                    # Knowledge base results
                    if "retrieved" in result:
                        formatted.append(f"{tool.upper()}:\n{result.get('retrieved', '')}\n")
                    
                    # Web search results
                    elif 'results' in result and isinstance(result['results'], list):
                        formatted.append(f"{tool.upper()} RESULTS:\n")
                        for item in result['results'][:3]:
                            title = item.get('title', '')
                            snippet = item.get('snippet', '')
                            formatted.append(f"- {title}\n  {snippet}\n")
                    
                    # Calculator or other results
                    elif 'result' in result:
                        formatted.append(f"{tool.upper()}: {result['result']}")
        
        return "\n".join(formatted) if formatted else "No usable data"

    def _extract_json(self, response: str) -> str:
        """Extract JSON from LLM response"""
        response = response.strip()
        
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
        
        # Remove any markdown artifacts
        if response.startswith('```'):
            lines = response.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            response = '\n'.join(lines)
        
        return response.strip()
    
    async def background_task_worker(self) -> None:
        """Process background tasks like memory storage"""
        while True:
            task: AddBackgroundTask = await self.task_queue.get()
            try:
                messages, user_id = task.params
                await task.func(messages=messages, user_id=user_id)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background task error: {e}")
            finally:
                self.task_queue.task_done()
    
    def _start_worker_if_needed(self):
        """Start background worker once"""
        if not self._worker_started:
            asyncio.create_task(self.background_task_worker())
            self._worker_started = True
            logger.info("✅ Background worker started")