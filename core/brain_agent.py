"""
Brain Agent - Pure LLM-driven orchestrator
ENHANCED with comprehensive logging and Business Opportunity Detection
"""

import asyncio
import json
import sqlite3
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from .llm_client import LLMClient
from .tools import ToolManager
from .exceptions import BrainAgentError

# Setup logger
logger = logging.getLogger(__name__)

def clean_llm_response(response: str) -> str:
    """Clean LLM response by removing markdown blocks and thinking tags"""
    cleaned_response = response.strip()
    
    # Remove thinking tags first
    if '<think>' in cleaned_response and '</think>' in cleaned_response:
        end_tag = cleaned_response.find('</think>')
        if end_tag != -1:
            cleaned_response = cleaned_response[end_tag + 8:].strip()
    
    # Remove markdown code blocks
    backticks = '`' * 3
    if cleaned_response.startswith(backticks):
        lines = cleaned_response.split('\n')
        lines = lines[1:]  # Remove first line with ```
        if lines and lines[-1].strip() == backticks:
            lines = lines[:-1]  # Remove closing ```
        cleaned_response = '\n'.join(lines)
    
    return cleaned_response

class BusinessOpportunityDetector:
    """Business opportunity detection system integrated into Brain Agent"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        logger.info("BusinessOpportunityDetector initialized")
    
    async def analyze_opportunity(self, query: str, chat_history: List[Dict]) -> Dict[str, Any]:
        """Detect business opportunities in user queries with high accuracy"""
        
        logger.info(f"🔍 Analyzing business opportunity for: '{query[:50]}...'")
        
        # Create context from recent chat history
        context_str = ""
        if chat_history:
            recent_context = chat_history[-3:]  # Last 3 interactions
            context_str = "\n".join([f"User: {item.get('query', '')}" for item in recent_context])
        
        opportunity_prompt = f"""Analyze this query for BUSINESS opportunities only:

USER QUERY: {query}
RECENT CONTEXT: {context_str if context_str else "No prior context"}

BUSINESS PAIN INDICATORS TO DETECT:
✅ Customer support problems/bottlenecks
✅ Manual business processes taking time  
✅ Inventory/supply chain management issues
✅ Business communication problems
✅ Scaling/growth challenges in business
✅ Business efficiency/productivity issues
✅ Revenue/sales process problems
✅ Team management difficulties
✅ Business automation needs

❌ EXCLUDE THESE (NOT BUSINESS):
- Personal health issues (sick, doctor, medical)
- Pet problems (cat/dog sick, vet bills)
- Personal relationships/family issues
- Entertainment queries (movies, music, jokes)
- Weather, general knowledge, casual chat
- Personal finance (unless business-related)
- Educational questions (unless for business)

CONFIDENCE SCORING:
- 0-30: No business context
- 31-50: Ambiguous/unclear  
- 51-70: Possible business relevance
- 71-85: Clear business pain point
- 86-100: Strong business opportunity

Return ONLY valid JSON:
{{
    "opportunity_detected": true/false,
    "opportunity_score": 0-100,
    "detected_pain_points": ["specific problem 1", "specific problem 2"],
    "solution_areas": ["area where we can help 1", "area 2"],
    "sales_mode": "none|casual|soft_pitch|consultative",
    "confidence_level": "low|medium|high",
    "reasoning": "Brief explanation of why this is/isn't a business opportunity"
}}

EXAMPLES:
"Customer support me bahut problem hai" → opportunity_detected: true, score: 85
"My cat is sick" → opportunity_detected: false, score: 0  
"Inventory track karna mushkil hai" → opportunity_detected: true, score: 90
"Weather kaisa hai?" → opportunity_detected: false, score: 0
"Manual data entry takes forever" → opportunity_detected: true, score: 80
"I love pizza" → opportunity_detected: false, score: 0"""

        try:
            logger.debug("🔍 Calling LLM for opportunity analysis...")
            response = await self.llm_client.generate(
                [{"role": "user", "content": opportunity_prompt}],
                0.1,
                system_prompt="You are a business opportunity detection expert. Analyze queries ONLY for business-related problems. Return valid JSON only."
            )
            cleaned_response = clean_llm_response(response)
            
            opportunity_data = json.loads(cleaned_response)
            
            # Log analysis results
            detected = opportunity_data.get("opportunity_detected", False)
            score = opportunity_data.get("opportunity_score", 0)
            confidence = opportunity_data.get("confidence_level", "low")
            
            if detected:
                logger.info(f"✅ 🔍 Business opportunity DETECTED (score: {score}, confidence: {confidence})")
                logger.info(f"   Pain points: {opportunity_data.get('detected_pain_points', [])}")
                logger.info(f"   Solution areas: {opportunity_data.get('solution_areas', [])}")
            else:
                logger.info(f"❌ 🔍 No business opportunity detected (score: {score})")
            
            return opportunity_data
            
        except Exception as e:
            logger.error(f"❌ 🔍 Opportunity analysis failed: {str(e)}")
            # Fallback - assume no opportunity on error
            return {
                "opportunity_detected": False,
                "opportunity_score": 0,
                "detected_pain_points": [],
                "solution_areas": [],
                "sales_mode": "none",
                "confidence_level": "low",
                "reasoning": f"Analysis failed: {str(e)}"
            }

class BrainMemory:
    """Simple memory system for Brain Agent"""
    
    def __init__(self, db_path: str = "brain_memory.db"):
        self.db_path = db_path
        self._initialize_db()
        logger.info(f"BrainMemory initialized with database: {db_path}")
    
    def _initialize_db(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    tools_used TEXT,
                    business_opportunity TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        logger.debug("BrainMemory database initialized")
    
    def store_memory(self, query: str, response: str, tools_used: List[str], business_opportunity: Dict[str, Any] = None):
        """Store interaction in memory with business opportunity data"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO memories (query, response, tools_used, business_opportunity)
                VALUES (?, ?, ?, ?)
            """, (query, response, json.dumps(tools_used), json.dumps(business_opportunity or {})))
            conn.commit()
        logger.info(f"Stored memory: query='{query[:30]}...', tools_used={tools_used}, opportunity_detected={business_opportunity.get('opportunity_detected', False) if business_opportunity else False}")
    
    def get_recent_memories(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent memories"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT query, response, tools_used, business_opportunity, timestamp
                FROM memories ORDER BY timestamp DESC LIMIT ?
            """, (limit,))
            
            memories = []
            for row in cursor:
                memories.append({
                    "query": row[0],
                    "response": row[1][:200] + "..." if len(row[1]) > 200 else row[1],
                    "tools_used": json.loads(row[2]),
                    "business_opportunity": json.loads(row[3]) if row[3] else {},
                    "timestamp": row[4]
                })
        
        logger.debug(f"Retrieved {len(memories)} recent memories")
        return memories

class BrainAgent:
    """Brain Agent - Pure LLM-driven orchestrator with Business Opportunity Detection"""
    
    def __init__(self, llm_client: LLMClient, tool_manager: ToolManager):
        logger.info("🧠 Initializing Brain Agent with Business Opportunity Detection")
        self.llm_client = llm_client
        self.tool_manager = tool_manager
        self.memory = BrainMemory()
        self.opportunity_detector = BusinessOpportunityDetector(llm_client)
        self.available_tools = tool_manager.get_available_tools()
        logger.info(f"🧠 Brain Agent initialized with tools: {self.available_tools}")
    
    async def process_query(self, query: str, user_id: str = None, **kwargs) -> Dict[str, Any]:
        """Process query using pure LLM decision making with business opportunity detection"""
        
        logger.info(f"🧠 PROCESSING QUERY: '{query[:50]}...'")
        logger.info(f"🧠 User ID: {user_id}")
        logger.info(f"🧠 Available tools: {self.available_tools}")
        
        try:
            # Get recent memories for context
            logger.debug("🧠 Getting recent memories...")
            recent_memories = self.memory.get_recent_memories(3)
            memory_context = self._format_memory_context(recent_memories)
            
            # STEP 1: Analyze business opportunity FIRST
            logger.info("🔍 Step 1: Analyzing business opportunity...")
            business_opportunity = await self.opportunity_detector.analyze_opportunity(query, recent_memories)
            
            # Get available tools information
            logger.debug("🧠 Step 2: Formatting tools info...")
            tools_info = self._format_tools_info()
            
            # STEP 3: Let LLM decide everything about how to handle this query
            logger.info("🧠 Step 3: Creating execution plan...")
            plan = await self._create_execution_plan(query, tools_info, memory_context, business_opportunity)
            logger.info(f"🧠 EXECUTION PLAN CREATED: {plan}")
            
            # STEP 4: Execute the LLM-generated plan
            logger.info("🧠 Step 4: Executing plan...")
            execution_results = await self._execute_plan(plan, query, user_id)
            logger.info(f"🧠 EXECUTION RESULTS: {execution_results}")
            
            # Add business opportunity data to execution results
            execution_results["business_opportunity"] = business_opportunity
            
            # STEP 5: Let LLM synthesize final response
            logger.info("🧠 Step 5: Synthesizing final response...")
            final_response = await self._synthesize_response(query, plan, execution_results)
            logger.info(f"🧠 FINAL RESPONSE LENGTH: {len(final_response)} chars")
            
            # Store in memory with business opportunity data
            tools_used = plan.get("tools_to_use", [])
            logger.debug(f"🧠 Storing memory with tools: {tools_used}")
            self.memory.store_memory(query, final_response, tools_used, business_opportunity)
            
            logger.info("✅ 🧠 Brain Agent processing COMPLETED successfully")
            return {
                "success": True,
                "query": query,
                "plan": plan,
                "execution_results": execution_results,
                "business_opportunity": business_opportunity,
                "response": final_response,
                "tools_used": tools_used
            }
            
        except Exception as e:
            logger.error(f"❌ 🧠 Brain Agent processing FAILED: {str(e)}")
            logger.error(f"❌ 🧠 Exception details: {type(e).__name__}")
            
            import traceback
            logger.error(f"❌ 🧠 Full traceback: {traceback.format_exc()}")
            
            return {
                "success": False,
                "error": f"Brain processing failed: {str(e)}",
                "query": query
            }
    
    async def _create_execution_plan(self, query: str, tools_info: str, memory_context: str, business_opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Let LLM create complete execution plan with business context"""
        
        logger.debug("🧠 Creating LLM planning prompt with business context...")
        
        # Include business opportunity context in planning
        business_context = ""
        if business_opportunity.get("opportunity_detected"):
            business_context = f"""
BUSINESS OPPORTUNITY DETECTED:
- Score: {business_opportunity.get('opportunity_score', 0)}/100
- Pain Points: {business_opportunity.get('detected_pain_points', [])}
- Solution Areas: {business_opportunity.get('solution_areas', [])}
- Recommended Sales Mode: {business_opportunity.get('sales_mode', 'none')}
"""
        
        planning_prompt = f"""Analyze this query step by step and create an execution plan: {query}

Available Tools:
{tools_info}

Recent Context:
{memory_context}

{business_context}

STEP-BY-STEP TOOL SELECTION:
1. What information is needed to answer this query?
2. Where can that information come from?
3. What processing/analysis is required?
4. Select appropriate tools based on these needs

TOOL USAGE RULES:
✅ RAG: When user mentions "my product/company/documents" OR asks about uploaded files
✅ Calculator: For calculations, percentages, market analysis, financial metrics  
✅ Web Search: For current information, market data, news, trends, competitor info
❌ NO TOOLS: For greetings, thanks, casual chat, personal questions

USE MULTIPLE TOOLS WHEN HELPFUL:
- Market comparisons → ["web_search", "calculator"]
- "My product vs competitor" → ["web_search", "rag", "calculator"]  
- Financial analysis → ["web_search", "calculator"]
- Document + market research → ["rag", "web_search"]

SPECIFIC EXAMPLES:
✅ "pepsi vs cola market share" → ["web_search", "calculator"]
✅ "compare my product revenue vs competitor" → ["rag", "web_search", "calculator"]
✅ "analyze my contract terms for payment" → ["rag"]
✅ "current AI market trends" → ["web_search"]  
✅ "calculate 15% growth on $50k revenue" → ["calculator"]
✅ "what's ROI of my marketing spend" → ["rag", "calculator"]
❌ "hi how are you" → []
❌ "thanks for help" → []

Create a JSON plan with:
- "approach": Type of approach needed
- "tools_to_use": Array of tools to use (can be multiple, empty array [] for no tools)
- "reasoning": Step-by-step explanation of tool selection
- "business_context_aware": true/false

Respond with valid JSON only."""

        messages = [{"role": "user", "content": planning_prompt}]
        system_prompt = """You are the Brain Agent - analyze queries step by step and select optimal tool combinations. Use multiple tools when they complement each other. For simple conversation, use no tools. Think carefully about what information sources and processing are needed. Respond with valid JSON only."""
        
        try:
            logger.info("🧠 Calling LLM for execution plan...")
            response = await self.llm_client.generate(messages, 0.1, system_prompt=system_prompt)
            logger.debug(f"🧠 LLM planning response: {response[:200]}...")
            
            # Clean markdown code blocks from response
            cleaned_response = clean_llm_response(response)

            # Parse cleaned JSON  
            plan = json.loads(cleaned_response)
            logger.info(f"🧠 Parsed execution plan successfully: {plan}")
            return plan
            
        except Exception as e:
            logger.error(f"❌ 🧠 LLM planning failed: {str(e)}")
            # Fallback plan - NO TOOLS for safety
            fallback_plan = {
                "approach": "conversational_response",
                "tools_to_use": [],
                "reasoning": f"LLM planning failed, using conversational approach: {e}",
                "business_context_aware": business_opportunity.get("opportunity_detected", False)
            }
            logger.info(f"🧠 Using fallback plan: {fallback_plan}")
            return fallback_plan
    
    async def _execute_plan(self, plan: Dict[str, Any], original_query: str, user_id: str = None) -> Dict[str, Any]:
        """Execute the LLM-generated plan"""
        
        logger.info(f"🧠 EXECUTING PLAN with user_id: {user_id}")
        execution_results = {}
        tools_to_use = plan.get("tools_to_use", [])
        
        logger.info(f"🧠 Tools to execute: {tools_to_use}")
        
        # If no tools needed, return empty results
        if not tools_to_use:
            logger.info("🧠 No tools needed - conversational response")
            return {"no_tools_used": True}
        
        for i, tool_name in enumerate(tools_to_use):
            step_key = f"step_{i+1}"
            logger.info(f"🧠 EXECUTING STEP {i+1}: tool='{tool_name}'")
            
            try:
                # Check if tool is available
                if tool_name not in self.available_tools:
                    logger.error(f"❌ 🧠 Tool '{tool_name}' not available in: {self.available_tools}")
                    execution_results[step_key] = {
                        "tool": tool_name,
                        "error": f"Tool '{tool_name}' not available"
                    }
                    continue
                
                # Execute tool with enhanced context-aware query
                # enhanced_query = await self._enhance_query_with_context(original_query, plan)
                

                
                result = await self.tool_manager.execute_tool(
                    tool_name, 
                    query=original_query,
                    user_id=user_id
                )
                
                logger.info(f"✅ 🧠 Step {i+1} SUCCESS: {type(result)} returned")
                logger.debug(f"🧠 Step {i+1} result details: {result}")
                
                execution_results[step_key] = {
                    "tool": tool_name,
                    "result": result
                }
                
            except Exception as e:
                logger.error(f"❌ 🧠 Step {i+1} FAILED: {str(e)}")
                logger.error(f"❌ 🧠 Tool: {tool_name}, Error type: {type(e).__name__}")
                
                import traceback
                logger.error(f"❌ 🧠 Step {i+1} traceback: {traceback.format_exc()}")
                
                execution_results[step_key] = {
                    "tool": tool_name,
                    "error": str(e)
                }
        
        logger.info(f"🧠 PLAN EXECUTION COMPLETED: {len(execution_results)} steps")
        return execution_results
    
    async def _synthesize_response(self, query: str, plan: Dict[str, Any], 
                                 execution_results: Dict[str, Any]) -> str:
        """Let LLM synthesize final response from all results including business context"""
        
        logger.info("🧠 Synthesizing final response with business context...")
        
        # Extract business opportunity data
        business_opportunity = execution_results.get("business_opportunity", {})
        
        synthesis_prompt = f"""Synthesize a comprehensive response for: {query}

Execution Plan: {json.dumps(plan, indent=2)}
Execution Results: {json.dumps(execution_results, indent=2)}

BUSINESS CONTEXT:
{json.dumps(business_opportunity, indent=2)}

Create a well-structured response that:
1. Directly addresses the user's query using all available information
2. If business opportunity detected, naturally incorporate helpful business insights
3. Maintain conversational and helpful tone
4. DO NOT sound pushy or overly sales-focused

If no tools were used, provide a natural conversational response."""

        messages = [{"role": "user", "content": synthesis_prompt}]
        system_prompt = """You are the Brain Agent creating comprehensive responses. Analyze all execution results and business context to create valuable responses that directly help users. For conversational queries, be natural and friendly. If business opportunities exist, mention solutions naturally without being pushy."""
        
        try:
            logger.debug("🧠 Calling LLM for response synthesis...")
            response = await self.llm_client.generate(messages, 0.1, system_prompt=system_prompt)
            logger.info(f"✅ 🧠 Response synthesis completed: {len(response)} chars")
            return response
            
        except Exception as e:
            logger.error(f"❌ 🧠 Response synthesis failed: {str(e)}")
            return f"Sorry, I encountered an error while synthesizing the response: {str(e)}"
    
    def _format_tools_info(self) -> str:
        """Format available tools information for LLM"""
        
        tools_descriptions = {
            "calculator": "Perform mathematical calculations and statistical analysis",
            "web_search": "Search the internet for current information", 
            "rag": "Retrieve information from uploaded knowledge base"
        }
        
        available_info = []
        for tool in self.available_tools:
            description = tools_descriptions.get(tool, "Tool available")
            available_info.append(f"- {tool}: {description}")
        
        formatted_info = "\n".join(available_info)
        logger.debug(f"🧠 Formatted tools info: {formatted_info}")
        return formatted_info
    
    def _format_memory_context(self, memories: List[Dict[str, Any]]) -> str:
        """Format recent memories for context"""
        
        if not memories:
            logger.debug("🧠 No recent memories available")
            return "No recent context available."
        
        context_parts = []
        for memory in memories:
            context_parts.append(f"Previous Query: {memory['query']}")
            context_parts.append(f"Tools Used: {', '.join(memory['tools_used'])}")
            
            # Include business opportunity context if available
            business_opp = memory.get('business_opportunity', {})
            if business_opp.get('opportunity_detected'):
                context_parts.append(f"Previous Business Opportunity: Score {business_opp.get('opportunity_score', 0)}/100")
            context_parts.append("")
        
        formatted_context = "\n".join(context_parts)
        logger.debug(f"🧠 Formatted memory context: {len(formatted_context)} chars")
        return formatted_context
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of Brain Agent memory including business opportunities"""
        
        recent_memories = self.memory.get_recent_memories(10)
        
        # Count business opportunities in recent memories
        business_opportunities_count = sum(
            1 for memory in recent_memories 
            if memory.get('business_opportunity', {}).get('opportunity_detected', False)
        )
        
        summary = {
            "total_memories": len(recent_memories),
            "recent_interactions": recent_memories,
            "available_tools": self.available_tools,
            "business_opportunities_detected": business_opportunities_count
        }
        
        logger.info(f"🧠 Memory summary: {summary['total_memories']} memories, {len(summary['available_tools'])} tools, {business_opportunities_count} business opportunities")
        return summary

    async def _enhance_query_with_context(self, query: str, plan: Dict[str, Any]) -> str:
        """Enhance query with context from reasoning for better tool results"""
        
        reasoning = plan.get("reasoning", "")
        
        # If reasoning mentions expanding pronouns to specific entities, do it
        if "refers to" in reasoning:
            import re
            # Look for patterns like "refers to Charlie Kirk" or "refers to 'Charlie Kirk'"
            person_match = re.search(r"refers to.*?(?:'([^']+)'|\"([^\"]+)\"|(\b[A-Z][a-zA-Z]+ [A-Z][a-zA-Z]+\b))", reasoning)
            if person_match:
                # Get the matched name (from any of the capture groups)
                person_name = person_match.group(1) or person_match.group(2) or person_match.group(3)
                if person_name:
                    enhanced = query.lower()
                    enhanced = enhanced.replace("he ", f"{person_name} ").replace("she ", f"{person_name} ")
                    enhanced = enhanced.replace("him ", f"{person_name} ").replace("her ", f"{person_name} ")
                    enhanced = enhanced.replace("his ", f"{person_name}'s ").replace("hers ", f"{person_name}'s ")
                    enhanced = enhanced.replace("it ", f"{person_name} ").replace("this ", f"{person_name} ")
                    enhanced = enhanced.replace("them ", f"{person_name} ").replace("they ", f"{person_name} ")
                    logger.info(f"🧠 Context enhancement: '{query}' → '{enhanced}'")
                    return enhanced
        
        # Return original query if no enhancement needed
        return query