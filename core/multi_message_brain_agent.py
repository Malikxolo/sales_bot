"""
Brain Agent - Pure LLM-driven orchestrator
ENHANCED with comprehensive logging for debugging
UPDATED to accept list of messages instead of single query
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
                    messages TEXT NOT NULL,
                    response TEXT NOT NULL,
                    tools_used TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        logger.debug("BrainMemory database initialized")
    
    def store_memory(self, messages: List[Dict[str, str]], response: str, tools_used: List[str]):
        """Store interaction in memory"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO memories (messages, response, tools_used)
                VALUES (?, ?, ?)
            """, (json.dumps(messages), response, json.dumps(tools_used)))
            conn.commit()
        
        # Extract last user message for logging
        last_user_msg = next((msg['content'] for msg in reversed(messages) if msg['role'] == 'user'), 'No user message')
        logger.info(f"Stored memory: last_user_msg='{last_user_msg[:30]}...', tools_used={tools_used}")
    
    def get_recent_memories(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent memories"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT messages, response, tools_used, timestamp
                FROM memories ORDER BY timestamp DESC LIMIT ?
            """, (limit,))
            
            memories = []
            for row in cursor:
                messages = json.loads(row[0])
                # Extract last user message for summary
                last_user_msg = next((msg['content'] for msg in reversed(messages) if msg['role'] == 'user'), 'No user message')
                
                memories.append({
                    "messages": messages,
                    "last_user_query": last_user_msg,
                    "response": row[1][:200] + "..." if len(row[1]) > 200 else row[1],
                    "tools_used": json.loads(row[2]),
                    "timestamp": row[3]
                })
        
        logger.debug(f"Retrieved {len(memories)} recent memories")
        return memories


class BrainAgent:
    """Brain Agent - Pure LLM-driven orchestrator"""
    
    def __init__(self, llm_client: LLMClient, tool_manager: ToolManager):
        logger.info("🧠 Initializing Brain Agent")
        self.llm_client = llm_client
        self.tool_manager = tool_manager
        self.memory = BrainMemory()
        self.available_tools = tool_manager.get_available_tools()
        logger.info(f"🧠 Brain Agent initialized with tools: {self.available_tools}")
    
    async def process_query(self, messages: List[Dict[str, str]], user_id: str = None, **kwargs) -> Dict[str, Any]:
        """Process query using pure LLM decision making
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
                     e.g., [{"role": "user", "content": "What is the weather?"}]
            user_id: Optional user identifier
            **kwargs: Additional arguments
            
        Returns:
            Dict containing success status, plan, execution results, and response
        """
        
        # Extract the current query from the last user message
        current_query = self._extract_current_query(messages)
        
        logger.info(f"🧠 PROCESSING MESSAGES: {len(messages)} messages")
        logger.info(f"🧠 Current query: '{current_query[:50]}...'")
        logger.info(f"🧠 User ID: {user_id}")
        logger.info(f"🧠 Available tools: {self.available_tools}")
        
        try:
            # Get available tools information
            logger.debug("🧠 Formatting tools info...")
            tools_info = self._format_tools_info()
            
            # Get recent memories for context
            logger.debug("🧠 Getting recent memories...")
            recent_memories = self.memory.get_recent_memories(3)
            memory_context = self._format_memory_context(recent_memories)
            
            # Let LLM decide everything about how to handle this query
            logger.info("🧠 Creating execution plan...")
            plan = await self._create_execution_plan(messages, current_query, tools_info, memory_context)
            logger.info(f"🧠 EXECUTION PLAN CREATED: {plan}")
            
            # Execute the LLM-generated plan
            logger.info("🧠 Executing plan...")
            execution_results = await self._execute_plan(plan, current_query, messages, user_id)
            logger.info(f"🧠 EXECUTION RESULTS: {execution_results}")
            
            # Let LLM synthesize final response
            logger.info("🧠 Synthesizing final response...")
            final_response = await self._synthesize_response(messages, current_query, plan, execution_results)
            logger.info(f"🧠 FINAL RESPONSE LENGTH: {len(final_response)} chars")
            
            # Store in memory
            tools_used = plan.get("tools_to_use", [])
            logger.debug(f"🧠 Storing memory with tools: {tools_used}")
            self.memory.store_memory(messages, final_response, tools_used)
            
            logger.info("✅ 🧠 Brain Agent processing COMPLETED successfully")
            return {
                "success": True,
                "messages": messages,
                "current_query": current_query,
                "plan": plan,
                "execution_results": execution_results,
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
                "messages": messages,
                "current_query": current_query
            }
    
    def _extract_current_query(self, messages: List[Dict[str, str]]) -> str:
        """Extract the current query from the last user message"""
        
        for message in reversed(messages):
            if message.get("role") == "user":
                return message.get("content", "")
        
        # Fallback if no user message found
        return "No user query found in messages"
    
    async def _create_execution_plan(self, messages: List[Dict[str, str]], current_query: str, 
                                   tools_info: str, memory_context: str) -> Dict[str, Any]:
        """Let LLM create complete execution plan"""
        
        logger.debug("🧠 Creating LLM planning prompt...")
        
        # Format conversation context
        conversation_context = self._format_conversation_context(messages)
        
        planning_prompt = f"""Analyze this conversation and the current query to create an execution plan.

CONVERSATION CONTEXT:
{conversation_context}

CURRENT QUERY: {current_query}

Available Tools:
{tools_info}

Recent Context:
{memory_context}

Create a JSON plan with:
- "approach": Type of approach needed
- "tools_to_use": List of tools to use (can be empty if no tools needed)
- "reasoning": Why this approach
- "requires_tools": boolean indicating if tools are needed

Consider the full conversation context when deciding which tools to use.
Respond with valid JSON only."""

        planning_messages = [{"role": "user", "content": planning_prompt}]
        system_prompt = """You are the Brain Agent - analyze conversations and create optimal execution plans using available tools. Consider the full conversation context, not just the last message. Respond with valid JSON only."""
        
        try:
            logger.info("🧠 Calling LLM for execution plan...")
            response = await self.llm_client.generate(planning_messages, system_prompt, temperature=0.3)
            logger.debug(f"🧠 LLM planning response: {response[:200]}...")
            
            # Clean markdown code blocks from response
            cleaned_response = response.strip()
            # Remove XML thinking tags (for qwen, deepseek models)
            if '<think>' in cleaned_response and '</think>' in cleaned_response:
                end_tag = cleaned_response.find('</think>')
                if end_tag != -1:
                    cleaned_response = cleaned_response[end_tag + 8:].strip()
                    
            backticks = '`' * 3  # This creates ```
            if cleaned_response.startswith(backticks):
                # Remove opening markdown block
                lines = cleaned_response.split('\n')
                lines = lines[1:]  # Remove first line with ```
                if lines and lines[-1].strip() == backticks:
                    lines = lines[:-1]  # Remove closing ```
                cleaned_response = '\n'.join(lines)

            # Parse cleaned JSON  
            plan = json.loads(cleaned_response)
            logger.info(f"🧠 Parsed execution plan successfully: {plan}")
            return plan
            
        except Exception as e:
            logger.error(f"❌ 🧠 LLM planning failed: {str(e)}")
            # Fallback plan
            fallback_plan = {
                "approach": "simple_analysis",
                "tools_to_use": ["rag"],
                "reasoning": f"LLM planning failed: {e}",
                "requires_tools": True,
                "execution_steps": [
                    {
                        "step": 1,
                        "tool": "rag",
                        "action": "analyze_query",
                        "parameters": {"query": current_query}
                    }
                ]
            }
            logger.info(f"🧠 Using fallback plan: {fallback_plan}")
            return fallback_plan
    
    def _format_conversation_context(self, messages: List[Dict[str, str]]) -> str:
        """Format conversation messages for context"""
        
        if not messages:
            return "No conversation context available."
        
        context_parts = []
        for i, message in enumerate(messages):
            role = message.get("role", "unknown")
            content = message.get("content", "")
            
            # Truncate long messages for context
            if len(content) > 150:
                content = content[:150] + "..."
            
            context_parts.append(f"{role.upper()}: {content}")
        
        formatted_context = "\n".join(context_parts)
        logger.debug(f"🧠 Formatted conversation context: {len(formatted_context)} chars")
        return formatted_context
    
    async def _execute_plan(self, plan: Dict[str, Any], current_query: str, 
                          messages: List[Dict[str, str]], user_id: str = None) -> Dict[str, Any]:
        """Execute the LLM-generated plan"""
        
        logger.info(f"🧠 EXECUTING PLAN with user_id: {user_id}")
        execution_results = {}
        tools_to_use = plan.get("tools_to_use", [])
        
        logger.info(f"🧠 Tools to execute: {tools_to_use}")
        
        # If no tools are needed, return empty results
        if not tools_to_use or not plan.get("requires_tools", True):
            logger.info("🧠 No tools required for this query")
            return {"no_tools_needed": True, "reason": "Plan indicates no tools are required"}
        
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
                
                # Execute tool with user_id for RAG
                logger.info(f"🧠 Calling tool_manager.execute_tool('{tool_name}', query='{current_query[:30]}...', user_id='{user_id}')")
                
                # CRITICAL: Pass user_id to tool execution
                # For tools that might need conversation context, we could pass messages too
                result = await self.tool_manager.execute_tool(
                    tool_name, 
                    query=current_query,
                    user_id=user_id,
                    messages=messages  # Pass full conversation context to tools
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
    
    async def _synthesize_response(self, messages: List[Dict[str, str]], current_query: str,
                                 plan: Dict[str, Any], execution_results: Dict[str, Any]) -> str:
        """Let LLM synthesize final response from all results"""
        
        logger.info("🧠 Synthesizing final response...")
        
        # Format conversation context for synthesis
        conversation_context = self._format_conversation_context(messages)
        
        synthesis_prompt = f"""Synthesize a comprehensive response based on this conversation and execution results.

CONVERSATION CONTEXT:
{conversation_context}

CURRENT QUERY: {current_query}

EXECUTION PLAN: {json.dumps(plan, indent=2)}

EXECUTION RESULTS: {json.dumps(execution_results, indent=2)}

Create a well-structured response that:
1. Directly addresses the current user query
2. Takes into account the full conversation context
3. Uses all available execution results
4. Maintains conversation flow and continuity
5. Is helpful and informative

Respond naturally as if continuing the conversation."""

        synthesis_messages = [{"role": "user", "content": synthesis_prompt}]
        system_prompt = """You are the Brain Agent creating comprehensive responses. Analyze all execution results and conversation context to create valuable responses that directly help users and maintain natural conversation flow."""
        
        try:
            logger.debug("🧠 Calling LLM for response synthesis...")
            response = await self.llm_client.generate(synthesis_messages, system_prompt, temperature=0.4)
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
            context_parts.append(f"Previous Query: {memory['last_user_query']}")
            context_parts.append(f"Tools Used: {', '.join(memory['tools_used'])}")
            context_parts.append("")
        
        formatted_context = "\n".join(context_parts)
        logger.debug(f"🧠 Formatted memory context: {len(formatted_context)} chars")
        return formatted_context
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of Brain Agent memory"""
        
        recent_memories = self.memory.get_recent_memories(10)
        
        summary = {
            "total_memories": len(recent_memories),
            "recent_interactions": recent_memories,
            "available_tools": self.available_tools
        }
        
        logger.info(f"🧠 Memory summary: {summary['total_memories']} memories, {len(summary['available_tools'])} tools")
        return summary

    
    async def process_single_query(self, query: str, user_id: str = None, **kwargs) -> Dict[str, Any]:
        """Legacy method for processing single queries - converts to messages format"""
        
        messages = [{"role": "user", "content": query}]
        return await self.process_query(messages, user_id, **kwargs)