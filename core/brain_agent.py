"""
Brain Agent - Pure LLM-driven orchestrator
"""

import asyncio
import json
import sqlite3
from typing import Dict, List, Any, Optional
from datetime import datetime
from .llm_client import LLMClient
from .tools import ToolManager
from .exceptions import BrainAgentError

class BrainMemory:
    """Simple memory system for Brain Agent"""
    
    def __init__(self, db_path: str = "brain_memory.db"):
        self.db_path = db_path
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    tools_used TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def store_memory(self, query: str, response: str, tools_used: List[str]):
        """Store interaction in memory"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO memories (query, response, tools_used)
                VALUES (?, ?, ?)
            """, (query, response, json.dumps(tools_used)))
            conn.commit()
    
    def get_recent_memories(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent memories"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT query, response, tools_used, timestamp
                FROM memories ORDER BY timestamp DESC LIMIT ?
            """, (limit,))
            
            memories = []
            for row in cursor:
                memories.append({
                    "query": row[0],
                    "response": row[1][:200] + "..." if len(row[1]) > 200 else row[1],
                    "tools_used": json.loads(row[2]),
                    "timestamp": row[3]
                })
            return memories

class BrainAgent:
    """Brain Agent - Pure LLM-driven orchestrator"""
    
    def __init__(self, llm_client: LLMClient, tool_manager: ToolManager):
        self.llm_client = llm_client
        self.tool_manager = tool_manager
        self.memory = BrainMemory()
        self.available_tools = tool_manager.get_available_tools()
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process query using pure LLM decision making"""
        
        try:
            # Get available tools information
            tools_info = self._format_tools_info()
            
            # Get recent memories for context
            recent_memories = self.memory.get_recent_memories(3)
            memory_context = self._format_memory_context(recent_memories)
            
            # Let LLM decide everything about how to handle this query
            plan = await self._create_execution_plan(query, tools_info, memory_context)
            
            # Execute the LLM-generated plan
            execution_results = await self._execute_plan(plan, query)
            
            # Let LLM synthesize final response
            final_response = await self._synthesize_response(query, plan, execution_results)
            
            # Store in memory
            tools_used = plan.get("tools_to_use", [])
            self.memory.store_memory(query, final_response, tools_used)
            
            return {
                "success": True,
                "query": query,
                "plan": plan,
                "execution_results": execution_results,
                "response": final_response,
                "tools_used": tools_used
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Brain processing failed: {str(e)}",
                "query": query
            }
    
    async def _create_execution_plan(self, query: str, tools_info: str, memory_context: str) -> Dict[str, Any]:
        """Let LLM create complete execution plan"""
        
        planning_prompt = f"""Analyze this query and create an execution plan: {query}

Available Tools:
{tools_info}

Recent Context:
{memory_context}

Create a JSON plan with:
- "approach": Type of approach needed
- "tools_to_use": List of tools to use
- "reasoning": Why this approach

Respond with valid JSON only."""

        messages = [{"role": "user", "content": planning_prompt}]
        system_prompt = """You are the Brain Agent - analyze queries and create optimal execution plans using available tools. Respond with valid JSON only."""
        
        try:
            response = await self.llm_client.generate(messages, system_prompt, temperature=0.3)
            plan = json.loads(response)
            return plan
        except Exception as e:
            # Fallback plan
            return {
                "approach": "simple_analysis",
                "tools_to_use": ["rag"],
                "reasoning": f"LLM planning failed: {e}",
                "execution_steps": [
                    {
                        "step": 1,
                        "tool": "rag",
                        "action": "analyze_query",
                        "parameters": {"query": query}
                    }
                ]
            }
    
    async def _execute_plan(self, plan: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """Execute the LLM-generated plan"""
        
        execution_results = {}
        tools_to_use = plan.get("tools_to_use", [])
        
        for i, tool_name in enumerate(tools_to_use):
            try:
                # Simple execution - use the tool with the original query
                result = await self.tool_manager.execute_tool(tool_name, query=original_query)
                
                execution_results[f"step_{i+1}"] = {
                    "tool": tool_name,
                    "result": result
                }
                
            except Exception as e:
                execution_results[f"step_{i+1}"] = {
                    "tool": tool_name,
                    "error": str(e)
                }
        
        return execution_results
    
    async def _synthesize_response(self, query: str, plan: Dict[str, Any], 
                                 execution_results: Dict[str, Any]) -> str:
        """Let LLM synthesize final response from all results"""
        
        synthesis_prompt = f"""Synthesize a comprehensive response for: {query}

Execution Plan: {json.dumps(plan, indent=2)}
Execution Results: {json.dumps(execution_results, indent=2)}

Create a well-structured response that directly addresses the user's query using all available information."""

        messages = [{"role": "user", "content": synthesis_prompt}]
        system_prompt = """You are the Brain Agent creating comprehensive responses. Analyze all execution results and create valuable responses that directly help users."""
        
        return await self.llm_client.generate(messages, system_prompt, temperature=0.4)
    
    def _format_tools_info(self) -> str:
        """Format available tools information for LLM"""
        
        tools_descriptions = {
            "calculator": "Perform mathematical calculations and statistical analysis",
            "web_search": "Search the internet for current information", 
            "rag": "Retrieve information from knowledge base"
        }
        
        available_info = []
        for tool in self.available_tools:
            description = tools_descriptions.get(tool, "Tool available")
            available_info.append(f"- {tool}: {description}")
        
        return "\n".join(available_info)
    
    def _format_memory_context(self, memories: List[Dict[str, Any]]) -> str:
        """Format recent memories for context"""
        
        if not memories:
            return "No recent context available."
        
        context_parts = []
        for memory in memories:
            context_parts.append(f"Previous Query: {memory['query']}")
            context_parts.append(f"Tools Used: {', '.join(memory['tools_used'])}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of Brain Agent memory"""
        
        recent_memories = self.memory.get_recent_memories(10)
        
        return {
            "total_memories": len(recent_memories),
            "recent_interactions": recent_memories,
            "available_tools": self.available_tools
        }