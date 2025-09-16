"""
Tool system for Brain-Heart Deep Research System
FIXED VERSION - Proper model selection flow
"""

import asyncio
import aiohttp
import json
import math
import statistics
import re
import sqlite3
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod
from .exceptions import ToolExecutionError
from .llm_client import LLMClient
from .web_search_agent import search_perplexity

class BaseTool(ABC):
    """Base class for all tools"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.usage_count = 0
        self.last_used = None
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters"""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get tool information"""
        return {
            "name": self.name,
            "description": self.description,
            "usage_count": self.usage_count,
            "last_used": self.last_used
        }
    
    def _record_usage(self):
        """Record tool usage"""
        self.usage_count += 1
        self.last_used = datetime.now().isoformat()

class CalculatorTool(BaseTool):
    """Mathematical calculator tool"""
    
    def __init__(self):
        super().__init__(
            "calculator",
            "Perform mathematical calculations and statistical operations"
        )
    
    async def execute(self, expression: str = None, operation: str = None, 
                     numbers: List[float] = None, **kwargs) -> Dict[str, Any]:
        """Execute mathematical operations"""
        
        self._record_usage()
        
        try:
            if expression:
                return await self._evaluate_expression(expression)
            elif operation and numbers:
                return await self._perform_operation(operation, numbers)
            else:
                return {
                    "success": False,
                    "error": "Provide either 'expression' or 'operation' with 'numbers'"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Calculation error: {str(e)}",
                "tool": self.name
            }
    
    async def _evaluate_expression(self, expression: str) -> Dict[str, Any]:
        """Safely evaluate mathematical expression"""
        
        # Simple security check
        dangerous_patterns = ['import', 'exec', 'eval', '__']
        if any(pattern in expression.lower() for pattern in dangerous_patterns):
            return {
                "success": False,
                "error": "Expression contains potentially dangerous operations"
            }
        
        try:
            # Safe evaluation with limited scope
            allowed_names = {
                "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
                "sqrt": math.sqrt, "pow": pow, "log": math.log, "exp": math.exp,
                "sin": math.sin, "cos": math.cos, "tan": math.tan,
                "pi": math.pi, "e": math.e
            }
            
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            
            return {
                "success": True,
                "result": result,
                "expression": expression,
                "formatted_result": f"{result:,.6g}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Invalid expression: {str(e)}",
                "expression": expression
            }
    
    async def _perform_operation(self, operation: str, numbers: List[float]) -> Dict[str, Any]:
        """Perform statistical operations"""
        
        if not numbers:
            return {"success": False, "error": "No numbers provided"}
        
        try:
            operations_map = {
                "mean": lambda x: statistics.mean(x),
                "median": lambda x: statistics.median(x),
                "mode": lambda x: statistics.mode(x),
                "stdev": lambda x: statistics.stdev(x) if len(x) > 1 else 0,
                "variance": lambda x: statistics.variance(x) if len(x) > 1 else 0,
                "sum": lambda x: sum(x),
                "min": lambda x: min(x),
                "max": lambda x: max(x)
            }
            
            if operation in operations_map:
                result = operations_map[operation](numbers)
            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}
            
            return {
                "success": True,
                "operation": operation,
                "numbers": numbers,
                "result": result,
                "count": len(numbers),
                "formatted_result": f"{result:,.6g}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Operation failed: {str(e)}",
                "operation": operation
            }

class WebSearchTool(BaseTool):
    """Web search tool with multiple provider support"""
    
    def __init__(self, api_key: str, web_model: str, provider: str = "perplexity"):
        super().__init__(
            "web_search", 
            "Search the internet for current information"
        )
        self.api_key = api_key
        self.provider = provider
        self.web_model = web_model  # Fixed: properly store the web model
        self.session = None
        
        print(f"WebSearchTool initialized with provider: {provider}, model: {web_model}")
    
    async def execute(self, query: str, num_results: int = 80, **kwargs) -> Dict[str, Any]:
        """Execute web search"""
        
        self._record_usage()
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            if self.provider == "serper":
                return await self._serper_search(query, num_results)
            elif self.provider == "valueserp":
                return await self._valueserp_search(query, num_results)
            elif self.provider == "perplexity":
                return await self._perplexity_search(query, 10)
            else:
                return {
                    "success": False,
                    "error": f"Provider {self.provider} not implemented"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Search failed: {str(e)}",
                "query": query,
                "provider": self.provider
            }
    
    async def _serper_search(self, query: str, num_results: int) -> Dict[str, Any]:
        """Search using Serper API"""
        
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "q": query,
            "num": min(num_results, 20)
        }
        
        async with self.session.post(
            "https://google.serper.dev/search", 
            headers=headers, 
            json=payload
        ) as response:
            
            if response.status != 200:
                raise ToolExecutionError(f"Serper API error: {response.status}")
            
            data = await response.json()
            
            results = []
            for item in data.get("organic", [])[:num_results]:
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "link": item.get("link", ""),
                    "position": item.get("position", 0)
                })
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "total_results": len(results),
                "provider": "serper"
            }
    
    async def _valueserp_search(self, query: str, num_results: int) -> Dict[str, Any]:
        """Search using ValueSerp API"""
        
        params = {
            "api_key": self.api_key,
            "q": query,
            "num": min(num_results, 20)
        }
        
        async with self.session.get(
            "https://api.valueserp.com/search", 
            params=params
        ) as response:
            
            if response.status != 200:
                raise ToolExecutionError(f"ValueSerp API error: {response.status}")
            
            data = await response.json()
            
            results = []
            for item in data.get("organic_results", []):
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "link": item.get("link", ""),
                    "position": item.get("position", len(results) + 1)
                })
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "total_results": len(results),
                "provider": "valueserp"
            }

    async def _perplexity_search(self, query: str, num_results: int) -> Dict[str, Any]:
        """Search using perplexity"""
        try:
            # Fixed: Use self.web_model instead of undefined self.model_name
            model_name = self.web_model if self.web_model else "perplexity/sonar"
            
            print(f"Using Perplexity model: {model_name} for query: {query}")
            
            response = await search_perplexity(query, model=model_name)
            
            # Create structured result matching other providers
            results = [{
                "title": f"Perplexity AI Search Results",
                "snippet": response,
                "link": "",
                "position": 1
            }]
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "total_results": 1,
                "provider": "perplexity",
                "model_used": model_name  # Include model info in response
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Perplexity search failed: {str(e)}",
                "query": query,
                "provider": "perplexity",
                "model_attempted": self.web_model
            }
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()

class RAGTool(BaseTool):
    """Simple RAG tool for knowledge retrieval"""
    
    def __init__(self, llm_client: LLMClient = None):
        super().__init__(
            "rag",
            "Retrieve information from knowledge base"
        )
        self.llm_client = llm_client
        self.knowledge_base = {
            "business_strategy": "Business strategy frameworks include SWOT analysis, Porter's Five Forces, and competitive positioning.",
            "market_analysis": "Market analysis examines market size, growth trends, customer segments, and competitive landscape.",
            "financial_analysis": "Financial analysis includes ratio analysis, cash flow analysis, and profitability assessment.",
            "ai_trends": "Current AI trends include large language models, multimodal AI, and AI safety research."
        }
    
    async def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute RAG query"""
        
        self._record_usage()
        
        try:
            # Simple keyword matching
            query_lower = query.lower()
            relevant_chunks = []
            
            for topic, content in self.knowledge_base.items():
                if any(word in content.lower() for word in query_lower.split()):
                    relevant_chunks.append(content)
            
            if not relevant_chunks:
                retrieved = "No specific knowledge found for this query."
            else:
                retrieved = "\n\n".join(relevant_chunks)
            
            return {
                "success": True,
                "retrieved": retrieved,
                "chunks": relevant_chunks,
                "query": query,
                "chunks_count": len(relevant_chunks)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"RAG query failed: {str(e)}",
                "query": query
            }

class ToolManager:
    """Manages all available tools"""
    
    def __init__(self, config, llm_client: LLMClient, web_model: str = None):
        self.config = config
        self.llm_client = llm_client
        self.web_model = web_model  # Store the selected web model
        self.tools: Dict[str, BaseTool] = {}
        self._initialize_tools()
        
        print(f"ToolManager initialized with web model: {web_model}")
    
    def _initialize_tools(self):
        """Initialize all configured tools"""
        
        tool_configs = self.config.get_tool_configs(self.web_model)
        
        print(f"Tool configs: {tool_configs}")
        
        # Calculator (always available)
        self.tools["calculator"] = CalculatorTool()
        
        # Web Search (if configured)
        web_config = tool_configs.get("web_search", {})
        if web_config.get("enabled"):
            # Check if we have an API key (for non-Perplexity providers)
            api_key = web_config.get("primary_key")
            
            # For Perplexity, we don't need a separate API key since it uses OpenRouter
            if web_config.get("provider") == "perplexity" or api_key:
                self.tools["web_search"] = WebSearchTool(
                    api_key or "",  # Empty string for Perplexity
                    web_model=web_config.get("web_model", self.web_model or "perplexity/sonar"),
                    provider=web_config.get("provider", "perplexity")
                )
                print(f"WebSearchTool created with model: {web_config.get('web_model', self.web_model)}")
            else:
                print("Web search disabled: No API key available")
        else:
            print("Web search disabled in config")
        
        # RAG Tool (always available)
        self.tools["rag"] = RAGTool(self.llm_client)
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get tool by name"""
        return self.tools.get(name)
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names"""
        return list(self.tools.keys())
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool by name"""
        
        tool = self.get_tool(tool_name)
        if not tool:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not available",
                "available_tools": self.get_available_tools()
            }
        
        try:
            result = await tool.execute(**kwargs)
            result["tool_name"] = tool_name
            return result
        except Exception as e:
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}",
                "tool_name": tool_name
            }
    
    async def cleanup(self):
        """Cleanup tool resources"""
        for tool in self.tools.values():
            if hasattr(tool, 'close'):
                await tool.close()