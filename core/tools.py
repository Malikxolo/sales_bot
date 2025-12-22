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
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod
from .exceptions import ToolExecutionError
from .quota_manager import QuotaManager
from .llm_client import LLMClient
from .knowledge_base import query_documents, get_collection_cache, get_org_cache
from .web_search_agent import search_perplexity, search_llmlayer
import ast
from redis.asyncio import Redis

logger = logging.getLogger(__name__)
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
    
    async def execute(self, query: str = None, expression: str = None, 
                      operation: str = None, numbers: List[float] = None, **kwargs) -> Dict[str, Any]:
        """Execute mathematical operations"""
        
        self._record_usage()
        if query and not expression:
            expression = query
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
        """Safely evaluate mathematical expression using AST parsing"""

        allowed_names = {
            "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
            "sqrt": math.sqrt, "pow": pow, "log": math.log, "exp": math.exp,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "pi": math.pi, "e": math.e
        }

        try:
            tree = ast.parse(expression, mode='eval')
            result = self._safe_eval_ast(tree.body, allowed_names)
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

    def _safe_eval_ast(self, node, allowed_names):
        """Recursively evaluate AST nodes safely"""

        if isinstance(node, ast.Num):  # e.g., 3, 4.5
            return node.n
        elif isinstance(node, ast.Constant):  # Python 3.8+
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numeric constants are allowed")

        elif isinstance(node, ast.BinOp):  # e.g., a + b
            left = self._safe_eval_ast(node.left, allowed_names)
            right = self._safe_eval_ast(node.right, allowed_names)

            if isinstance(node.op, ast.Add): return left + right
            elif isinstance(node.op, ast.Sub): return left - right
            elif isinstance(node.op, ast.Mult): return left * right
            elif isinstance(node.op, ast.Div): return left / right
            elif isinstance(node.op, ast.FloorDiv): return left // right
            elif isinstance(node.op, ast.Mod): return left % right
            elif isinstance(node.op, ast.Pow): return left ** right
            else:
                raise ValueError(f"Unsupported operator: {ast.dump(node.op)}")

        elif isinstance(node, ast.UnaryOp):  # e.g., -a
            operand = self._safe_eval_ast(node.operand, allowed_names)
            if isinstance(node.op, ast.UAdd): return +operand
            elif isinstance(node.op, ast.USub): return -operand
            else:
                raise ValueError(f"Unsupported unary operator: {ast.dump(node.op)}")

        elif isinstance(node, ast.Call):  # e.g., sin(x)
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only named functions are allowed")
            func_name = node.func.id
            if func_name not in allowed_names:
                raise ValueError(f"Function '{func_name}' is not allowed")

            args = [self._safe_eval_ast(arg, allowed_names) for arg in node.args]
            return allowed_names[func_name](*args)

        elif isinstance(node, ast.Name):  # e.g., pi, e
            if node.id in allowed_names:
                return allowed_names[node.id]
            raise ValueError(f"Use of unknown variable '{node.id}'")

        else:
            raise ValueError(f"Unsupported expression element: {ast.dump(node)}")

    async def _perform_operation(self, operation: str, numbers: List[float]) -> Dict[str, Any]:
        """Perform statistical operations"""
        
        if not numbers:
            return {"success": False, "error": "No numbers provided"}
        
        try:
            operations_map = {
                "mean": statistics.mean,
                "median": statistics.median,
                "mode": statistics.mode,
                "stdev": lambda x: statistics.stdev(x) if len(x) > 1 else 0,
                "variance": lambda x: statistics.variance(x) if len(x) > 1 else 0,
                "sum": sum,
                "min": min,
                "max": max
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
    """
    Web search tool with multi-provider support and quota management
      Integrated: Quota-aware provider selection with auto-fallback
      Added: Google CSE, Brave, Serper support alongside existing providers
    """
    
    def __init__(
        self, 
        provider: str = "auto",  # "auto" enables quota-aware selection
        web_model: str = None,
        # API Keys
        google_cse_key: str = None,
        google_cse_id: str = None,
        brave_key: str = None,
        scrapingdog_key: str = None,
        serper_key: str = None,
        valueserp_key: str = None,
        perplexity_key: str = None,
        llmlayer_key: str = None,
        llmlayer_url: str = None,
        jina_api_key: str = None
    ):
        super().__init__(
            "web_search", 
            "Search the internet for current information with intelligent quota management"
        )
        
        # Store API keys
        self.google_cse_key = google_cse_key or os.getenv("GOOGLE_CSE_KEY")
        self.google_cse_id = google_cse_id or os.getenv("GOOGLE_CSE_ID")
        self.brave_key = brave_key or os.getenv("BRAVE_API_KEY")
        self.scrapingdog_key = scrapingdog_key or os.getenv("SCRAPINGDOG_API_KEY")
        self.serper_key = serper_key or os.getenv("SERPER_API_KEY")
        self.valueserp_key = valueserp_key or os.getenv("VALUESERP_API_KEY")
        self.perplexity_key = perplexity_key or os.getenv("PERPLEXITY_API_KEY")
        self.llmlayer_key = llmlayer_key or os.getenv("LLMLAYER_API_KEY")
        self.llmlayer_url = llmlayer_url or os.getenv("LLMLAYER_API_URL", "https://api.llmlayer.dev/api/v2/answer")
        self.jina_api_key = jina_api_key or os.getenv("JINA_API_KEY")
        
        self.provider = provider
        self.web_model = web_model
        self.session = None
        self.quota_manager = QuotaManager()
        
        
        # Build available providers list (priority order)
        self.available_providers = []
        
        if self.llmlayer_key:
            self.available_providers.append("llmlayer")
        if self.google_cse_key and self.google_cse_id:
            self.available_providers.append("google_cse")
        if self.brave_key:
            self.available_providers.append("brave")
        if self.scrapingdog_key:
            self.available_providers.append("scrapingdog")
        if self.serper_key:
            self.available_providers.append("serper")
        if self.valueserp_key:
            self.available_providers.append("valueserp")
        if self.perplexity_key:
            self.available_providers.append("perplexity")
        
        # Statistics
        self.stats = {
            "llmlayer_success": 0,
            "llmlayer_failed": 0,
            "google_cse_success": 0,
            "google_cse_failed": 0,
            "brave_success": 0,
            "brave_failed": 0,
            "scrapingdog_success": 0,
            "scrapingdog_failed": 0,
            "serper_success": 0,
            "serper_failed": 0,
            "valueserp_success": 0,
            "valueserp_failed": 0,
            "perplexity_success": 0,
            "perplexity_failed": 0,
            "total_searches": 0,
            "total_scraped": 0,
            "fallback_attempts": 0
        }
        
        if not self.available_providers:
            logger.warning("⚠️ No search providers configured!")
        else:
            logger.info(f"🔍 WebSearchTool initialized with {len(self.available_providers)} provider(s)")
            logger.info(f"   📋 Available: {' → '.join(self.available_providers)}")
            if self.quota_manager:
                logger.info(f"   💰 Quota management: ENABLED")
            if self.provider != "auto":
                logger.info(f"   🎯 Fixed provider mode: {self.provider}")
    
    async def execute(self, query: str, num_results: int = 10, scrape_top: int = 3, **kwargs) -> Dict[str, Any]:
        """
        Execute web search with intelligent provider selection
        
        Args:
            query: Search query
            num_results: Number of results to return
            scrape_top: Number of top results to scrape with Jina
        """
        
        self._record_usage()
        self.stats["total_searches"] += 1
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        if not self.available_providers:
            return {
                "success": False,
                "error": "No search providers configured",
                "query": query
            }
        
        # Determine provider selection strategy
        if self.provider == "auto":
            # Quota-aware automatic selection
            providers_to_try = self._get_provider_order()
        elif self.provider in self.available_providers:
            # Fixed provider with fallback
            providers_to_try = [self.provider] + [p for p in self.available_providers if p != self.provider]
        else:
            # Fallback to all available
            providers_to_try = self.available_providers
        
        logger.info(f"🔍 Search query: {query[:50]}...")
        logger.info(f"   📊 Provider order: {' → '.join(providers_to_try)}")
        
        # Try each provider in order
        last_error = None
        for idx, provider in enumerate(providers_to_try):
            if idx > 0:
                self.stats["fallback_attempts"] += 1
                logger.info(f"   ⏭️  Attempting fallback to {provider.upper()}...")
            
            try:
                logger.info(f"   🔄 Using {provider.upper()}...")
                
                # Execute search based on provider
                if provider == "llmlayer":
                    result = await self._llmlayer_search(query)
                elif provider == "google_cse":
                    result = await self._google_cse_search(query, num_results, scrape_top)
                elif provider == "brave":
                    result = await self._brave_search(query, num_results, scrape_top)
                elif provider == "scrapingdog":
                    result = await self._scrapingdog_search(query, num_results, scrape_top)
                elif provider == "serper":
                    result = await self._serper_search(query, num_results, scrape_top)
                elif provider == "valueserp":
                    result = await self._valueserp_search(query, num_results, scrape_top)
                elif provider == "perplexity":
                    result = await self._perplexity_search(query, num_results)
                else:
                    continue
                
                # Validate results
                if result.get("success") and self._validate_results(result):
                    self.stats[f"{provider}_success"] += 1
                    
                    # Record successful usage with quota manager
                    if self.quota_manager:
                        self.quota_manager.record_usage(provider, num_queries=1, success=True)
                    
                    logger.info(f"     {provider.upper()}: {result.get('total_results', 0)} results, "
                              f"{result.get('scraped_count', 0)} scraped")
                    return result
                else:
                    logger.warning(f"   ⚠️ {provider.upper()} returned invalid results")
                    self.stats[f"{provider}_failed"] += 1
                    
                    if self.quota_manager:
                        self.quota_manager.record_usage(provider, num_queries=1, success=False)
                    
                    last_error = result.get("error", "Invalid results")
                    
            except Exception as e:
                error_msg = str(e)[:100]
                logger.warning(f"   ❌ {provider.upper()} failed: {error_msg}")
                self.stats[f"{provider}_failed"] += 1
                
                if self.quota_manager:
                    self.quota_manager.record_usage(provider, num_queries=1, success=False)
                
                last_error = error_msg
                continue
        
        # All providers failed
        logger.error(f"❌ All {len(providers_to_try)} provider(s) failed")
        return {
            "success": False,
            "error": f"All providers failed. Last error: {last_error}",
            "query": query,
            "providers_attempted": providers_to_try
        }
    
    def _get_provider_order(self) -> List[str]:
        """Get quota-aware provider order"""
        if self.quota_manager:
            # Use quota manager to get best available provider
            best_provider = self.quota_manager.get_available_provider(self.available_providers)
            if best_provider:
                # Put best provider first, then others
                return [best_provider] + [p for p in self.available_providers if p != best_provider]
            else:
                logger.warning("⚠️ All free tiers exhausted, trying all providers")
        
        # Default priority order
        return self.available_providers
    
    def _validate_results(self, result: Dict[str, Any]) -> bool:
        """Validate search results quality"""
        if not result.get("success"):
            return False
        
        results = result.get("results", [])
        if not results:
            return False
        
        # Check for meaningful content
        valid_count = sum(
            1 for r in results 
            if r.get("title") and r.get("snippet") and len(r.get("snippet", "")) > 20
        )
        
        return valid_count >= min(3, len(results))
    
    # ==================== SEARCH PROVIDER IMPLEMENTATIONS ====================
    
    async def _google_cse_search(self, query: str, num_results: int, scrape_top: int) -> Dict[str, Any]:
        """Google Custom Search API - 100 free queries/day"""
        
        params = {
            "key": self.google_cse_key,
            "cx": self.google_cse_id,
            "q": query,
            "num": min(num_results, 10),
            "gl": "in",
            "safe": "off"
        }
        
        async with self.session.get(
            "https://www.googleapis.com/customsearch/v1",
            params=params
        ) as response:
            
            if response.status != 200:
                error_data = await response.json()
                raise ToolExecutionError(
                    f"Google CSE API error: {error_data.get('error', {}).get('message', response.status)}"
                )
            
            data = await response.json()
            
            if "error" in data:
                raise ToolExecutionError(data["error"].get("message", "Unknown error"))
            
            items = data.get("items", [])[:num_results]
            
            results = [
                {
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "link": item.get("link", ""),
                    "displayLink": item.get("displayLink", ""),
                    "position": idx + 1
                }
                for idx, item in enumerate(items)
            ]
            
            # Scrape top results
            scraped_count = await self._scrape_results(results, scrape_top)
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "total_results": len(results),
                "scraped_count": scraped_count,
                "provider": "google_cse"
            }
    
    async def _brave_search(self, query: str, num_results: int, scrape_top: int) -> Dict[str, Any]:
        """Brave Search API - 2000 free queries/month"""
        
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.brave_key
        }
        
        params = {
            "q": query,
            "count": num_results,
            "country": "IN",
            "search_lang": "en",
            "safesearch": "off"
        }
        
        async with self.session.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers=headers,
            params=params
        ) as response:
            
            if response.status != 200:
                raise ToolExecutionError(f"Brave API error: {response.status}")
            
            data = await response.json()
            items = data.get("web", {}).get("results", [])[:num_results]
            
            results = [
                {
                    "title": item.get("title", ""),
                    "snippet": item.get("description", ""),
                    "link": item.get("url", ""),
                    "displayLink": item.get("profile", {}).get("name", ""),
                    "position": idx + 1
                }
                for idx, item in enumerate(items)
            ]
            
            # Scrape top results
            scraped_count = await self._scrape_results(results, scrape_top)
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "total_results": len(results),
                "scraped_count": scraped_count,
                "provider": "brave"
            }
    
    async def _scrapingdog_search(self, query: str, num_results: int, scrape_top: int) -> Dict[str, Any]:
        """ScrapingDog Google SERP API - 200 free searches"""
        
        params = {
            "api_key": self.scrapingdog_key,
            "query": query,
            "results": min(num_results, 20),
            "page": "0",
            "country": "in"
        }
        
        async with self.session.get(
            "https://api.scrapingdog.com/google/", 
            params=params
        ) as response:
            
            if response.status != 200:
                raise ToolExecutionError(f"ScrapingDog API error: {response.status}")
            
            data = await response.json()
            
            results = [
                {
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "link": item.get("link", ""),
                    "displayLink": item.get("displayed_link", ""),
                    "position": item.get("rank", idx + 1)
                }
                for idx, item in enumerate(data.get("organic_results", [])[:num_results])
            ]
            
            # Scrape top results
            scraped_count = await self._scrape_results(results, scrape_top)
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "total_results": len(results),
                "scraped_count": scraped_count,
                "provider": "scrapingdog"
            }
    
    async def _serper_search(self, query: str, num_results: int, scrape_top: int) -> Dict[str, Any]:
        """Serper.dev Google Search API - 2500 free queries"""
        
        headers = {
            "X-API-KEY": self.serper_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "q": query,
            "num": num_results,
            "gl": "in",
            "hl": "en"
        }
        
        async with self.session.post(
            "https://google.serper.dev/search",
            headers=headers,
            json=payload
        ) as response:
            
            if response.status != 200:
                raise ToolExecutionError(f"Serper API error: {response.status}")
            
            data = await response.json()
            
            if not data.get("organic"):
                raise ToolExecutionError("No results from Serper")
            
            items = data.get("organic", [])[:num_results]
            
            results = [
                {
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "link": item.get("link", ""),
                    "displayLink": item.get("displayedLink", ""),
                    "position": item.get("position", idx + 1)
                }
                for idx, item in enumerate(items)
            ]
            
            # Scrape top results
            scraped_count = await self._scrape_results(results, scrape_top)
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "total_results": len(results),
                "scraped_count": scraped_count,
                "provider": "serper"
            }
    
    async def _valueserp_search(self, query: str, num_results: int, scrape_top: int) -> Dict[str, Any]:
        """ValueSerp API"""
        
        params = {
            "api_key": self.valueserp_key,
            "q": query,
            "num": min(num_results, 20),
            "gl": "in"
        }
        
        async with self.session.get(
            "https://api.valueserp.com/search", 
            params=params
        ) as response:
            
            if response.status != 200:
                raise ToolExecutionError(f"ValueSerp API error: {response.status}")
            
            data = await response.json()
            
            results = [
                {
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "link": item.get("link", ""),
                    "displayLink": item.get("domain", ""),
                    "position": item.get("position", idx + 1)
                }
                for idx, item in enumerate(data.get("organic_results", [])[:num_results])
            ]
            
            # Scrape top results
            scraped_count = await self._scrape_results(results, scrape_top)
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "total_results": len(results),
                "scraped_count": scraped_count,
                "provider": "valueserp"
            }
    
    async def _perplexity_search(self, query: str, num_results: int) -> Dict[str, Any]:
        """Perplexity AI Search (returns synthesized answer)"""
        
        try:
            model_name = self.web_model if self.web_model else "perplexity/sonar"
            
            logger.info(f"   📡 Using Perplexity model: {model_name}")
            
            response = await search_perplexity(query, model=model_name)
            
            results = [{
                "title": "Perplexity AI Search Results",
                "snippet": response,
                "link": "",
                "position": 1
            }]
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "total_results": 1,
                "scraped_count": 0,
                "provider": "perplexity",
                "model_used": model_name
            }
            
        except Exception as e:
            raise ToolExecutionError(f"Perplexity search failed: {str(e)}")
    
    async def _llmlayer_search(self, query: str) -> Dict[str, Any]:
        """LLMLayer Search (returns pre-formatted answer, no scraping needed)"""
        
        try:
            logger.info(f"   🌐 Using LLMLayer...")
            
            response = await search_llmlayer(query, self.llmlayer_key, self.llmlayer_url)
            answer, sources = response.get("answer", ""), response.get("sources", [])
            
            return {
                "success": True,
                "query": query,
                "results": sources,
                "total_results": 1,
                "scraped_count": 0,
                "provider": "llmlayer",
                "llm_response": answer
            }
            
        except Exception as e:
            raise ToolExecutionError(f"LLMLayer search failed: {str(e)}")
    
    # ==================== SCRAPING UTILITIES ====================
    
    async def _scrape_results(self, results: List[Dict], scrape_top: int) -> int:
        """
        Scrape top N results concurrently with Jina Reader
        
        Returns: Number of successfully scraped pages
        """
        
        if scrape_top <= 0 or not self.jina_api_key or not results:
            return 0
        
        logger.info(f"   🔄 Scraping top {scrape_top} results with Jina...")
        
        # Create scraping tasks
        scrape_tasks = []
        urls_to_scrape = []
        
        for i, result in enumerate(results[:scrape_top]):
            url = result.get("link", "")
            if url:
                urls_to_scrape.append((i, url))
                scrape_tasks.append(self._scrape_with_jina(url))
        
        if not scrape_tasks:
            return 0
        
        # Execute all scraping tasks concurrently
        scraped_results = await asyncio.gather(*scrape_tasks, return_exceptions=True)
        
        # Assign scraped content back to results
        scraped_count = 0
        for (idx, url), scraped in zip(urls_to_scrape, scraped_results):
            if isinstance(scraped, Exception):
                logger.debug(f"      ❌ [{idx+1}] Scraping failed: {str(scraped)[:50]}")
                results[idx]["scraped_content"] = f"[Scraping failed]"
            elif scraped.startswith("["):
                logger.debug(f"      ⚠️ [{idx+1}] {scraped}")
                results[idx]["scraped_content"] = scraped
            else:
                results[idx]["scraped_content"] = scraped[:20000]  # Limit to 20000 chars
                scraped_count += 1
                self.stats["total_scraped"] += 1
                logger.debug(f"       [{idx+1}] Scraped {len(scraped)} chars")
        
        if scraped_count > 0:
            logger.info(f"    Successfully scraped {scraped_count}/{scrape_top} pages")
        
        return scraped_count
    
    async def _scrape_with_jina(self, url: str) -> str:
        """Scrape URL using Jina Reader API"""
        
        try:
            jina_url = f"https://r.jina.ai/{url}"
            
            headers = {
                "Authorization": f"Bearer {self.jina_api_key}",
                "X-Return-Format": "markdown",
                "X-With-Generated-Alt": "true",
                "X-Timeout": "8",
                "Accept": "text/plain"
            }
            
            async with self.session.get(jina_url, headers=headers, timeout=15) as response:
                if response.status == 200:
                    content = await response.text()
                    if len(content) < 100:
                        return "[Content too short]"
                    return content.strip()
                else:
                    return f"[HTTP {response.status}]"
                    
        except asyncio.TimeoutError:
            return "[Timeout]"
        except Exception as e:
            return f"[Error: {str(e)[:50]}]"
    
    # ==================== STATISTICS ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        
        total_attempts = self.stats["total_searches"]
        total_success = sum(
            self.stats[f"{p}_success"] 
            for p in ["llmlayer", "google_cse", "brave", "scrapingdog", "serper", "valueserp", "perplexity"]
        )
        
        return {
            **self.stats,
            "providers_configured": len(self.available_providers),
            "providers": self.available_providers,
            "success_rate": f"{(total_success / max(1, total_attempts) * 100):.1f}%",
            "avg_scraped_per_search": f"{(self.stats['total_scraped'] / max(1, total_success)):.1f}",
        }
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            logger.debug("🔒 WebSearchTool session closed")

class RAGTool(BaseTool):
    """RAG tool using user-specific ChromaDB collections"""
    def __init__(self, llm_client: LLMClient = None):
        super().__init__("rag", "Retrieve information from uploaded knowledge base")
        self.llm_client = llm_client
        logger.info("RAGTool initialized")
    
    async def execute(self, query: str, user_id: str = None, **kwargs) -> Dict[str, Any]:
        """Execute RAG query on user's vector database"""
        self._record_usage()
        logger.info(f"RAG query started: user_id={user_id}, query='{query[:50]}...'")
        
        try:
            if not user_id:
                logger.error("RAG query failed: User ID missing")
                return {
                    "success": False,
                    "error": "User ID required for RAG queries",
                    "query": query
                }
            
            
            logger.debug(f"Checking collections for user: {user_id}")
            org_id = await get_org_cache(user_id)
            if not org_id:
                org_id = "org_global"
            collection_name = await get_collection_cache(user_id)
            if not collection_name:
                collection_name = "global_collection"
                user_id = "system"

            # Query the user's collection
            result = await query_documents(org_id, collection_name, query, user_id, n_results=5)
            
            if result["success"]:
                chunks_count = len(result["results"])
                distances = result.get("distances", [])
                
                # Log success with detailed metrics
                logger.info(f" RAG query SUCCESS for user {user_id}")
                logger.info(f"   Collection: {collection_name}")
                logger.info(f"   Retrieved chunks: {chunks_count}")
                logger.info(f"   Query: '{query[:50]}...'")
                
                # Log distances for relevance analysis
                if distances:
                    avg_distance = sum(distances) / len(distances)
                    min_distance = min(distances)
                    max_distance = max(distances)
                    
                    logger.info(f"   Distance metrics - Min: {min_distance:.4f}, Max: {max_distance:.4f}, Avg: {avg_distance:.4f}")
                    logger.debug(f"   All distances: {distances}")
                    
                    # Log quality assessment
                    if avg_distance < 0.3:
                        logger.info("   Quality: HIGH relevance (avg distance < 0.3)")
                    elif avg_distance < 0.6:
                        logger.info("   Quality: MEDIUM relevance (avg distance < 0.6)")
                    else:
                        logger.warning("   Quality: LOW relevance (avg distance >= 0.6)")
                else:
                    logger.warning("   No distance information available")
                
                # Log first chunk preview for debugging
                documents = []
                if result["results"]:
                    documents = [r["document"] for r in result["results"] if isinstance(r, dict) and "document" in r]
                    first_chunk = documents[0][:200] + ("..." if len(documents[0]) > 200 else "")   
                    logger.info(f"   First chunk preview: '{first_chunk}'")
                    logger.info(f"   Total retrieved documents: {len(documents)}")
                
                return {
                    "success": True,
                    "retrieved": "\n\n".join(documents),
                    "chunks": result["results"],
                    "query": query,
                    "chunks_count": chunks_count,
                    "collection": collection_name,
                    "distances": distances,
                    "avg_distance": avg_distance if distances else None
                }
            else:
                logger.error(f"❌ RAG query FAILED for user {user_id}")
                logger.error(f"   Collection: {collection_name}")
                logger.error(f"   Error: {result.get('error', 'Unknown error')}")
                logger.error(f"   Query: '{query[:50]}...'")
                
                return {
                    "success": False,
                    "error": result["error"],
                    "query": query
                }
                
        except Exception as e:
            logger.error(f"❌ RAG query EXCEPTION for user {user_id}")
            logger.error(f"   Exception: {str(e)}")
            logger.error(f"   Query: '{query[:50]}...'")
            
            # Log full traceback for debugging
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            
            return {
                "success": False,
                "error": f"RAG query failed: {str(e)}",
                "query": query
            }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"RAG query failed: {str(e)}",
                "query": query
            }

class ToolManager:
    """
    Manages all available tools with multi-provider support
      Updated: Passes all SERP provider keys to WebSearchTool
      NEW: Zapier MCP integration for 8000+ app actions
      NEW: Grievance tool for DM grievance tracking
    """
    
    def __init__(
        self, 
        config, 
        llm_client: LLMClient, 
        web_model: str = None, 
        use_premium_search: bool = False,
        enable_zapier: bool = None  # Auto-detect from env if None
    ):
        self.config = config
        self.llm_client = llm_client
        self.web_model = web_model
        self.use_premium_search = use_premium_search
        self.tools: Dict[str, BaseTool] = {}
        
        # Zapier integration
        self._zapier_manager = None
        self._zapier_enabled = enable_zapier
        
        # MongoDB integration
        self._mongodb_manager = None
        self._query_agent = None
        
        # Redis integration
        self._redis_manager = None
        
        # Grievance integration
        self._grievance_agent = None
        self._grievance_enabled = None  # Auto-detect from env
        
        self._initialize_tools()
        
        logger.info(f"ToolManager initialized with web model: {web_model}")
    
    def _initialize_tools(self):
        """Initialize all configured tools"""
        
        tool_configs = self.config.get_tool_configs(self.web_model, self.use_premium_search)
        
        logger.debug(f"Tool configs: {tool_configs}")
        
        # Calculator (always available)
        self.tools["calculator"] = CalculatorTool()
        logger.info("  Calculator tool initialized")
        
        # Web Search (if configured)
        self._initialize_web_search(tool_configs)
        
        # RAG Tool (always available)
        self.tools["rag"] = RAGTool(self.llm_client)
        logger.info("  RAG tool initialized")
        
        # Note: Zapier tools initialized separately via initialize_zapier_async()
    
    def _initialize_web_search(self, tool_configs: Dict[str, Any]):
        """Initialize web search with multi-provider support"""
        
        # Check explicit WEB_SEARCH_ENABLED toggle first
        web_search_enabled = os.getenv("WEB_SEARCH_ENABLED", "true").lower() == "true"
        
        if not web_search_enabled:
            logger.info("⚠️ Web search DISABLED via WEB_SEARCH_ENABLED=false")
            return
        
        web_config = tool_configs.get("web_search", {})
        
        if not web_config.get("enabled"):
            logger.info("Web search disabled in config")
            return
        
        # Collect all available API keys from config
        # Priority: Environment variables > Config file
        
        # Google Custom Search
        google_cse_key = os.getenv("GOOGLE_CSE_KEY") or web_config.get("google_cse_key")
        google_cse_id = os.getenv("GOOGLE_CSE_ID") or web_config.get("google_cse_id")
        
        # Brave Search
        brave_key = os.getenv("BRAVE_API_KEY") or web_config.get("brave_key")
        
        # ScrapingDog
        scrapingdog_key = os.getenv("SCRAPINGDOG_API_KEY") or web_config.get("scrapingdog_key") or web_config.get("primary_key")
        
        # Serper
        serper_key = os.getenv("SERPER_API_KEY") or web_config.get("serper_key")
        
        # ValueSerp
        valueserp_key = os.getenv("VALUESERP_API_KEY") or web_config.get("valueserp_key")
        
        # Perplexity
        perplexity_key = os.getenv("PERPLEXITY_API_KEY") or web_config.get("perplexity_key")
        
        # LLMLayer
        llmlayer_key = os.getenv("LLMLAYER_API_KEY") or web_config.get("llmlayer_key")
        llmlayer_url = os.getenv("LLMLAYER_API_URL") or web_config.get("llmlayer_url", "https://api.llmlayer.dev/api/v2/answer")
        
        # Jina Reader (for scraping)
        jina_api_key = os.getenv("JINA_API_KEY") or web_config.get("jina_key")
        
        # Count available providers
        available_providers = []
        if llmlayer_key:
            available_providers.append("llmlayer")
        if google_cse_key and google_cse_id:
            available_providers.append("google_cse")
        if brave_key:
            available_providers.append("brave")
        if scrapingdog_key:
            available_providers.append("scrapingdog")
        if serper_key:
            available_providers.append("serper")
        if valueserp_key:
            available_providers.append("valueserp")
        if perplexity_key:
            available_providers.append("perplexity")
        
        # Check if we have at least one provider
        if not available_providers:
            logger.warning("⚠️ Web search enabled but no API keys configured")
            logger.info("💡 Add API keys for: LLMLayer, Google CSE, Brave, ScrapingDog, Serper, ValueSerp, or Perplexity")
            return
        
        # Determine provider mode
        provider_mode = web_config.get("provider", "auto")
        
        # Validate provider mode
        if provider_mode not in ["auto"] + available_providers:
            logger.warning(f"⚠️ Configured provider '{provider_mode}' not available, using 'auto'")
            provider_mode = "auto"
        
        # Initialize WebSearchTool with all keys
        try:
            self.tools["web_search"] = WebSearchTool(
                provider=provider_mode,
                web_model=web_config.get("web_model", self.web_model or "perplexity/sonar"),
                # Pass all API keys
                google_cse_key=google_cse_key,
                google_cse_id=google_cse_id,
                brave_key=brave_key,
                scrapingdog_key=scrapingdog_key,
                serper_key=serper_key,
                valueserp_key=valueserp_key,
                perplexity_key=perplexity_key,
                llmlayer_key=llmlayer_key,
                llmlayer_url=llmlayer_url,
                jina_api_key=jina_api_key
            )
            
            logger.info("  Web search tool initialized")
            logger.info(f"   🔍 Mode: {provider_mode}")
            logger.info(f"   📋 Available providers: {', '.join(available_providers)}")
            if jina_api_key:
                logger.info(f"   🔧 Jina scraping: ENABLED")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize web search: {str(e)}")
    
    async def initialize_zapier_async(self) -> bool:
        """
        Initialize Zapier MCP integration (async).
        
        Call this after ToolManager creation to enable Zapier tools.
        Zapier requires async initialization for network calls.
        
        Returns:
            True if Zapier initialized successfully
        """
        # Check if Zapier should be enabled
        if self._zapier_enabled is None:
            # Check explicit ZAPIER_ENABLED toggle first
            zapier_explicit = os.getenv("ZAPIER_ENABLED", "true").lower() == "true"
            if not zapier_explicit:
                logger.info("ℹ️ Zapier MCP integration disabled (ZAPIER_ENABLED=false)")
                self._zapier_enabled = False
                return False
            
            # Auto-detect from environment
            mcp_enabled = os.getenv("MCP_ENABLED", "false").lower() == "true"
            zapier_url = os.getenv("ZAPIER_MCP_SERVER_URL")
            self._zapier_enabled = mcp_enabled and bool(zapier_url)
        
        if not self._zapier_enabled:
            logger.info("ℹ️ Zapier MCP integration disabled (set MCP_ENABLED=true and ZAPIER_MCP_SERVER_URL)")
            return False
        
        try:
            # Import here to avoid circular imports and make it optional
            from .mcp import ZapierToolManager, MCPSecurityManager
            
            logger.info("🔄 Initializing Zapier MCP integration...")
            
            # Create security manager
            security = MCPSecurityManager()
            
            if not security.is_zapier_configured():
                logger.warning("⚠️ Zapier MCP not configured - skipping")
                return False
            
            # Create and initialize Zapier tool manager
            self._zapier_manager = ZapierToolManager(security)
            initialized = await self._zapier_manager.initialize()
            
            if initialized:
                zapier_tools = self._zapier_manager.get_tool_names()
                logger.info(f"  Zapier MCP integration initialized with {len(zapier_tools)} tools")
                
                # Log some example tools
                if zapier_tools:
                    examples = zapier_tools[:5]
                    logger.info(f"   📋 Example tools: {', '.join(examples)}")
                
                return True
            else:
                logger.warning("⚠️ Zapier MCP initialization failed")
                return False
                
        except ImportError as e:
            logger.warning(f"⚠️ Zapier MCP module not available: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Zapier MCP initialization error: {e}")
            return False
    
    async def initialize_mongodb_async(self) -> bool:
        """
        Initialize MongoDB MCP integration (async).
        
        Call this after ToolManager creation to enable MongoDB tools.
        MongoDB requires async initialization for MCP server connection.
        
        Returns:
            True if MongoDB initialized successfully
        """
        # Check if MongoDB is explicitly disabled via .env
        mongodb_enabled = os.getenv("MONGODB_ENABLED", "true").lower() == "true"
        if not mongodb_enabled:
            logger.info("ℹ️ MongoDB MCP integration disabled (MONGODB_ENABLED=false)")
            return False
        
        # Check if MongoDB connection string is configured
        connection_string = os.getenv("MONGODB_MCP_CONNECTION_STRING")
        
        if not connection_string:
            logger.info("ℹ️ MongoDB MCP integration disabled (MONGODB_MCP_CONNECTION_STRING not set)")
            return False
        
        try:
            # Import here to avoid circular imports
            from .mcp.mongodb import MongoDBMCPClient
            from .mcp.query_agent import QueryAgent
            
            logger.info("🔄 Initializing MongoDB MCP integration...")
            
            # Create MongoDB MCP client
            self._mongodb_manager = MongoDBMCPClient(connection_string=connection_string)
            
            # Connect to MCP server
            connected = await self._mongodb_manager.connect()
            
            if connected:
                # CRITICAL: The MCP server is running but NOT connected to MongoDB yet!
                # We need to explicitly call the 'connect' tool to establish database connection
                logger.info("🔗 Establishing database connection...")
                
                try:
                    connect_result = await self._mongodb_manager.execute_tool("connect", {
                        "connectionString": connection_string
                    })
                    
                    # Verify connection by listing databases
                    verify_result = await self._mongodb_manager.execute_tool("list-databases", {})
                    
                    if hasattr(verify_result, 'result'):
                        result_str = str(verify_result.result).lower()
                        if "you need to connect" in result_str:
                            logger.error("❌ MongoDB database connection failed - server says 'need to connect'")
                            return False
                        else:
                            logger.info("  MongoDB database connection verified!")
                    
                except Exception as conn_err:
                    logger.warning(f"⚠️ Database connection step: {conn_err}")
                    # Continue anyway - some MCP versions may auto-connect
                
                # Create QueryAgent with shared LLM client
                self._query_agent = QueryAgent(llm_client=self.llm_client)
                
                tools = await self._mongodb_manager.list_tools()
                logger.info(f"  MongoDB MCP integration initialized with {len(tools)} tools")
                
                # Log some example tools
                if tools:
                    examples = [t.name for t in tools[:5]]
                    logger.info(f"   📋 Example tools: {', '.join(examples)}")
                
                return True
            else:
                logger.warning("⚠️ MongoDB MCP connection failed")
                return False
                
        except ImportError as e:
            logger.warning(f"⚠️ MongoDB MCP module not available: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ MongoDB MCP initialization error: {e}")
            return False
    
    async def initialize_redis_async(self) -> bool:
        """
        Initialize Redis MCP integration (async).
        
        Call this after ToolManager creation to enable Redis tools.
        Redis requires async initialization for MCP server connection.
        
        Returns:
            True if Redis initialized successfully
        """
        # Check if Redis is explicitly disabled via .env
        redis_enabled = os.getenv("REDIS_ENABLED", "true").lower() == "true"
        if not redis_enabled:
            logger.info("ℹ️ Redis MCP integration disabled (REDIS_ENABLED=false)")
            return False
        
        # Check if Redis URL is configured
        redis_url = os.getenv("REDIS_MCP_URL")
        
        if not redis_url:
            logger.info("ℹ️ Redis MCP integration disabled (REDIS_MCP_URL not set)")
            return False
        
        try:
            # Import here to avoid circular imports
            from .mcp.redis import RedisMCPClient
            from .mcp.query_agent import QueryAgent
            
            logger.info("🔄 Initializing Redis MCP integration...")
            
            # Create Redis MCP client
            self._redis_manager = RedisMCPClient(redis_url=redis_url)
            
            # Connect to MCP server
            connected = await self._redis_manager.connect()
            
            if connected:
                # Create QueryAgent with shared LLM client if not already created
                if self._query_agent is None:
                    self._query_agent = QueryAgent(llm_client=self.llm_client)
                
                tools = await self._redis_manager.list_tools()
                logger.info(f"  Redis MCP integration initialized with {len(tools)} tools")
                
                # Log some example tools
                if tools:
                    examples = [t.name for t in tools[:5]]
                    logger.info(f"   📋 Example tools: {', '.join(examples)}")
                
                return True
            else:
                logger.warning("⚠️ Redis MCP connection failed")
                return False
                
        except ImportError as e:
            logger.warning(f"⚠️ Redis MCP module not available: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Redis MCP initialization error: {e}")
            return False
    
    async def initialize_grievance_async(self) -> bool:
        """
        Initialize Grievance tool (async).
        
        Call this after ToolManager creation to enable grievance parameter extraction.
        
        Returns:
            True if Grievance initialized successfully
        """
        # Check if Grievance is explicitly disabled via .env
        grievance_enabled = os.getenv("GRIEVANCE_ENABLED", "true").lower() == "true"
        if not grievance_enabled:
            logger.info("ℹ️ Grievance tool disabled (GRIEVANCE_ENABLED=false)")
            self._grievance_enabled = False
            return False
        
        try:
            # Import here to avoid circular imports
            from .grievance_query_agent import GrievanceAgent
            
            logger.info("🔄 Initializing Grievance tool...")
            
            # Get grievance LLM config from .env
            grievance_provider = os.getenv("GRIEVANCE_LLM_PROVIDER", "openrouter")
            grievance_model = os.getenv("GRIEVANCE_LLM_MODEL", "meta-llama/llama-4-maverick")
            
            # Create custom LLM config for grievance
            from core.config import Config
            config = Config()
            grievance_llm_config = config.create_llm_config(
                provider=grievance_provider,
                model=grievance_model,
                max_tokens=2048
            )
            grievance_llm = LLMClient(grievance_llm_config)
            
            # Create GrievanceAgent with dedicated LLM client
            self._grievance_agent = GrievanceAgent(llm_client=grievance_llm)
            self._grievance_enabled = True
            
            logger.info("✅ Grievance tool initialized")
            return True
                
        except ImportError as e:
            logger.warning(f"⚠️ Grievance module not available: {e}")
            self._grievance_enabled = False
            return False
        except Exception as e:
            logger.error(f"❌ Grievance initialization error: {e}")
            self._grievance_enabled = False
            return False
    
    @property
    def mongodb_available(self) -> bool:
        """Check if MongoDB tools are available"""
        return self._mongodb_manager is not None and self._mongodb_manager.is_connected
    
    @property
    def redis_available(self) -> bool:
        """Check if Redis tools are available"""
        return self._redis_manager is not None and self._redis_manager.is_connected
    
    @property
    def zapier_available(self) -> bool:
        """Check if Zapier tools are available"""
        return self._zapier_manager is not None and self._zapier_manager.is_available
    
    @property
    def grievance_available(self) -> bool:
        """Check if Grievance tool is available"""
        return self._grievance_agent is not None and self._grievance_enabled
    
    def get_zapier_tools(self) -> List[str]:
        """Get list of available Zapier tool names"""
        if self._zapier_manager:
            return self._zapier_manager.get_tool_names()
        return []
    
    def get_zapier_tool_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get Zapier tool schemas for LLM prompts"""
        if self._zapier_manager:
            return self._zapier_manager.get_tool_schemas()
        return {}
    
    def get_zapier_tools_prompt(self) -> str:
        """
        Get dynamic prompt section for all Zapier tools.
        
        This returns a formatted string for LLM prompts that automatically
        includes ALL available Zapier tools with their descriptions and
        required parameters.
        
        Universal design: When tools are added/removed in Zapier, the prompt
        automatically updates - NO code changes required.
        
        Returns:
            Formatted prompt string, or empty string if Zapier not available
        """
        if self._zapier_manager:
            return self._zapier_manager.get_tools_prompt()
        return ""
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get tool by name"""
        return self.tools.get(name)
    
    def get_available_tools(self, include_zapier: bool = False, include_mongodb: bool = False, include_redis: bool = False) -> List[str]:
        """
        Get list of available tool names.
        
        Args:
            include_zapier: Include Zapier tools in the list
            include_mongodb: Include MongoDB tool in the list
            include_redis: Include Redis tool in the list
            
        Returns:
            List of tool names
        """
        tools = list(self.tools.keys())
        
        if include_zapier and self._zapier_manager:
            tools.extend(self._zapier_manager.get_tool_names())
        
        if include_mongodb and self._mongodb_manager and self._mongodb_manager.is_connected:
            tools.append("mongodb")
        
        if include_redis and self._redis_manager and self._redis_manager.is_connected:
            tools.append("redis")
        
        if self._grievance_agent and self._grievance_enabled:
            tools.append("grievance")
        
        return tools
    
    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all available tools"""
        descriptions = {
            name: tool.description 
            for name, tool in self.tools.items()
        }
        
        # Add grievance description if available
        if self._grievance_agent and self._grievance_enabled:
            descriptions["grievance"] = "Extract structured grievance parameters from natural language complaints (for DM grievance tracking)"
        
        return descriptions
    
    async def execute_tool(
        self, 
        tool_name: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a tool by name (supports both built-in and Zapier tools)
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool-specific arguments
        """
        
        # Check if it's a Zapier tool
        if tool_name.startswith("zapier_") and self._zapier_manager:
            logger.info(f"🔧 Executing Zapier tool: {tool_name}")
            
            try:
                # Extract params from kwargs
                params = kwargs.get("params", {})
                
                if not params:
                    # Build params from remaining kwargs
                    params = {k: v for k, v in kwargs.items() 
                              if k not in ["query", "user_id"]}
                
                # SMART PARAM HANDLING FOR ZAPIER:
                # If LLM provided JSON in 'query' (e.g., {"to": "...", "subject": "...", "body": "..."})
                # Convert it to 'instructions' format that Zapier expects
                query_str = kwargs.get("query", "")
                
                if query_str and not params.get("instructions"):
                    # Check if query is JSON (LLM-generated structured params)
                    import json
                    try:
                        parsed_json = json.loads(query_str)
                        if isinstance(parsed_json, dict):
                            # LLM generated structured params - convert to instructions
                            # Build natural language instructions from the params
                            instructions_parts = []
                            for key, value in parsed_json.items():
                                instructions_parts.append(f"{key}: {value}")
                            params["instructions"] = "\n".join(instructions_parts)
                            logger.info(f"📝 Converted LLM JSON params to Zapier instructions")
                    except (json.JSONDecodeError, TypeError):
                        # Not JSON - use as-is for instructions
                        params["instructions"] = query_str
                        logger.info(f"📝 Using query string as Zapier instructions")
                
                result = await self._zapier_manager.execute(
                    query=kwargs.get("query", ""),
                    user_id=kwargs.get("user_id"),
                    tool_name=tool_name,
                    params=params
                )
                
                if result.get("success"):
                    logger.info(f"  Zapier tool '{tool_name}' executed successfully")
                else:
                    logger.warning(f"⚠️ Zapier tool '{tool_name}' failed: {result.get('error')}")
                
                return result
                
            except Exception as e:
                logger.error(f"❌ Zapier tool '{tool_name}' error: {str(e)}")
                return {
                    "success": False,
                    "error": f"Zapier tool execution failed: {str(e)}",
                    "tool_name": tool_name,
                    "provider": "zapier_mcp"
                }
        
        # Check if it's a MongoDB tool
        if tool_name == "mongodb" and self._mongodb_manager and self._query_agent:
            logger.info(f"🔧 Executing MongoDB tool via QueryAgent")
            
            try:
                query = kwargs.get("query", "")
                
                if not query:
                    return {
                        "success": False,
                        "error": "No query provided for MongoDB operation",
                        "tool_name": tool_name,
                        "provider": "mongodb_mcp"
                    }
                
                # Get tools prompt from MongoDB MCP client
                tools_prompt = self._mongodb_manager.get_tools_prompt()
                
                # Execute via QueryAgent (NL → structured query → execution)
                result = await self._query_agent.execute(
                    tools_prompt=tools_prompt,
                    instruction=query,
                    mcp_client=self._mongodb_manager
                )
                
                # Convert QueryResult to dict format
                if result.needs_clarification:
                    logger.info(f"❓ MongoDB needs clarification: {result.clarification_message}")
                    return {
                        "success": False,
                        "needs_clarification": True,
                        "clarification_message": result.clarification_message,
                        "missing_fields": result.missing_fields or [],
                        "tool_name": tool_name,
                        "provider": "mongodb_mcp"
                    }
                elif result.success:
                    logger.info(f"  MongoDB tool executed successfully")
                    return {
                        "success": True,
                        "result": result.result,
                        "executed_tool": result.tool_name,
                        "params": result.params,
                        "tool_name": tool_name,
                        "provider": "mongodb_mcp"
                    }
                else:
                    logger.warning(f"⚠️ MongoDB tool failed: {result.error}")
                    return {
                        "success": False,
                        "error": result.error,
                        "tool_name": tool_name,
                        "provider": "mongodb_mcp"
                    }
                
            except Exception as e:
                logger.error(f"❌ MongoDB tool error: {str(e)}")
                return {
                    "success": False,
                    "error": f"MongoDB tool execution failed: {str(e)}",
                    "tool_name": tool_name,
                    "provider": "mongodb_mcp"
                }
        
        # Check if it's a Redis tool
        if tool_name == "redis" and self._redis_manager and self._query_agent:
            logger.info(f"🔧 Executing Redis tool via QueryAgent")
            
            try:
                query = kwargs.get("query", "")
                
                if not query:
                    return {
                        "success": False,
                        "error": "No query provided for Redis operation",
                        "tool_name": tool_name,
                        "provider": "redis_mcp"
                    }
                
                # Get tools prompt from Redis MCP client
                tools_prompt = self._redis_manager.get_tools_prompt()
                
                # Execute via QueryAgent (NL → structured query → execution)
                result = await self._query_agent.execute(
                    tools_prompt=tools_prompt,
                    instruction=query,
                    mcp_client=self._redis_manager
                )
                
                # Convert QueryResult to dict format
                if result.needs_clarification:
                    logger.info(f"❓ Redis needs clarification: {result.clarification_message}")
                    return {
                        "success": False,
                        "needs_clarification": True,
                        "clarification_message": result.clarification_message,
                        "missing_fields": result.missing_fields or [],
                        "tool_name": tool_name,
                        "provider": "redis_mcp"
                    }
                elif result.success:
                    logger.info(f"  Redis tool executed successfully")
                    return {
                        "success": True,
                        "result": result.result,
                        "executed_tool": result.tool_name,
                        "params": result.params,
                        "tool_name": tool_name,
                        "provider": "redis_mcp"
                    }
                else:
                    logger.warning(f"⚠️ Redis tool failed: {result.error}")
                    return {
                        "success": False,
                        "error": result.error,
                        "tool_name": tool_name,
                        "provider": "redis_mcp"
                    }
                
            except Exception as e:
                logger.error(f"❌ Redis tool error: {str(e)}")
                return {
                    "success": False,
                    "error": f"Redis tool execution failed: {str(e)}",
                    "tool_name": tool_name,
                    "provider": "redis_mcp"
                }
        
        # Check if it's a Grievance tool
        if tool_name == "grievance" and self._grievance_agent:
            logger.info(f"🔧 Executing Grievance tool via GrievanceAgent")
            
            try:
                query = kwargs.get("query", "")
                
                if not query:
                    return {
                        "success": False,
                        "error": "No query provided for grievance extraction",
                        "tool_name": tool_name,
                        "provider": "grievance_agent"
                    }
                
                # Execute via GrievanceAgent (NL → structured params)
                result = await self._grievance_agent.execute(instruction=query)
                
                # Convert GrievanceResult to dict format
                if result.needs_clarification:
                    logger.info(f"❓ Grievance needs clarification: {result.clarification_message}")
                    return {
                        "success": False,
                        "needs_clarification": True,
                        "clarification_message": result.clarification_message,
                        "missing_fields": result.missing_fields or [],
                        "tool_name": tool_name,
                        "provider": "grievance_agent"
                    }
                elif result.success:
                    logger.info(f"✅ Grievance extracted successfully")
                    return {
                        "success": True,
                        "result": result.params.to_dict() if result.params else {},
                        "params": result.params.to_dict() if result.params else {},
                        "tool_name": tool_name,
                        "provider": "grievance_agent"
                    }
                else:
                    logger.warning(f"⚠️ Grievance extraction failed: {result.error}")
                    return {
                        "success": False,
                        "error": result.error,
                        "tool_name": tool_name,
                        "provider": "grievance_agent"
                    }
                
            except Exception as e:
                logger.error(f"❌ Grievance tool error: {str(e)}")
                return {
                    "success": False,
                    "error": f"Grievance tool execution failed: {str(e)}",
                    "tool_name": tool_name,
                    "provider": "grievance_agent"
                }
        
        # Standard tool execution
        tool = self.get_tool(tool_name)
        if not tool:
            logger.error(f"❌ Tool '{tool_name}' not available")
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not available",
                "available_tools": self.get_available_tools()
            }
        
        try:
            logger.info(f"🔧 Executing tool: {tool_name}")
            logger.debug(f"   Args: {kwargs}")
            
            result = await tool.execute(**kwargs)
            result["tool_name"] = tool_name
            
            if result.get("success"):
                logger.info(f"  Tool '{tool_name}' executed successfully")
            else:
                logger.warning(f"⚠️ Tool '{tool_name}' execution failed: {result.get('error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Tool '{tool_name}' execution error: {str(e)}")
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}",
                "tool_name": tool_name
            }
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all tools including Zapier"""
        
        stats = {}
        for name, tool in self.tools.items():
            tool_stats = {
                "usage_count": tool.usage_count,
            }
            
            # Add provider-specific stats for web search
            if hasattr(tool, 'get_stats'):
                tool_stats.update(tool.get_stats())
            
            stats[name] = tool_stats
        
        # Add Zapier stats if available
        if self._zapier_manager:
            stats["zapier_mcp"] = self._zapier_manager.get_stats()
        
        return stats
    
    async def cleanup(self):
        """Cleanup tool resources including Zapier and MongoDB"""
        
        logger.info("🧹 Cleaning up tools...")
        
        # Cleanup standard tools
        for name, tool in self.tools.items():
            if hasattr(tool, 'close'):
                try:
                    await tool.close()
                    logger.debug(f"     Closed {name}")
                except Exception as e:
                    logger.warning(f"   ⚠️ Error closing {name}: {str(e)}")
        
        # Cleanup Zapier
        if self._zapier_manager:
            try:
                await self._zapier_manager.close()
                logger.debug("     Closed Zapier MCP")
            except Exception as e:
                logger.warning(f"   ⚠️ Error closing Zapier MCP: {str(e)}")
        
        # Cleanup MongoDB
        if self._mongodb_manager:
            try:
                await self._mongodb_manager.disconnect()
                logger.debug("     Closed MongoDB MCP")
            except Exception as e:
                logger.warning(f"   ⚠️ Error closing MongoDB MCP: {str(e)}")
        
        # Cleanup QueryAgent
        if self._query_agent:
            try:
                await self._query_agent.close()
                logger.debug("     Closed QueryAgent")
            except Exception as e:
                logger.warning(f"   ⚠️ Error closing QueryAgent: {str(e)}")
        
        # Cleanup GrievanceAgent
        if self._grievance_agent:
            try:
                await self._grievance_agent.close()
                logger.debug("     Closed GrievanceAgent")
            except Exception as e:
                logger.warning(f"   ⚠️ Error closing GrievanceAgent: {str(e)}")
        
        logger.info("  Tool cleanup complete")