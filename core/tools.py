"""
Tool system for Brain-Heart Deep Research System
FIXED VERSION - Proper model selection flow
"""

import asyncio
import aiohttp
import json
import re
import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod
from .exceptions import ToolExecutionError
from .quota_manager import QuotaManager
from .llm_client import LLMClient
from .weaviate_rag import get_weaviate_rag_client
from .web_search_agent import search_perplexity, search_llmlayer
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

class PaymentTool(BaseTool):
    """Generate WhatsApp native payment parameters for order processing"""

    def __init__(self, rag_tool: "RAGTool" = None):
        super().__init__(
            "payment",
            "Generate payment order for WhatsApp native payment"
        )
        self.rag_tool = rag_tool
        self.configuration_name = os.getenv("PAYMENT_GATEWAY_CONFIG_NAME", "FoodNests")
        self.payment_gateway_type = os.getenv("PAYMENT_GATEWAY_TYPE", "razorpay")
        self.currency = os.getenv("PAYMENT_CURRENCY", "INR")
        self.retailer_id = os.getenv("PAYMENT_RETAILER_ID", "FOODNEST-MAIN")
        logger.info("PaymentTool initialized")

    async def execute(self, query: str, user_id: str = None, **kwargs) -> Dict[str, Any]:
        """
        Generate payment parameters for WhatsApp order.
        
        Looks up actual prices from RAG, generates unique reference_id,
        constructs Meta Graph API compliant payment params.
        """
        self._record_usage()
        import uuid as uuid_mod

        try:
            # Step 1: Parse items from the query
            items_to_price = self._parse_items(query)
            if not items_to_price:
                return {
                    "success": False,
                    "error": "Could not determine items to purchase. Please specify what you'd like to buy.",
                    "needs_clarification": True
                }

            # Step 2: Look up prices from RAG (knowledge base)
            priced_items = []
            for item in items_to_price:
                if self.rag_tool:
                    price_query = f"price of {item['name']}"
                    rag_result = await self.rag_tool.execute(
                        query=price_query,
                        user_id=user_id,
                        **{k: v for k, v in kwargs.items() if k in ['businessId', 'email', 'collection_ids']}
                    )
                    if rag_result.get("success"):
                        price_paise = self._extract_price_from_rag(
                            rag_result.get("retrieved", ""),
                            item["name"]
                        )
                        if price_paise:
                            priced_items.append({
                                "name": item["name"],
                                "quantity": item["quantity"],
                                "unit_price": price_paise
                            })
                        else:
                            return {
                                "success": False,
                                "error": f"Could not find price for '{item['name']}' in our catalog.",
                                "needs_clarification": True
                            }
                    else:
                        return {
                            "success": False,
                            "error": f"Could not look up '{item['name']}' in catalog.",
                            "needs_clarification": True
                        }
                else:
                    return {
                        "success": False,
                        "error": "Product catalog not available for price lookup.",
                    }

            # Step 3: Generate unique reference ID
            reference_id = f"ORD-{uuid_mod.uuid4().hex[:12].upper()}"

            # Step 4: Build payment params
            params = {
                "reference_id": reference_id,
                "retailer_id": self.retailer_id,
                "items": priced_items,
                "payment_gateway": {
                    "type": self.payment_gateway_type,
                    "configuration_name": self.configuration_name
                },
                "currency": self.currency
            }

            # Step 5: Generate human-readable summary
            total_paise = sum(i["unit_price"] * i["quantity"] for i in priced_items)
            total_inr = total_paise / 100
            item_lines = [f"  {i['name']} x{i['quantity']} = ₹{(i['unit_price'] * i['quantity']) / 100:.0f}" for i in priced_items]
            order_summary = f"Order {reference_id}:\\n" + "\\n".join(item_lines) + f"\\n  Total: ₹{total_inr:.0f}"

            logger.info(f"✅ Payment order generated: {reference_id}, total=₹{total_inr:.0f}, items={len(priced_items)}")

            return {
                "success": True,
                "params": params,
                "order_summary": order_summary,
                "total_amount_inr": total_inr
            }

        except Exception as e:
            logger.error(f"❌ Payment generation failed: {e}")
            return {
                "success": False,
                "error": f"Payment generation failed: {str(e)}"
            }

    def _parse_items(self, query: str) -> List[Dict]:
        """Parse item names and quantities from query string"""
        items = []
        parts = [p.strip() for p in query.split(",") if p.strip()]
        for part in parts:
            match = re.match(r'(\d+)\s*[xX×]\s*(.+)', part)
            if match:
                items.append({"name": match.group(2).strip(), "quantity": int(match.group(1))})
            else:
                items.append({"name": part.strip(), "quantity": 1})
        return items

    def _extract_price_from_rag(self, rag_text: str, item_name: str) -> Optional[int]:
        """Extract price in paise from RAG text for a given item."""
        text_lower = rag_text.lower()
        item_lower = item_name.lower()

        item_pos = text_lower.find(item_lower)
        if item_pos == -1:
            for word in item_lower.split():
                if len(word) > 3:
                    pos = text_lower.find(word)
                    if pos != -1:
                        item_pos = pos
                        break

        if item_pos == -1:
            search_text = rag_text
        else:
            start = max(0, item_pos - 200)
            end = min(len(rag_text), item_pos + 300)
            search_text = rag_text[start:end]

        patterns = [
            r'₹\s*([\d,]+(?:\.\d{1,2})?)',
            r'[Rr]s\.?\s*([\d,]+(?:\.\d{1,2})?)',
            r'INR\s*([\d,]+(?:\.\d{1,2})?)',
            r'([\d,]+(?:\.\d{1,2})?)\s*(?:INR|inr)',
            r'[Pp]rice[:\s]+₹?\s*([\d,]+(?:\.\d{1,2})?)',
            r'[Cc]ost[:\s]+₹?\s*([\d,]+(?:\.\d{1,2})?)',
            r'[Mm][Rr][Pp][:\s]+₹?\s*([\d,]+(?:\.\d{1,2})?)',
        ]

        for pattern in patterns:
            match = re.search(pattern, search_text)
            if match:
                price_str = match.group(1).replace(",", "")
                try:
                    price_float = float(price_str)
                    price_paise = int(price_float * 100)
                    if 100 <= price_paise <= 10000000:  # ₹1 to ₹1,00,000
                        return price_paise
                except ValueError:
                    continue
        return None

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
    """RAG tool — retrieves from Weaviate RAG backend (FN-Weaviate-DB)"""

    def __init__(self, llm_client: LLMClient = None):
        super().__init__("rag", "Retrieve information from uploaded knowledge base")
        self.llm_client = llm_client
        self._weaviate = get_weaviate_rag_client()
        logger.info("RAGTool initialized (Weaviate backend)")

    async def execute(self, query: str, user_id: str = None, **kwargs) -> Dict[str, Any]:
        """Execute RAG query against the Weaviate RAG API."""
        self._record_usage()
        logger.info(f"RAG query started: user_id={user_id}, query='{query[:50]}...'")

        try:
            # Run synchronous HTTP call in thread pool so we don't block the event loop
            import asyncio
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._weaviate.query(
                    query,
                    top_k=5,
                    use_hybrid=False,
                    similarity_threshold=0.6,
                )
            )

            if result["success"]:
                chunks_count = len(result["results"])
                distances = result.get("distances", [])

                logger.info(f"✅ RAG query SUCCESS for user {user_id}")
                logger.info(f"   Retrieved chunks: {chunks_count}")
                logger.info(f"   Query: '{query[:50]}...'")

                if distances:
                    avg_distance = sum(distances) / len(distances)
                    min_distance = min(distances)
                    max_distance = max(distances)
                    logger.info(
                        f"   Distance metrics - Min: {min_distance:.4f}, "
                        f"Max: {max_distance:.4f}, Avg: {avg_distance:.4f}"
                    )
                    if avg_distance < 0.3:
                        logger.info("   Quality: HIGH relevance")
                    elif avg_distance < 0.6:
                        logger.info("   Quality: MEDIUM relevance")
                    else:
                        logger.warning("   Quality: LOW relevance")
                else:
                    avg_distance = None
                    logger.warning("   No distance information available")

                documents = [
                    r["document"]
                    for r in result["results"]
                    if isinstance(r, dict) and r.get("document")
                ]

                if documents:
                    first_chunk = documents[0][:200] + ("..." if len(documents[0]) > 200 else "")
                    logger.info(f"   First chunk preview: '{first_chunk}'")

                return {
                    "success":      True,
                    "retrieved":    "\n\n".join(documents),
                    "chunks":       result["results"],
                    "query":        query,
                    "chunks_count": chunks_count,
                    "collection":   "weaviate",
                    "distances":    distances,
                    "avg_distance": avg_distance,
                }
            else:
                logger.error(f"❌ RAG query FAILED: {result.get('error')}")
                return {
                    "success": False,
                    "error":   result.get("error", "Unknown error"),
                    "query":   query,
                }

        except Exception as e:
            import traceback
            logger.error(f"❌ RAG query EXCEPTION: {e}\n{traceback.format_exc()}")
            return {
                "success": False,
                "error":   f"RAG query failed: {str(e)}",
                "query":   query,
            }

class ToolManager:
    """
    Manages all available tools with multi-provider support
      Updated: Passes all SERP provider keys to WebSearchTool
    """
    
    def __init__(
        self, 
        config, 
        llm_client: LLMClient, 
        web_model: str = None, 
        use_premium_search: bool = False
    ):
        self.config = config
        self.llm_client = llm_client
        self.web_model = web_model
        self.use_premium_search = use_premium_search
        self.tools: Dict[str, BaseTool] = {}
        
        self._initialize_tools()
        self._zapier_manager = None  # Initialized async via initialize_zapier() in lifespan
        
        logger.info(f"ToolManager initialized with web model: {web_model}")
    
    def _initialize_tools(self):
        """Initialize all configured tools"""
        
        tool_configs = self.config.get_tool_configs(self.web_model, self.use_premium_search)
        
        logger.debug(f"Tool configs: {tool_configs}")
        
        # Web Search (if configured)
        self._initialize_web_search(tool_configs)
        
        # RAG Tool (always available)
        self.tools["rag"] = RAGTool(self.llm_client)
        logger.info("  RAG tool initialized")
        
        # Payment Tool (uses RAG for price lookups)
        self.tools["payment"] = PaymentTool(rag_tool=self.tools.get("rag"))
        logger.info("  Payment tool initialized")
    
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
    
    async def initialize_zapier(self):
        """
        Initialize Zapier MCP connection.
        Must be called from async context (lifespan in chat.py) after ToolManager is created.
        """
        from .zapier_mcp import ZapierMCPManager

        token = os.getenv("ZAPIER_MCP_TOKEN")
        if not token:
            logger.info("⚠️ ZAPIER_MCP_TOKEN not set — Zapier MCP disabled")
            return

        try:
            self._zapier_manager = ZapierMCPManager(token)
            await self._zapier_manager.initialize()

            tool_count = len(self._zapier_manager.get_tool_names())
            if tool_count > 0:
                logger.info(f"✅ Zapier MCP ready with {tool_count} tools")
            else:
                logger.warning("⚠️ Zapier MCP connected but no tools discovered — check your enabled Zaps in Zapier")
        except Exception as e:
            logger.error(f"❌ Zapier MCP initialization failed: {e}")
            self._zapier_manager = None

    async def execute_zapier_tool(self, tool_name: str, arguments: dict) -> dict:
        """
        Execute a Zapier MCP tool by name with given arguments.

        Smart param handling:
        - If no 'instructions' key is present, build one from 'query'
        - If 'query' looks like JSON (LLM generated structured params),
          convert each key:value pair to a natural language line
        - If 'query' is plain text, use it directly as instructions
        - Strip the raw 'query' key before sending — Zapier doesn't understand it
        """
        if not self._zapier_manager:
            return {
                "success": False,
                "error": "Zapier MCP is not initialized (missing ZAPIER_MCP_TOKEN or init failed)"
            }

        # ── SMART PARAM CONVERSION ──────────────────────────────────────────
        if not arguments.get("instructions"):
            query_str = arguments.get("query", "")

            if query_str:
                try:
                    parsed = json.loads(query_str)
                    if isinstance(parsed, dict):
                        # LLM generated structured JSON — convert to natural language
                        instructions_parts = [f"{k}: {v}" for k, v in parsed.items()]
                        arguments["instructions"] = "\n".join(instructions_parts)
                        logger.info(f"📝 Converted LLM JSON params to Zapier instructions for '{tool_name}'")
                    else:
                        arguments["instructions"] = query_str
                except (ValueError, TypeError):
                    # Plain text — use as-is
                    arguments["instructions"] = query_str
                    logger.info(f"📝 Using query string as Zapier instructions for '{tool_name}'")

            # Remove the raw 'query' key — Zapier doesn't know what to do with it
            arguments.pop("query", None)
        # ────────────────────────────────────────────────────────────────────

        return await self._zapier_manager.call_tool(tool_name, arguments)

    def get_zapier_tool_names(self) -> list:
        """Get list of available Zapier tool names. Returns empty list if not initialized."""
        if not self._zapier_manager:
            return []
        return self._zapier_manager.get_tool_names()

    def get_zapier_tool_descriptions(self) -> dict:
        """Get {name: description} dict for all available Zapier tools."""
        if not self._zapier_manager:
            return {}
        return self._zapier_manager.get_tool_descriptions()

    def get_zapier_tool_required_params(self) -> dict:
        """
        Get {tool_name: [required_param, ...]} for all Zapier tools.
        Sourced from the inputSchema Zapier returns during tools/list.
        Used by SalesAgent to build better LLM prompts.
        """
        if not self._zapier_manager:
            return {}
        return self._zapier_manager.get_tool_required_params()

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get tool by name"""
        return self.tools.get(name)
    
    def get_available_tools(self) -> List[str]:
        """
        Get list of available tool names.
            
        Returns:
            List of tool names
        """
        tools = list(self.tools.keys())
        
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
        
        # Add grievance_status description if available
        if self._grievance_status_url:
            descriptions["grievance_status"] = "Fetch status of a grievance by its ID (track complaint progress)"
        
        return descriptions
    
    async def execute_tool(
        self, 
        tool_name: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a tool by name
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool-specific arguments
        """
        
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
        """Cleanup tool resources"""
        
        logger.info("🧹 Cleaning up tools...")

        # Cleanup Zapier MCP connection
        if self._zapier_manager:
            try:
                await self._zapier_manager.close()
                logger.info("✅ Zapier MCP connection closed")
            except Exception as e:
                logger.warning(f"⚠️ Error closing Zapier MCP: {e}")
        
        # Cleanup standard tools
        for name, tool in self.tools.items():
            if hasattr(tool, 'close'):
                try:
                    await tool.close()
                    logger.debug(f"     Closed {name}")
                except Exception as e:
                    logger.warning(f"   ⚠️ Error closing {name}: {str(e)}")
        
        logger.info("  Tool cleanup complete")