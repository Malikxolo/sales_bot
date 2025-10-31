"""
Configuration system for Brain-Heart Deep Research System
FIXED VERSION - Proper web model handling
WITH REDIS CACHE MANAGER
WITH SCRAPING GUIDANCE CONFIGURATION
"""

import os
import json
import hashlib
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from typing import Callable, Coroutine, Tuple, Any
import redis.asyncio as redis

logger = logging.getLogger(__name__)

# ==================== SCRAPING GUIDANCE CONFIGURATION ====================

# Map scraping levels to actual counts
SCRAPING_LEVELS = {
    "low": 1,      # Scrape top 1 result
    "medium": 3,   # Scrape top 3 results
    "high": 5      # Scrape top 5 results
}

# Threshold for scraping that requires user confirmation (pages)
SCRAPING_CONFIRMATION_THRESHOLD = 3  # Ask confirmation for medium (3) and high (5) scraping

# Feature flags
ENABLE_SCRAPING_GUIDANCE = True
ENABLE_SCRAPING_CONFIRMATION = True  # Enable/disable confirmation flow
ENABLE_SUPERSEDE_ON_NEW_QUERY = True  # Auto-cancel pending confirmations when user sends new non-confirmation query
ENABLE_LLM_CONFIRMATION_DETECTION = True  # Use LLM for semantic confirmation intent detection
ENABLE_CONFIRMATION_REGEX_FALLBACK = True  # Use regex fallback when LLM confidence is low

# Confirmation settings
SCRAPING_CONFIRMATION_TTL = 300  # 5 minutes for pending confirmations
ESTIMATED_TIME_PER_PAGE = 5  # Seconds per page scraped (approximate)
LLM_CONFIRMATION_CONFIDENCE_THRESHOLD = 70  # Minimum confidence to trust LLM intent detection

@dataclass
class AddBackgroundTask:
    """Dataclass for adding message task"""
    func: Callable[..., Coroutine[Any, Any, Any]]
    params: Tuple[Any, ...]

@dataclass
class LLMConfig:
    """Configuration for LLM models"""
    provider: str
    model: str
    api_key: str
    max_tokens: int = 4000
    timeout: int = 30
    base_url: Optional[str] = None


class RedisCacheManager:
    """Redis-based cache manager for queries and tool results"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.enabled = False
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection from environment variables"""
        try:
            redis_host = os.getenv('REDIS_HOST')
            redis_port = int(os.getenv('REDIS_PORT', 6379))
            redis_password = os.getenv('REDIS_PASSWORD')
            redis_username = os.getenv('REDIS_USERNAME', 'default')
            
            if redis_host:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    username=redis_username,
                    password=redis_password,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                self.enabled = True
                logger.info(f"✅ Redis cache manager initialized: {redis_host}:{redis_port}")
            else:
                logger.warning("⚠️ Redis not configured - caching disabled")
        except Exception as e:
            logger.error(f"❌ Redis initialization failed: {e}")
            self.enabled = False
    
    def _generate_cache_key(self, prefix: str, data: str, user_id: str = None) -> str:
        """Generate a unique cache key using hash"""
        key_data = f"{data}_{user_id or 'anonymous'}"
        hash_key = hashlib.md5(key_data.encode()).hexdigest()
        return f"{prefix}:{hash_key}"
    
    async def get_cached_query(self, query: str, user_id: str = None) -> Optional[Dict]:
        """Get cached analysis for a query"""
        if not self.enabled or not self.redis_client:
            return None
        
        try:
            cache_key = self._generate_cache_key("query_analysis", query, user_id)
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                logger.info(f"🎯 Cache HIT for query: {query[:50]}...")
                return json.loads(cached_data)
            else:
                logger.info(f"❌ Cache MISS for query: {query[:50]}...")
                return None
        except Exception as e:
            logger.error(f"❌ Redis get error: {e}")
            return None
    
    async def cache_query(self, query: str, analysis: Dict, user_id: str = None, ttl: int = 3600):
        """Cache query analysis with TTL (default 1 hour)"""
        if not self.enabled or not self.redis_client:
            return
        
        try:
            cache_key = self._generate_cache_key("query_analysis", query, user_id)
            await self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(analysis, ensure_ascii=False)
            )
            logger.info(f"💾 Cached query analysis: {query[:50]}... (TTL: {ttl}s)")
        except Exception as e:
            logger.error(f"❌ Redis set error: {e}")
    
    async def get_cached_tool_data(self, tool_results: Dict, user_id: str = None) -> Optional[str]:
        """Get cached formatted tool data"""
        if not self.enabled or not self.redis_client:
            return None
        
        try:
            # Create a hash from tool results structure
            tool_key = json.dumps(tool_results, sort_keys=True)
            cache_key = self._generate_cache_key("tool_data", tool_key, user_id)
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                logger.info(f"🎯 Cache HIT for formatted tool data")
                return cached_data
            else:
                logger.info(f"❌ Cache MISS for formatted tool data")
                return None
        except Exception as e:
            logger.error(f"❌ Redis get error for tool data: {e}")
            return None
    
    async def cache_tool_data(self, tool_results: Dict, formatted_data: str, user_id: str = None, ttl: int = 7200):
        """Cache formatted tool data with TTL (default 2 hours)"""
        if not self.enabled or not self.redis_client:
            return
        
        try:
            tool_key = json.dumps(tool_results, sort_keys=True)
            cache_key = self._generate_cache_key("tool_data", tool_key, user_id)
            await self.redis_client.setex(
                cache_key,
                ttl,
                formatted_data
            )
            logger.info(f"💾 Cached formatted tool data (TTL: {ttl}s)")
        except Exception as e:
            logger.error(f"❌ Redis set error for tool data: {e}")
    
    async def get_cached_tool_results(self, query: str, tools: List[str], user_id: str = None, scraping_guidance: Dict = None) -> Optional[Dict]:
        """Get cached tool execution results for a query (includes scraping guidance in key)"""
        if not self.enabled or not self.redis_client:
            return None
        
        try:
            # Create cache key from query + tools combination + scraping guidance
            tools_str = json.dumps(sorted(tools))
            scraping_str = json.dumps(scraping_guidance, sort_keys=True) if scraping_guidance else ""
            cache_data = f"{query}_{tools_str}_{scraping_str}"
            cache_key = self._generate_cache_key("tool_results", cache_data, user_id)
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                logger.info(f"🎯 Cache HIT for tool results: {query[:50]}...")
                return json.loads(cached_data)
            else:
                logger.info(f"❌ Cache MISS for tool results: {query[:50]}...")
                return None
        except Exception as e:
            logger.error(f"❌ Redis get error for tool results: {e}")
            return None
    
    async def cache_tool_results(self, query: str, tools: List[str], tool_results: Dict, user_id: str = None, scraping_guidance: Dict = None, ttl: int = 3600):
        """Cache tool execution results with TTL (default 1 hour) - includes scraping guidance in key"""
        if not self.enabled or not self.redis_client:
            return
        
        try:
            # Create cache key from query + tools combination + scraping guidance
            tools_str = json.dumps(sorted(tools))
            scraping_str = json.dumps(scraping_guidance, sort_keys=True) if scraping_guidance else ""
            cache_data = f"{query}_{tools_str}_{scraping_str}"
            cache_key = self._generate_cache_key("tool_results", cache_data, user_id)
            await self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(tool_results, ensure_ascii=False)
            )
            logger.info(f"💾 Cached tool results for: {query[:50]}... (TTL: {ttl}s)")
        except Exception as e:
            logger.error(f"❌ Redis set error for tool results: {e}")
    
    async def clear_user_cache(self, user_id: str):
        """Clear all cache for a specific user"""
        if not self.enabled or not self.redis_client:
            return
        
        try:
            # Scan for user-specific keys
            pattern = f"*{user_id}*"
            cursor = 0
            deleted_count = 0
            
            while True:
                cursor, keys = await self.redis_client.scan(cursor, match=pattern, count=100)
                if keys:
                    await self.redis_client.delete(*keys)
                    deleted_count += len(keys)
                if cursor == 0:
                    break
            
            logger.info(f"🗑️ Cleared {deleted_count} cache entries for user: {user_id}")
        except Exception as e:
            logger.error(f"❌ Redis clear error: {e}")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.enabled or not self.redis_client:
            return {"enabled": False}
        
        try:
            info = await self.redis_client.info()
            return {
                "enabled": True,
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "N/A"),
                "total_keys": await self.redis_client.dbsize()
            }
        except Exception as e:
            logger.error(f"❌ Redis stats error: {e}")
            return {"enabled": False, "error": str(e)}
    
    # ==================== CONFIRMATION FLOW METHODS ====================
    
    async def set_pending_confirmation(self, token: str, payload: Dict, user_id: str, ttl: int = None) -> bool:
        """
        Store a pending confirmation action in Redis
        
        Args:
            token: Unique confirmation token (UUID)
            payload: Dict containing query, analysis, tools, scraping_guidance, etc.
            user_id: User ID for security validation
            ttl: Time-to-live in seconds (default: SCRAPING_CONFIRMATION_TTL)
        
        Returns:
            True if stored successfully, False otherwise
        """
        if not self.enabled or not self.redis_client:
            logger.warning("⚠️ Redis not enabled, cannot store pending confirmation")
            return False
        
        try:
            ttl = ttl or SCRAPING_CONFIRMATION_TTL
            cache_key = f"pending_confirm:{token}"
            
            # Add metadata
            payload["user_id"] = user_id
            payload["token"] = token
            payload["created_at"] = datetime.now().isoformat()
            
            await self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(payload, ensure_ascii=False)
            )
            
            logger.info(f"💾 Stored pending confirmation: {token} for user {user_id} (TTL: {ttl}s)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store pending confirmation: {e}")
            return False
    
    async def get_pending_confirmation(self, token: str) -> Optional[Dict]:
        """
        Retrieve a pending confirmation by token
        
        Args:
            token: Confirmation token
        
        Returns:
            Payload dict if found, None otherwise
        """
        if not self.enabled or not self.redis_client:
            return None
        
        try:
            cache_key = f"pending_confirm:{token}"
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                logger.info(f"🎯 Retrieved pending confirmation: {token}")
                return json.loads(cached_data)
            else:
                logger.debug(f"❌ No pending confirmation found for token: {token}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error retrieving pending confirmation: {e}")
            return None
    
    async def get_pending_confirmation_for_user(self, user_id: str) -> Optional[Dict]:
        """
        Retrieve the most recent pending confirmation for a user
        
        Args:
            user_id: User ID
        
        Returns:
            Payload dict if found, None otherwise
        """
        if not self.enabled or not self.redis_client:
            return None
        
        try:
            # Scan for user's pending confirmations
            pattern = f"pending_confirm:*"
            cursor = 0
            
            while True:
                cursor, keys = await self.redis_client.scan(cursor, match=pattern, count=100)
                
                for key in keys:
                    cached_data = await self.redis_client.get(key)
                    if cached_data:
                        payload = json.loads(cached_data)
                        if payload.get("user_id") == user_id:
                            logger.info(f"🎯 Found pending confirmation for user: {user_id}")
                            return payload
                
                if cursor == 0:
                    break
            
            logger.debug(f"❌ No pending confirmation found for user: {user_id}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error searching pending confirmations: {e}")
            return None
    
    async def delete_pending_confirmation(self, token: str) -> bool:
        """
        Delete a pending confirmation
        
        Args:
            token: Confirmation token
        
        Returns:
            True if deleted, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            cache_key = f"pending_confirm:{token}"
            deleted = await self.redis_client.delete(cache_key)
            
            if deleted:
                logger.info(f"🗑️ Deleted pending confirmation: {token}")
                return True
            else:
                logger.debug(f"⚠️ No pending confirmation to delete: {token}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error deleting pending confirmation: {e}")
            return False
    
    async def cancel_all_pending_confirmations_for_user(self, user_id: str) -> int:
        """
        Cancel (delete) all pending confirmations for a specific user
        Used when user sends a new non-confirmation query to prevent accidental resumption
        
        Args:
            user_id: User ID
        
        Returns:
            Number of pending confirmations cancelled
        """
        if not self.enabled or not self.redis_client:
            return 0
        
        try:
            # Scan for user's pending confirmations
            pattern = f"pending_confirm:*"
            cursor = 0
            cancelled_count = 0
            
            while True:
                cursor, keys = await self.redis_client.scan(cursor, match=pattern, count=100)
                
                for key in keys:
                    cached_data = await self.redis_client.get(key)
                    if cached_data:
                        payload = json.loads(cached_data)
                        if payload.get("user_id") == user_id:
                            # Delete this pending confirmation
                            token = payload.get("token")
                            await self.redis_client.delete(key)
                            logger.info(f"🔻 Superseded pending confirmation {token} for user {user_id}")
                            cancelled_count += 1
                
                if cursor == 0:
                    break
            
            if cancelled_count > 0:
                logger.info(f"✅ Cancelled {cancelled_count} pending confirmation(s) for user {user_id}")
            
            return cancelled_count
            
        except Exception as e:
            logger.error(f"❌ Error cancelling pending confirmations: {e}")
            return 0

class Config:
    """Dynamic configuration manager - fixed version"""
    
    def __init__(self):
        self.available_providers: Dict[str, str] = {}
        self.available_models: Dict[str, List[str]] = {}
        self.available_web_models: List[str] = [
            "perplexity/sonar",
            "perplexity/sonar-deep-research",
            "perplexity/sonar-pro", 
            "perplexity/sonar-reasoning-pro"
        ]
        self.load_configuration()
    
    def load_configuration(self):
        """Load configuration and detect available providers"""
        self._detect_providers()
        self._load_available_models()
    
    def _detect_providers(self):
        """Auto-detect available LLM providers from environment"""
        
        provider_mappings = {
            'OPENAI_API_KEY': 'openai',
            'ANTHROPIC_API_KEY': 'anthropic', 
            'OPENROUTER_API_KEY': 'openrouter',
            'GROQ_API_KEY': 'groq',
            'DEEPSEEK_API_KEY': 'deepseek'
        }
        
        for env_key, provider in provider_mappings.items():
            api_key = os.getenv(env_key)
            if api_key:
                self.available_providers[provider] = api_key
                print(f"✅ Detected {provider.upper()} API key")
        
        if not self.available_providers:
            raise Exception("No LLM provider API keys found. Please set at least one API key in .env file.")
    
    def _load_available_models(self):
        """Load available models for each provider"""
        
        model_catalog = {
            'openai': ['gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo'],
            # 'anthropic': ['claude-3-5-sonnet-20241022', 'claude-3-opus-20240229', 'claude-3-haiku-20240307'],
            'openrouter': ['qwen/qwen3-next-80b-a3b-thinking','anthropic/claude-3.5-sonnet', 'openai/gpt-4o', 'meta-llama/llama-3.3-70b-instruct','qwen/qwen3-next-80b-a3b-instruct','deepseek/deepseek-chat-v3.1'],
            'groq': ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'qwen/qwen3-32b','deepseek-r1-distill-llama-70b','openai/gpt-oss-20b'],
            # 'deepseek': ['deepseek-chat', 'deepseek-reasoner']
        }
        
        for provider in self.available_providers:
            if provider in model_catalog:
                self.available_models[provider] = model_catalog[provider]
            else:
                self.available_models[provider] = ['default']
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return list(self.available_providers.keys())
    
    def get_available_models(self, provider: str) -> List[str]:
        """Get available models for provider"""
        return self.available_models.get(provider, [])

    def get_available_web_models(self) -> List[str]:
        """Get available web search models"""
        return self.available_web_models.copy()
    
    def create_llm_config(self, provider: str, model: str, 
                        max_tokens: int = 4000) -> LLMConfig:
        """Create LLM configuration for any provider/model combination"""
        
        if provider not in self.available_providers:
            raise Exception(f"Provider {provider} not available")
        
        api_key = self.available_providers[provider]
        
        # Provider-specific configurations
        base_url = None
        if provider == 'openrouter':
            base_url = "https://openrouter.ai/api/v1"
        elif provider == 'groq':
            base_url = "https://api.groq.com/openai/v1"
        elif provider == 'deepseek':
            base_url = "https://api.deepseek.com/v1"
        
        return LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            base_url=base_url
        )
    
    def get_tool_configs(self, web_model: str = None, use_premium_search: bool = False) -> Dict[str, Any]:

        """Get tool configurations with selected web model"""
        # Default web model if none provided
        if not web_model:
            web_model = self.available_web_models[0]
        
        # Check for search API keys
        scrapingdog_key = os.getenv('SCRAPINGDOG_API_KEY')
        valueserp_key = os.getenv('VALUESERP_API_KEY')
        openrouter_key = os.getenv('OPENROUTER_API_KEY')
        jina_key = os.getenv('JINA_API_KEY')
        print(f"🔍 DEBUG: JINA_API_KEY from env: {jina_key[:20] if jina_key else '❌ NOT FOUND IN ENV!'}")

        
        # Web search is enabled if we have OpenRouter (for Perplexity) or search API keys
        web_search_enabled = bool(openrouter_key or scrapingdog_key or valueserp_key)
        
        tools = {
            'calculator': {
                'enabled': True,
                'description': 'Mathematical calculations and statistical operations'
            },
            'web_search': {
                'enabled': web_search_enabled,
                'provider': 'perplexity' if use_premium_search else ('scrapingdog' if scrapingdog_key else 'valueserp'),
                'web_model': web_model if use_premium_search else None,
                'primary_key': scrapingdog_key or valueserp_key,  # For non-Perplexity providers
                'jina_key': jina_key,
                'description': f'Internet search using {"Perplexity " + web_model if use_premium_search else "ScrapingDog/ValueSerp"}',
                'model_info': {
                    'selected_model': web_model,
                    'available_models': self.available_web_models
                }
            },
            'social_search': {
                'enabled': bool(os.getenv('REDDIT_CLIENT_ID') or os.getenv('YOUTUBE_API_KEY')),
                'reddit_enabled': bool(os.getenv('REDDIT_CLIENT_ID')),
                'youtube_enabled': bool(os.getenv('YOUTUBE_API_KEY'))
            },
            'rag': {
                'enabled': True,
                'description': 'Knowledge base retrieval and analysis'
            },
            'reasoning': {
                'enabled': True,
                'description': 'Advanced reasoning and analysis'
            }
        }
        
        print(f"🚀 Premium search enabled: {use_premium_search}")
        print(f"🔍 Web search enabled: {web_search_enabled}")
        
        return tools
    
    def validate_web_model(self, model: str) -> bool:
        """Validate if web model is available"""
        return model in self.available_web_models
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return {
            'available_providers': list(self.available_providers.keys()),
            'available_models': self.available_models,
            'available_web_models': self.available_web_models,
            'tool_configs': self.get_tool_configs()
        }