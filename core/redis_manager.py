"""
Redis Cache Manager for Brain-Heart Deep Research System
Handles query caching and tool results caching
"""

import os
import json
import hashlib
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import redis.asyncio as redis

logger = logging.getLogger(__name__)


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
    
    async def get_cached_tool_results(self, query: str, tools: List[str], user_id: str = None) -> Optional[Dict]:
        """Get cached tool execution results for a query"""
        if not self.enabled or not self.redis_client:
            return None
        
        try:
            # Create cache key from query + tools combination
            tools_str = json.dumps(sorted(tools))
            cache_data = f"{query}_{tools_str}"
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
    
    async def cache_tool_results(self, query: str, tools: List[str], tool_results: Dict, user_id: str = None, ttl: int = 3600):
        """Cache tool execution results with TTL (default 1 hour)"""
        if not self.enabled or not self.redis_client:
            return
        
        try:
            # Create cache key from query + tools combination
            tools_str = json.dumps(sorted(tools))
            cache_data = f"{query}_{tools_str}"
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
    
    # === Business Context Cache ===
    
    async def get_business_context(self, business_id: str) -> Optional[dict]:
        """Get cached business context"""
        if not self.enabled or not self.redis_client:
            return None
        try:
            cache_key = f"business_context:{business_id}"
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                logger.info(f"🎯 Cache HIT for business context: {business_id}")
                return json.loads(cached_data)
            logger.info(f"❌ Cache MISS for business context: {business_id}")
            return None
        except Exception as e:
            logger.error(f"❌ Redis get error for business context: {e}")
            return None
    
    async def cache_business_context(self, business_id: str, context: dict, ttl: int = 86400):
        """Cache business context for 24 hours"""
        if not self.enabled or not self.redis_client:
            return
        try:
            cache_key = f"business_context:{business_id}"
            await self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(context, ensure_ascii=False)
            )
            logger.info(f"💾 Cached business context: {business_id} (TTL: {ttl}s)")
        except Exception as e:
            logger.error(f"❌ Redis set error for business context: {e}")
    
    # === Conversation Summary Cache ===
    
    async def get_conversation_summary(self, user_id: str) -> Optional[dict]:
        """Get cached conversation summary"""
        if not self.enabled or not self.redis_client:
            return None
        try:
            cache_key = f"conversation_summary:{user_id}"
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                logger.info(f"🎯 Cache HIT for conversation summary: {user_id}")
                return json.loads(cached_data)
            logger.info(f"❌ Cache MISS for conversation summary: {user_id}")
            return None
        except Exception as e:
            logger.error(f"❌ Redis get error for conversation summary: {e}")
            return None
    
    async def cache_conversation_summary(self, user_id: str, summary: dict, ttl: int = 86400):
        """Cache conversation summary for 24 hours"""
        if not self.enabled or not self.redis_client:
            return
        try:
            cache_key = f"conversation_summary:{user_id}"
            await self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(summary, ensure_ascii=False)
            )
            logger.info(f"💾 Cached conversation summary: {user_id} (TTL: {ttl}s)")
        except Exception as e:
            logger.error(f"❌ Redis set error for conversation summary: {e}")
