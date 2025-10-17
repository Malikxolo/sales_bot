"""
Quota Manager - Tracks API usage and enforces limits
Automatically switches to next cheapest provider when quota is exhausted
"""

import logging
import time
import json
import os
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class QuotaManager:
    """
    Tracks API quotas and costs across providers
    Enforces free tier limits and switches providers intelligently
    """
    
    # Provider configurations with limits and costs
    PROVIDER_CONFIG = {
        "google_cse": {
            "free_daily": 100,
            "free_monthly": 3000,
            "max_daily": 10000,
            "cost_per_1k": 5.00,
            "reset_period": "daily",
            "priority": 1
        },
        "brave": {
            "free_daily": None,
            "free_monthly": 2000,
            "max_daily": None,
            "cost_per_1k": 3.00,
            "reset_period": "monthly",
            "priority": 2
        },
        "scrapingdog": {
            "free_daily": None,
            "free_monthly": 200,  # 1000 credits = 200 Google searches (5 credits each)
            "max_daily": None,
            "cost_per_1k": 1.00,  # $40/40k searches = $1/1k
            "reset_period": "monthly",
            "priority": 3
        },
        "serper": {
            "free_daily": None,
            "free_monthly": 2500,  # One-time signup bonus
            "max_daily": None,
            "cost_per_1k": 0.30,  # At volume
            "reset_period": "monthly",
            "priority": 4
        }
    }
    
    def __init__(self, persistence_file: str = "quota_usage.json"):
        self.persistence_file = persistence_file
        self.usage = self._load_usage()
        
        # Initialize usage tracking for all providers
        for provider in self.PROVIDER_CONFIG.keys():
            if provider not in self.usage:
                self.usage[provider] = {
                    "daily_count": 0,
                    "monthly_count": 0,
                    "daily_cost": 0.0,
                    "monthly_cost": 0.0,
                    "last_daily_reset": self._get_day_start(),
                    "last_monthly_reset": self._get_month_start(),
                    "total_queries": 0
                }
        
        logger.info("💰 QuotaManager initialized")
        self._log_quota_status()
    
    def get_available_provider(self, providers: list) -> Optional[str]:
        """
        Get the best available provider based on:
        1. Free tier availability
        2. Priority order
        3. Cost per query
        
        Returns provider name or None if all exhausted
        """
        
        # Check and reset quotas if needed
        self._check_and_reset_quotas()
        
        # Sort providers by priority
        sorted_providers = sorted(
            [p for p in providers if p in self.PROVIDER_CONFIG],
            key=lambda p: self.PROVIDER_CONFIG[p]["priority"]
        )
        
        for provider in sorted_providers:
            if self._is_provider_available(provider):
                logger.info(f"✅ Selected provider: {provider.upper()} (Priority: {self.PROVIDER_CONFIG[provider]['priority']})")
                return provider
        
        logger.warning("⚠️ All providers exhausted or unavailable!")
        return None
    
    def _is_provider_available(self, provider: str) -> bool:
        """Check if provider has quota available"""
        
        config = self.PROVIDER_CONFIG[provider]
        usage = self.usage[provider]
        
        # Check daily limit (if exists)
        if config["free_daily"]:
            if usage["daily_count"] >= config["free_daily"]:
                if config["max_daily"] and usage["daily_count"] >= config["max_daily"]:
                    logger.debug(f"❌ {provider}: Daily limit exhausted ({usage['daily_count']}/{config['max_daily']})")
                    return False
                # Still has paid quota available
                logger.debug(f"⚠️ {provider}: Free daily tier exhausted, using paid")
        
        # Check monthly limit
        if config["free_monthly"]:
            if usage["monthly_count"] >= config["free_monthly"]:
                logger.debug(f"⚠️ {provider}: Free monthly tier exhausted ({usage['monthly_count']}/{config['free_monthly']})")
                # For now, we'll still allow paid usage
                # In production, you might want to enforce a monthly budget
        
        return True
    
    def record_usage(
        self, 
        provider: str, 
        num_queries: int = 1,
        success: bool = True
    ) -> None:
        """Record API usage for a provider"""
        
        if provider not in self.usage:
            logger.warning(f"⚠️ Unknown provider: {provider}")
            return
        
        if not success:
            # Don't count failed queries
            return
        
        config = self.PROVIDER_CONFIG.get(provider, {})
        usage = self.usage[provider]
        
        # Update counts
        usage["daily_count"] += num_queries
        usage["monthly_count"] += num_queries
        usage["total_queries"] += num_queries
        
        # Calculate cost (if over free tier)
        is_free = self._is_in_free_tier(provider, usage)
        
        if not is_free:
            cost = (num_queries / 1000.0) * config.get("cost_per_1k", 0)
            usage["daily_cost"] += cost
            usage["monthly_cost"] += cost
            logger.info(f"💵 {provider}: ${cost:.4f} charged ({num_queries} queries)")
        else:
            logger.debug(f"🆓 {provider}: Free tier usage ({usage['monthly_count']}/{config.get('free_monthly', 'unlimited')})")
        
        # Persist to disk
        self._save_usage()
    
    def _is_in_free_tier(self, provider: str, usage: Dict) -> bool:
        """Check if current usage is within free tier"""
        
        config = self.PROVIDER_CONFIG[provider]
        
        # Check daily free tier
        if config["free_daily"] and usage["daily_count"] <= config["free_daily"]:
            return True
        
        # Check monthly free tier
        if config["free_monthly"] and usage["monthly_count"] <= config["free_monthly"]:
            return True
        
        return False
    
    def _check_and_reset_quotas(self) -> None:
        """Check if quotas need to be reset (daily/monthly)"""
        
        current_day = self._get_day_start()
        current_month = self._get_month_start()
        
        for provider, usage in self.usage.items():
            # Daily reset
            if usage["last_daily_reset"] < current_day:
                logger.info(f"🔄 {provider}: Resetting daily quota")
                usage["daily_count"] = 0
                usage["daily_cost"] = 0.0
                usage["last_daily_reset"] = current_day
            
            # Monthly reset
            if usage["last_monthly_reset"] < current_month:
                logger.info(f"🔄 {provider}: Resetting monthly quota")
                usage["monthly_count"] = 0
                usage["monthly_cost"] = 0.0
                usage["last_monthly_reset"] = current_month
        
        self._save_usage()
    
    def get_quota_status(self) -> Dict[str, Any]:
        """Get current quota status for all providers"""
        
        self._check_and_reset_quotas()
        
        status = {}
        
        for provider, usage in self.usage.items():
            config = self.PROVIDER_CONFIG.get(provider, {})
            
            status[provider] = {
                "daily_usage": f"{usage['daily_count']}/{config.get('free_daily', 'unlimited')}",
                "monthly_usage": f"{usage['monthly_count']}/{config.get('free_monthly', 'unlimited')}",
                "daily_cost": f"${usage['daily_cost']:.2f}",
                "monthly_cost": f"${usage['monthly_cost']:.2f}",
                "total_queries": usage["total_queries"],
                "in_free_tier": self._is_in_free_tier(provider, usage),
                "available": self._is_provider_available(provider)
            }
        
        return status
    
    def _log_quota_status(self) -> None:
        """Log current quota status"""
        
        logger.info("📊 Current Quota Status:")
        for provider, usage in self.usage.items():
            config = self.PROVIDER_CONFIG.get(provider, {})
            in_free = self._is_in_free_tier(provider, usage)
            tier = "🆓 FREE" if in_free else "💵 PAID"
            
            daily_str = f"{usage['daily_count']}/{config.get('free_daily', '∞')}" if config.get('free_daily') else "∞"
            monthly_str = f"{usage['monthly_count']}/{config.get('free_monthly', '∞')}"
            
            logger.info(f"  {provider.upper()}: {tier} | Daily: {daily_str} | Monthly: {monthly_str} | Cost: ${usage['monthly_cost']:.2f}")
    
    def _get_day_start(self) -> int:
        """Get timestamp for start of current day (UTC)"""
        now = datetime.now(timezone.utc)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return int(day_start.timestamp())
    
    def _get_month_start(self) -> int:
        """Get timestamp for start of current month (UTC)"""
        now = datetime.now(timezone.utc)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return int(month_start.timestamp())
    
    def _load_usage(self) -> Dict:
        """Load usage data from disk"""
        if os.path.exists(self.persistence_file):
            try:
                with open(self.persistence_file, 'r') as f:
                    data = json.load(f)
                    logger.info(f"📂 Loaded quota usage from {self.persistence_file}")
                    return data
            except Exception as e:
                logger.warning(f"⚠️ Failed to load quota data: {e}")
        
        return {}
    
    def _save_usage(self) -> None:
        """Save usage data to disk"""
        try:
            with open(self.persistence_file, 'w') as f:
                json.dump(self.usage, f, indent=2)
            logger.debug(f"💾 Saved quota usage to {self.persistence_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save quota data: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detailed statistics"""
        
        self._check_and_reset_quotas()
        
        total_queries = sum(u["total_queries"] for u in self.usage.values())
        total_cost = sum(u["monthly_cost"] for u in self.usage.values())
        
        return {
            "total_queries": total_queries,
            "total_monthly_cost": f"${total_cost:.2f}",
            "providers": self.get_quota_status()
        }