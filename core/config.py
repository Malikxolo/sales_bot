"""
Configuration system for Brain-Heart Deep Research System
FIXED VERSION - Proper web model handling
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

@dataclass
class LLMConfig:
    """Configuration for LLM models"""
    provider: str
    model: str
    api_key: str
    max_tokens: int = 4000
    timeout: int = 30
    base_url: Optional[str] = None

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