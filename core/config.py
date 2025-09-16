"""
Configuration system for Brain-Heart Deep Research System
Simplified version to avoid import issues
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
    temperature: float = 0.7
    max_tokens: int = 4000
    timeout: int = 30
    base_url: Optional[str] = None

class Config:
    """Dynamic configuration manager - simplified version"""
    
    def __init__(self):
        self.available_providers: Dict[str, str] = {}
        self.available_models: Dict[str, List[str]] = {}
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
        }
        
        for env_key, provider in provider_mappings.items():
            api_key = os.getenv(env_key)
            if api_key:
                self.available_providers[provider] = api_key
                print(f"Detected {provider.upper()} API key")
        
        if not self.available_providers:
            raise Exception("No LLM provider API keys found. Please set at least one API key in .env file.")
    
    def _load_available_models(self):
        """Load available models for each provider"""
        
        model_catalog = {
            'openai': ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo'],
            'anthropic': ['claude-3-5-sonnet-20241022', 'claude-3-opus-20240229', 'claude-3-haiku-20240307'],
            'openrouter': ['qwen/qwen3-next-80b-a3b-thinking','anthropic/claude-3.5-sonnet', 'openai/gpt-4o', 'meta-llama/llama-3.1-70b-instruct'],
            'groq': ['llama-3.1-70b-versatile', 'llama-3.1-8b-instant', 'mixtral-8x7b-32768'],
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
    
    def create_llm_config(self, provider: str, model: str, 
                         temperature: float = 0.7, max_tokens: int = 4000) -> LLMConfig:
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
        
        return LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url
        )
    
    def get_tool_configs(self) -> Dict[str, Any]:
        """Get tool configurations"""
        
        tools = {
            'calculator': {'enabled': True},
            'web_search': {
                'enabled': bool(os.getenv('SERPER_API_KEY') or os.getenv('VALUESERP_API_KEY')),
                'primary_key': os.getenv('SERPER_API_KEY'),
                'provider': 'serper'
            },
            'social_search': {
                'enabled': bool(os.getenv('REDDIT_CLIENT_ID') or os.getenv('YOUTUBE_API_KEY')),
                'reddit_enabled': bool(os.getenv('REDDIT_CLIENT_ID')),
                'youtube_enabled': bool(os.getenv('YOUTUBE_API_KEY'))
            },
            'rag': {'enabled': True},
            'reasoning': {'enabled': True}
        }
        
        return tools
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return {
            'available_providers': list(self.available_providers.keys()),
            'available_models': self.available_models,
            'tool_configs': self.get_tool_configs()
        }