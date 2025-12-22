"""
Configuration system for Brain-Heart Deep Research System
FIXED VERSION - Proper web model handling
"""

import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from mem0.configs.base import MemoryConfig
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from os import getenv
from typing import Callable, Coroutine, Tuple, Any

logger = logging.getLogger(__name__)

custom_fact_extraction_prompt = """
    You are an extractor. From the conversation below, return a JSON object with a single key "facts" whose value is a list of concise strings.

    Rules:
    - Include short declarative facts (e.g., "User loves pizza").
    - **Always** include user questions/enquiries as facts in the format: "User asked: <question text>".
    - Output only valid JSON.

    Examples:
    Input: "I love pizza." => {"facts":["User loves pizza"]}
    Input: "How do I reset my password?" => {"facts":["User asked for instruction on resetting password?"]}
    Input: "What is photosynthesis?" => {"facts":["User asked about photosynthesis?"]}

    DO NOT store assistant responses.
    Conversation:
"""

memory_config = MemoryConfig(
    graph_store={
        "provider": "neo4j",
        "config": {
            "url": getenv('NEO4J_URL'),
            "username": getenv('NEO4J_USER'),
            "password": getenv('NEO4J_PASSWORD')
        }
    },
    vector_store={
        "provider": "chroma",
        "config": {
            "collection_name": "mem0_collection",
            "path": ".chromadb"
        }
    },
    custom_fact_extraction_prompt=custom_fact_extraction_prompt
    
)

SARVAM_SUPPORTED_LANGUAGES:set = {
    "Hindi",
    "Bengali",
    "Gujarati",
    "Kannada",
    "Malayalam",
    "Marathi",
    "Odia",
    "Punjabi",
    "Tamil",
    "Telugu"
}


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
    timeout: int = 120  # Thinking models need 60-120s for complex reasoning
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
        # Language detection configuration
        self.language_detection_enabled = os.getenv('LANGUAGE_DETECTION_ENABLED', 'false').lower() == 'true'
        self.language_detection_provider = os.getenv('LANGUAGE_DETECTION_PROVIDER', 'openrouter')
        self.language_detection_model = os.getenv('LANGUAGE_DETECTION_MODEL', 'google/gemini-2.5-flash-lite-preview-09-2025')
        
        # Language detection configuration
        self.language_detection_enabled = os.getenv('LANGUAGE_DETECTION_ENABLED', 'false').lower() == 'true'
        self.language_detection_provider = os.getenv('LANGUAGE_DETECTION_PROVIDER', 'openrouter')
        self.language_detection_model = os.getenv('LANGUAGE_DETECTION_MODEL', 'google/gemini-2.5-flash-lite-preview-09-2025')
        
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
            'DEEPSEEK_API_KEY': 'deepseek',
            'SARVAM_API_KEY': 'sarvam'
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
        elif provider == 'sarvam':
            base_url = "https://api.sarvam.ai/v1"
        elif provider == 'sarvam':
            base_url = "https://api.sarvam.ai/v1"
        
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
        
        # LLMLayer configuration
        llmlayer_enabled = os.getenv('LLMLAYER_ENABLED', 'false').lower() == 'true'
        llmlayer_key = os.getenv('LLMLAYER_API_KEY')
        llmlayer_url = os.getenv('LLMLAYER_API_URL', 'https://api.llmlayer.dev/api/v2/answer')
        
        print(f"🔍 DEBUG: JINA_API_KEY from env: {jina_key[:20] if jina_key else '❌ NOT FOUND IN ENV!'}")
        print(f"🌐 DEBUG: LLMLAYER_ENABLED: {llmlayer_enabled}")
        print(f"🌐 DEBUG: LLMLAYER_API_KEY: {llmlayer_key[:20] if llmlayer_key else '❌ NOT FOUND!'}")

        
        # Web search is enabled ONLY if explicitly enabled AND we have API keys
        web_search_env_enabled = os.getenv('WEB_SEARCH_ENABLED', 'true').lower() == 'true'
        web_search_has_keys = bool(openrouter_key or scrapingdog_key or valueserp_key or llmlayer_key)
        web_search_enabled = web_search_env_enabled and web_search_has_keys
        
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
                'llmlayer_enabled': llmlayer_enabled,
                'llmlayer_key': llmlayer_key,
                'llmlayer_url': llmlayer_url,
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
    
    def create_language_detection_config(self) -> LLMConfig:
        """Create LLM configuration for language detection layer"""
        if not self.language_detection_enabled:
            raise ValueError("Language detection is disabled in configuration")
        
        if self.language_detection_provider not in self.available_providers:
            raise ValueError(f"Language detection provider '{self.language_detection_provider}' not available")
        
        api_key = self.available_providers[self.language_detection_provider]
        
        # Provider-specific base URL
        base_url = None
        if self.language_detection_provider == 'openrouter':
            base_url = 'https://openrouter.ai/api/v1'
        
        return LLMConfig(
            provider=self.language_detection_provider,
            model=self.language_detection_model,
            api_key=api_key,
            max_tokens=500,  # Language detection needs minimal tokens
            timeout=30,  # Fast model, shorter timeout
            base_url=base_url
        )
    
    def create_language_detection_config(self) -> LLMConfig:
        """Create LLM configuration for language detection layer"""
        if not self.language_detection_enabled:
            raise ValueError("Language detection is disabled in configuration")
        
        if self.language_detection_provider not in self.available_providers:
            raise ValueError(f"Language detection provider '{self.language_detection_provider}' not available")
        
        api_key = self.available_providers[self.language_detection_provider]
        
        # Provider-specific base URL
        base_url = None
        if self.language_detection_provider == 'openrouter':
            base_url = 'https://openrouter.ai/api/v1'
        
        return LLMConfig(
            provider=self.language_detection_provider,
            model=self.language_detection_model,
            api_key=api_key,
            max_tokens=500,  # Language detection needs minimal tokens
            timeout=30,  # Fast model, shorter timeout
            base_url=base_url
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return {
            'available_providers': list(self.available_providers.keys()),
            'available_models': self.available_models,
            'available_web_models': self.available_web_models,
            'tool_configs': self.get_tool_configs()
        }
        
        
ENABLE_SCRAPING_CONFIRMATION, SCRAPING_CONFIRMATION_THRESHOLD, ESTIMATED_TIME_PER_PAGE, ENABLE_LLM_CONFIRMATION_DETECTION = False, 5, 3, True  # For testing; override with .env or config as needed