"""
Universal LLM client supporting multiple providers
Simplified version to avoid import issues
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Any, Optional

class LLMClient:
    """Universal async LLM client with multi-provider support"""
    
    def __init__(self, config):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        await self.start_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_session()
    
    async def start_session(self):
        """Start HTTP session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=getattr(self.config, 'timeout', 30))
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def close_session(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def generate(self, messages: List[Dict[str, str]], 
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None) -> str:
        """Generate response using configured LLM"""
        
        if not self.session:
            await self.start_session()
        
        temp = temperature if temperature is not None else getattr(self.config, 'temperature', 0.7)
        tokens = max_tokens if max_tokens is not None else getattr(self.config, 'max_tokens', 4000)
        
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        try:
            if self.config.provider == 'anthropic':
                return await self._anthropic_request(messages, temp, tokens)
            elif self.config.provider in ['openai', 'openrouter', 'groq']:
                return await self._openai_compatible_request(messages, temp, tokens)
            else:
                raise Exception(f"Unsupported provider: {self.config.provider}")
        except Exception as e:
            raise Exception(f"Generation failed: {e}")
    
    async def _anthropic_request(self, messages: List[Dict[str, str]], 
                               temperature: float, max_tokens: int) -> str:
        """Handle Anthropic API requests"""
        
        system_content = ""
        user_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                user_messages.append(msg)
        
        headers = {
            "x-api-key": self.config.api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": user_messages
        }
        
        if system_content:
            payload["system"] = system_content
        
        async with self.session.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload
        ) as response:
            
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"API error {response.status}: {error_text}")
            
            result = await response.json()
            return result["content"][0]["text"]
    
    async def _openai_compatible_request(self, messages: List[Dict[str, str]], 
                                       temperature: float, max_tokens: int) -> str:
        """Handle OpenAI-compatible API requests"""
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        if self.config.provider == 'openrouter':
            headers["HTTP-Referer"] = "https://github.com/brain-heart-research"
            headers["X-Title"] = "Brain-Heart Research System"
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if hasattr(self.config, 'base_url') and self.config.base_url:
            url = f"{self.config.base_url}/chat/completions"
        else:
            url = "https://api.openai.com/v1/chat/completions"
        
        async with self.session.post(url, headers=headers, json=payload) as response:
            
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"API error {response.status}: {error_text}")
            
            result = await response.json()
            return result["choices"][0]["message"]["content"]
    
    def get_model_info(self) -> Dict[str, str]:
        """Get model information"""
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "temperature": str(getattr(self.config, 'temperature', 0.7)),
            "max_tokens": str(getattr(self.config, 'max_tokens', 4000))
        }