"""
Heart Agent - Pure LLM-driven synthesis
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from .llm_client import LLMClient
from .exceptions import HeartAgentError

class HeartAgent:
    """Heart Agent - Pure LLM-driven synthesizer"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    async def synthesize_response(self, brain_result: Dict[str, Any], 
                                user_query: str, style: str = "professional") -> Dict[str, Any]:
        """Synthesize Brain Agent results into final response"""
        
        try:
            # Let LLM determine how to handle this synthesis
            synthesis_approach = await self._determine_synthesis_approach(
                brain_result, user_query, style
            )
            
            # Execute LLM-determined synthesis strategy
            final_response = await self._execute_synthesis(
                synthesis_approach, brain_result, user_query, style
            )
            
            return {
                "success": True,
                "response": final_response,
                "synthesis_approach": synthesis_approach,
                "style": style,
                "original_brain_result": brain_result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Heart synthesis failed: {e}",
                "fallback_response": brain_result.get("response", "Unable to process response")
            }
    
    async def _determine_synthesis_approach(self, brain_result: Dict[str, Any], 
                                          user_query: str, style: str) -> Dict[str, Any]:
        """Let LLM determine optimal synthesis approach"""
        
        approach_prompt = f"""Analyze the Brain Agent's work and determine synthesis approach:

User Query: {user_query}
Communication Style: {style}
Brain Results: {json.dumps(brain_result, indent=2)}

Determine the optimal way to present this information to the user.
Consider what would be most valuable and how to structure the response.

Respond with JSON:
{{
    "approach_type": "comprehensive|focused|executive",
    "key_messages": ["message1", "message2"],
    "structure": ["section1", "section2"],
    "tone": "description_of_tone"
}}"""

        messages = [{"role": "user", "content": approach_prompt}]
        system_prompt = f"""You are the Heart Agent - expert at human-centered communication. Design optimal communication strategies using {style} principles. Respond with valid JSON only."""
        
        try:
            response = await self.llm_client.generate(messages, system_prompt, temperature=0.6)
            approach = json.loads(response)
            return approach
        except:
            # Fallback approach
            return {
                "approach_type": "comprehensive",
                "key_messages": ["Provide helpful analysis"],
                "structure": ["Introduction", "Main Content", "Conclusion"],
                "tone": "professional and helpful"
            }
    
    async def _execute_synthesis(self, approach: Dict[str, Any], brain_result: Dict[str, Any], 
                               user_query: str, style: str) -> str:
        """Execute the LLM-determined synthesis strategy"""
        
        synthesis_prompt = f"""Create the final response for the user using this synthesis approach:

User Query: {user_query}
Communication Style: {style}
Synthesis Approach: {json.dumps(approach, indent=2)}
Brain Agent Results: {json.dumps(brain_result, indent=2)}

Create a response that maximally serves the user's needs while being engaging, clear, and actionable.
Follow the synthesis approach exactly."""

        messages = [{"role": "user", "content": synthesis_prompt}]
        system_prompt = f"""You are the Heart Agent - a master communicator who transforms analysis into compelling, user-focused responses. Excel at {style} communication and always put the user's needs first."""
        
        return await self.llm_client.generate(messages, system_prompt, temperature=0.7)
    
    def get_synthesis_info(self) -> Dict[str, Any]:
        """Get information about synthesis capabilities"""
        return {
            "capabilities": [
                "dynamic_synthesis_strategy",
                "style_optimization",
                "value_maximization"
            ],
            "communication_styles": [
                "professional", "executive", "technical", "creative"
            ]
        }