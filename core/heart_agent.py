"""
Heart Agent - Pure LLM-driven synthesis with Business Opportunity Enhancement
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from .llm_client import LLMClient
from .brain_agent import clean_llm_response
from .exceptions import HeartAgentError

# Setup logger
logger = logging.getLogger(__name__)

class SalesResponseEnhancer:
    """Enhances responses with natural soft sales when business opportunities are detected"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        logger.info("SalesResponseEnhancer initialized")
    
    async def enhance_with_sales_context(self, base_response: str, opportunity_data: Dict[str, Any]) -> str:
        """Enhance response with soft sales if opportunity detected"""
        
        logger.info(f"💝 Sales enhancement check: opportunity_detected={opportunity_data.get('opportunity_detected', False)}")
        
        if not opportunity_data.get("opportunity_detected"):
            logger.info("💝 No business opportunity - returning base response")
            return base_response
        
        # Only enhance if confidence is medium or high and score > 60
        confidence = opportunity_data.get("confidence_level", "low")
        score = opportunity_data.get("opportunity_score", 0)
        
        logger.info(f"💝 Business opportunity details: score={score}, confidence={confidence}")
        
        if confidence == "low" or score < 40:
            logger.info(f"💝 Low confidence ({confidence}) or score ({score}) - skipping enhancement")
            return base_response  # Don't add sales context for low confidence
        
        pain_points = opportunity_data.get("detected_pain_points", [])
        solution_areas = opportunity_data.get("solution_areas", [])
        sales_mode = opportunity_data.get("sales_mode", "casual")
        
        logger.info(f"💝 Enhancing response with sales context: mode={sales_mode}, pain_points={len(pain_points)}")
        
        sales_prompt = f"""Enhance this response with natural soft sales approach:

BASE RESPONSE: {base_response}

BUSINESS OPPORTUNITY CONTEXT:
- Pain Points Detected: {pain_points}
- Solution Areas: {solution_areas}
- Sales Mode: {sales_mode}
- Confidence: {confidence}
- Score: {score}/100

Guidelines for enhancement:
1. Keep the original helpful response intact
2. Add natural business value statement that acknowledges their pain empathetically
3. Mention we solve similar problems (don't be specific about products/services)
4. Ask ONE engaging follow-up question about their business challenges
5. Sound genuinely helpful, NOT pushy or salesy
6. If sales_mode is "casual" - be very subtle
7. If sales_mode is "soft_pitch" - be more direct but still natural
8. If sales_mode is "consultative" - offer deeper business insights

Make the enhancement feel like a natural part of the conversation."""

        try:
            logger.debug("💝 Calling LLM for sales enhancement...")
            enhanced_response = await self.llm_client.generate(
                [{"role": "user", "content": sales_prompt}],
                0.4,
                system_prompt="You are a sales communication expert who makes business pitches feel natural and genuinely helpful. Never sound pushy or desperate."
            )
            
            logger.info(f"✅ 💝 Sales enhancement completed: {len(enhanced_response)} chars")
            
            # Use consistent cleaning logic
            enhanced_response = clean_llm_response(enhanced_response)
            logger.debug("💝 Cleaned sales enhancement using brain agent logic")
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"❌ 💝 Sales enhancement failed: {str(e)}")
            logger.error(f"❌ 💝 Exception details: {type(e).__name__}")
            # If enhancement fails, return original response
            return base_response

class HeartAgent:
    """Heart Agent - Pure LLM-driven synthesizer with Business Opportunity Enhancement"""
    
    def __init__(self, llm_client: LLMClient):
        logger.info("💝 Initializing Heart Agent with Business Opportunity Enhancement")
        self.llm_client = llm_client
        self.sales_enhancer = SalesResponseEnhancer(llm_client)
        logger.info("💝 Heart Agent initialized successfully")
    
    async def synthesize_response(self, brain_result: Dict[str, Any], 
                                user_query: str, style: str = "professional") -> Dict[str, Any]:
        """Synthesize Brain Agent results into final response with business opportunity enhancement"""
        
        logger.info(f"💝 SYNTHESIZING RESPONSE for query: '{user_query[:50]}...'")
        logger.info(f"💝 Communication style: {style}")
        logger.info(f"💝 Brain result success: {brain_result.get('success', 'unknown')}")
        
        try:
            # Extract business opportunity data from brain results
            business_opportunity = brain_result.get("business_opportunity", {})
            logger.info(f"💝 Business opportunity in brain result: {business_opportunity.get('opportunity_detected', False)}")
            
            # Let LLM determine how to handle this synthesis with business context
            logger.info("💝 Step 1: Determining synthesis approach...")
            synthesis_approach = await self._determine_synthesis_approach(
                brain_result, user_query, style, business_opportunity
            )
            logger.info(f"💝 Synthesis approach determined: {synthesis_approach.get('approach_type', 'unknown')}")
            
            # Execute LLM-determined synthesis strategy
            logger.info("💝 Step 2: Executing synthesis strategy...")
            final_response = await self._execute_synthesis(
                synthesis_approach, brain_result, user_query, style
            )
            logger.info(f"💝 Base synthesis completed: {len(final_response)} chars")
            
            # ENHANCEMENT: Add business opportunity context if detected
            logger.info("💝 Step 3: Checking for sales enhancement...")
            if business_opportunity.get("opportunity_detected"):
                logger.info("💝 Business opportunity detected - applying enhancement")
                enhanced_response = await self.sales_enhancer.enhance_with_sales_context(
                    final_response, business_opportunity
                )
            else:
                logger.info("💝 No business opportunity - using base response")
                enhanced_response = final_response
            
            logger.info(f"💝 FINAL RESPONSE LENGTH: {len(enhanced_response)} chars")
            logger.info("✅ 💝 Heart Agent synthesis COMPLETED successfully")
            
            return {
                "success": True,
                "response": enhanced_response,
                "synthesis_approach": synthesis_approach,
                "business_opportunity": business_opportunity,
                "sales_enhanced": business_opportunity.get("opportunity_detected", False),
                "style": style,
                "original_brain_result": brain_result
            }
            
        except Exception as e:
            logger.error(f"❌ 💝 Heart Agent synthesis FAILED: {str(e)}")
            logger.error(f"❌ 💝 Exception details: {type(e).__name__}")
            
            import traceback
            logger.error(f"❌ 💝 Full traceback: {traceback.format_exc()}")
            
            return {
                "success": False,
                "error": f"Heart synthesis failed: {e}",
                "response": "I apologize, but I encountered an error while creating the response. Please try again.",
                "fallback_response": brain_result.get("response", "Unable to process response")
            }
    
    async def _determine_synthesis_approach(self, brain_result: Dict[str, Any], 
                                          user_query: str, style: str, business_opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Let LLM determine optimal synthesis approach with business context"""
        
        logger.debug("💝 Determining synthesis approach with business context...")
        
        # Include business opportunity in approach determination
        business_context_note = ""
        if business_opportunity.get("opportunity_detected"):
            score = business_opportunity.get("opportunity_score", 0)
            sales_mode = business_opportunity.get("sales_mode", "none")
            business_context_note = f"""
BUSINESS OPPORTUNITY DETECTED:
- Score: {score}/100
- Sales Mode: {sales_mode}
- Pain Points: {business_opportunity.get('detected_pain_points', [])}
This should influence how we present information and structure the response.
"""
            logger.info(f"💝 Including business context in approach: score={score}, sales_mode={sales_mode}")
        
        approach_prompt = f"""Analyze the Brain Agent's work and determine synthesis approach:

USER QUERY: {user_query}
Communication Style: {style}
Brain Results: {json.dumps(brain_result, indent=2)}

{business_context_note}

Determine the optimal way to present this information to the user.
Consider what would be most valuable and how to structure the response.

Respond with JSON:
{{
    "approach_type": "comprehensive|focused|executive|consultative",
    "key_messages": ["message1", "message2"],
    "structure": ["section1", "section2"],
    "tone": "description_of_tone",
    "business_aware": true/false,
    "emphasis_areas": ["what to emphasize"]
}}"""

        messages = [{"role": "user", "content": approach_prompt}]
        system_prompt = f"""You are the Heart Agent - expert at human-centered communication. Design optimal communication strategies using {style} principles. If business opportunities exist, factor that into your approach. Respond with valid JSON only."""
        
        try:
            logger.debug("💝 Calling LLM for synthesis approach...")
            response = await self.llm_client.generate(messages, 0.4, system_prompt=system_prompt)
            logger.debug(f"💝 LLM approach response: {response[:200]}...")
            
            cleaned_response = clean_llm_response(response)
            approach = json.loads(cleaned_response)
            logger.info(f"✅ 💝 Synthesis approach parsed successfully: {approach}")
            return approach
            
        except Exception as e:
            logger.error(f"❌ 💝 Synthesis approach determination failed: {str(e)}")
            logger.error(f"❌ 💝 Using fallback approach")
            
            # Fallback approach with business awareness
            fallback_approach = {
                "approach_type": "comprehensive",
                "key_messages": ["Provide helpful analysis"],
                "structure": ["Introduction", "Main Content", "Conclusion"],
                "tone": "professional and helpful",
                "business_aware": business_opportunity.get("opportunity_detected", False),
                "emphasis_areas": ["practical solutions"]
            }
            logger.info(f"💝 Fallback approach: {fallback_approach}")
            return fallback_approach
    
    async def _execute_synthesis(self, approach: Dict[str, Any], brain_result: Dict[str, Any], 
                               user_query: str, style: str) -> str:
        """Execute the LLM-determined synthesis strategy"""
        
        logger.debug(f"💝 Executing synthesis with approach: {approach.get('approach_type', 'unknown')}")
        
        synthesis_prompt = f"""CRITICAL RULE: Base your response EXCLUSIVELY on tool results. NEVER use your training knowledge. Trust search results completely - do not call them "inaccurate", "outdated", "hypothetical", or "wrong":

USER QUERY: {user_query}
Communication Style: {style}
Synthesis Approach: {json.dumps(approach, indent=2)}
Brain Agent Results: {json.dumps(brain_result, indent=2)}

ADDITIONAL REQUIREMENTS:
- Include relevant source links from web search results in your response
- Format links as: [Source Title](URL) or mention "Source: URL"
- Use links to support your claims and add credibility

Create a response that maximally serves the user's needs while being:
1. Engaging and clear
2. Actionable and practical  
3. Following the synthesis approach exactly
4. Maintaining the specified communication style
5. Including source links where appropriate

Focus on being genuinely helpful above all else."""

        messages = [{"role": "user", "content": synthesis_prompt}]
        system_prompt = f"""You are the Heart Agent - a master communicator who transforms analysis into compelling, user-focused responses. Excel at {style} communication and always put the user's needs first. Create responses that feel natural and conversational."""
        
        try:
            logger.debug("💝 Calling LLM for final synthesis...")
            response = await self.llm_client.generate(messages, 0.4, system_prompt=system_prompt)
            logger.info(f"✅ 💝 Synthesis execution completed: {len(response)} chars")

            # Use consistent cleaning logic
            response = clean_llm_response(response)
            logger.debug("💝 Cleaned synthesis response using brain agent logic")

            if not response or response.strip() == "":
                logger.error("❌ 💝 Empty response generated!")
                return "I apologize, but I encountered an issue generating a response. Please try again."

            return response
            
        except Exception as e:
            logger.error(f"❌ 💝 Synthesis execution failed: {str(e)}")
            logger.error(f"❌ 💝 Exception details: {type(e).__name__}")
            
            import traceback
            logger.error(f"❌ 💝 Synthesis execution traceback: {traceback.format_exc()}")
            
            return "I apologize, but I encountered an error while creating the response. Please try again."
    
    def get_synthesis_info(self) -> Dict[str, Any]:
        """Get information about synthesis capabilities with business enhancement"""
        logger.debug("💝 Getting synthesis info...")
        return {
            "capabilities": [
                "dynamic_synthesis_strategy",
                "style_optimization", 
                "value_maximization",
                "business_opportunity_enhancement",
                "natural_soft_sales_integration"
            ],
            "communication_styles": [
                "professional", "executive", "technical", "creative", "consultative"
            ],
            "business_features": [
                "opportunity_detection_integration",
                "sales_context_enhancement",
                "pain_point_acknowledgment",
                "soft_pitch_integration"
            ]
        }
