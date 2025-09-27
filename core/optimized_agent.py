"""
Optimized Single-Pass Agent System
Combines semantic analysis, tool execution, and response generation in minimal LLM calls
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class OptimizedAgent:
    """Single-pass agent that minimizes LLM calls while maintaining all functionality"""
    
    def __init__(self, llm_client, tool_manager):
        self.llm_client = llm_client
        self.tool_manager = tool_manager
        self.available_tools = tool_manager.get_available_tools()
        logger.info(f"OptimizedAgent initialized with tools: {self.available_tools}")
    
    async def process_query(self, query: str, chat_history: List[Dict] = None, user_id: str = None) -> Dict[str, Any]:
        """Process query with minimal LLM calls"""
        
        logger.info(f"🚀 PROCESSING QUERY: '{query[:50]}...'")
        start_time = datetime.now()
        
        try:
            # STEP 1: Single comprehensive analysis (combines semantic + business + planning)
            analysis_start = datetime.now()
            analysis = await self._comprehensive_analysis(query, chat_history)
            analysis_time = (datetime.now() - analysis_start).total_seconds()
            logger.info(f"⏱️ Analysis completed in {analysis_time:.2f}s")
            
            # STEP 2: Execute tools if needed (no LLM calls)
            tool_start = datetime.now()
            tool_results = await self._execute_tools(
                analysis.get('tools_to_use', []),
                query,
                user_id
            )
            tool_time = (datetime.now() - tool_start).total_seconds()
            logger.info(f"⏱️ Tools executed in {tool_time:.2f}s")
            
            # STEP 3: Generate final response (single LLM call)
            response_start = datetime.now()
            final_response = await self._generate_response(
                query,
                analysis,
                tool_results,
                chat_history
            )
            response_time = (datetime.now() - response_start).total_seconds()
            logger.info(f"⏱️ Response generated in {response_time:.2f}s")
            
            total_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ TOTAL PROCESSING TIME: {total_time:.2f}s (2 LLM calls only)")
            
            return {
                "success": True,
                "response": final_response,
                "analysis": analysis,
                "tool_results": tool_results,
                "tools_used": analysis.get('tools_to_use', []),
                "business_opportunity": analysis.get('business_opportunity', {}),
                "processing_time": {
                    "analysis": analysis_time,
                    "tools": tool_time,
                    "response": response_time,
                    "total": total_time
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Processing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": "I apologize, but I encountered an error. Please try again."
            }
    
    async def _comprehensive_analysis(self, query: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        """Single LLM call for ALL analysis needs"""
        
        # Build context from chat history
        context = self._build_context(chat_history)
        
        # Create comprehensive prompt that does everything in one shot
        analysis_prompt = f"""Analyze this query comprehensively and return a complete execution plan:

USER QUERY: {query}
RECENT CONTEXT: {context}

AVAILABLE TOOLS:
- web_search: Search internet for current information
- rag: Retrieve from uploaded knowledge base  
- calculator: Perform calculations and analysis

Perform ALL of the following analyses in ONE response:

1. SEMANTIC INTENT (what user really wants)
2. BUSINESS OPPORTUNITY DETECTION:
   - Is this a business pain point? (customer support, manual processes, scaling issues, etc.)
   - Score 0-100 (0=no opportunity, 100=strong opportunity)
   - Only mark as opportunity if it's BUSINESS-RELATED, not personal
3. TOOL SELECTION:
   - What tools are needed? (can be multiple or none)
   - Use web_search for: current info, news, market data, trends
   - Use rag for: "my product/company/documents" or uploaded files
   - Use calculator for: math, percentages, financial metrics
   - Use NO tools for: greetings, thanks, casual chat
4. SENTIMENT & PERSONALITY:
   - User's emotional state (frustrated/excited/casual/urgent/confused)
   - Best response personality (empathetic_friend/excited_buddy/helpful_dost/urgent_solver/patient_guide)
5. RESPONSE STRATEGY:
   - Response length (micro/short/medium/detailed)
   - Language style (hinglish/english/professional/casual)

Return ONLY valid JSON:
{{
    "semantic_intent": "clear description of what user wants",
    "business_opportunity": {{
        "detected": true/false,
        "score": 0-100,
        "pain_points": ["specific problems"],
        "solution_areas": ["how we can help"],
        "sales_mode": "none|casual|soft_pitch|consultative"
    }},
    "tools_to_use": ["tool1", "tool2"],
    "tool_reasoning": "why these tools",
    "sentiment": {{
        "primary_emotion": "frustrated|excited|casual|urgent|confused",
        "intensity": "low|medium|high"
    }},
    "response_strategy": {{
        "personality": "empathetic_friend|excited_buddy|helpful_dost|urgent_solver|patient_guide",
        "length": "micro|short|medium|detailed",
        "language": "hinglish|english|professional|casual",
        "tone": "friendly|professional|empathetic|excited"
    }},
    "key_points_to_address": ["point1", "point2"]
}}"""

        try:
            response = await self.llm_client.generate(
                [{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                system_prompt="You are an expert analyst. Analyze queries comprehensively covering semantics, business opportunities, tool needs, and communication strategy. Return valid JSON only."
            )
            
            # Clean response
            cleaned = self._clean_json_response(response)
            analysis = json.loads(cleaned)
            
            logger.info(f"✅ Analysis complete: intent={analysis.get('semantic_intent')}, "
                       f"business={analysis.get('business_opportunity', {}).get('detected')}, "
                       f"tools={analysis.get('tools_to_use', [])}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            # Return safe defaults
            return {
                "semantic_intent": "general_inquiry",
                "business_opportunity": {"detected": False, "score": 0},
                "tools_to_use": [],
                "sentiment": {"primary_emotion": "casual", "intensity": "medium"},
                "response_strategy": {
                    "personality": "helpful_dost",
                    "length": "medium",
                    "language": "hinglish",
                    "tone": "friendly"
                }
            }
    
    async def _execute_tools(self, tools: List[str], query: str, user_id: str = None) -> Dict[str, Any]:
        """Execute tools in parallel for speed"""
        
        if not tools:
            return {}
        
        results = {}
        
        # Execute tools in parallel for speed
        tasks = []
        for tool in tools:
            if tool in self.available_tools:
                task = self.tool_manager.execute_tool(tool, query=query, user_id=user_id)
                tasks.append((tool, task))
        
        if tasks:
            # Gather all results in parallel
            for tool_name, task in tasks:
                try:
                    result = await task
                    results[tool_name] = result
                    logger.info(f"✅ Tool {tool_name} executed successfully")
                except Exception as e:
                    logger.error(f"❌ Tool {tool_name} failed: {e}")
                    results[tool_name] = {"error": str(e)}
        
        return results
    
    async def _generate_response(self, query: str, analysis: Dict, tool_results: Dict, chat_history: List[Dict]) -> str:
        """Single LLM call to generate final response"""
        
        # Extract key elements
        intent = analysis.get('semantic_intent', '')
        business_opp = analysis.get('business_opportunity', {})
        sentiment = analysis.get('sentiment', {})
        strategy = analysis.get('response_strategy', {})
        
        # Format tool results
        tool_data = self._format_tool_results(tool_results)
        
        # Build memory context to avoid repetition
        recent_phrases = self._extract_recent_phrases(chat_history)
        
        response_prompt = f"""Generate a natural response based on this analysis:

USER QUERY: {query}
INTENT: {intent}
KEY POINTS TO ADDRESS: {analysis.get('key_points_to_address', [])}

AVAILABLE DATA FROM TOOLS:
{tool_data}

BUSINESS CONTEXT:
- Opportunity Detected: {business_opp.get('detected', False)}
- Score: {business_opp.get('score', 0)}/100
- Pain Points: {business_opp.get('pain_points', [])}
- Sales Mode: {business_opp.get('sales_mode', 'none')}

RESPONSE REQUIREMENTS:
- Personality: {strategy.get('personality', 'helpful_dost')}
- Length: {strategy.get('length', 'medium')} 
- Language: {strategy.get('language', 'hinglish')}
- Tone: {strategy.get('tone', 'friendly')}
- User Emotion: {sentiment.get('primary_emotion', 'casual')} ({sentiment.get('intensity', 'medium')})

AVOID THESE RECENT PHRASES: {recent_phrases}

CRITICAL RULES:
1. Use tool data naturally without announcing "I found" or "Let me search"
2. Never repeat user's words back to them
3. If business opportunity detected with score > 60, weave in soft sales naturally
4. Match user's emotional state with appropriate empathy
5. End with engaging question if appropriate
6. For jokes/fun requests, be creative and entertaining
7. For technical queries, be precise and helpful

Generate a response that feels natural, helpful, and conversational."""

        try:
            # Determine max tokens based on length strategy
            max_tokens = {
                "micro": 150,
                "short": 300,
                "medium": 500,
                "detailed": 700
            }.get(strategy.get('length', 'medium'), 500)
            
            response = await self.llm_client.generate(
                [{"role": "user", "content": response_prompt}],
                temperature=0.4,
                max_tokens=max_tokens,
                system_prompt="You are Mochand Dost - a naturally helpful AI friend who becomes a smart sales consultant when needed. Create responses that are engaging, helpful, and conversational."
            )
            
            # Clean and format
            response = self._clean_response(response)
            
            logger.info(f"✅ Response generated: {len(response)} chars")
            return response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "I apologize, but I had trouble generating a response. Could you please try again?"
    
    def _build_context(self, chat_history: List[Dict]) -> str:
        """Build context from chat history"""
        if not chat_history:
            return "No previous context"
        
        recent = chat_history[-3:] if len(chat_history) >= 3 else chat_history
        context_parts = []
        for item in recent:
            if 'query' in item:
                context_parts.append(f"User: {item['query'][:100]}")
            elif 'content' in item:
                context_parts.append(f"{item.get('role', 'user')}: {item['content'][:100]}")
        
        return " | ".join(context_parts) if context_parts else "No previous context"
    
    def _format_tool_results(self, tool_results: Dict) -> str:
        """Format tool results for response generation"""
        if not tool_results:
            return "No external data available"
        
        formatted = []
        for tool, result in tool_results.items():
            if isinstance(result, dict) and 'error' not in result:
                if 'data' in result:
                    formatted.append(f"{tool.upper()} DATA: {result['data']}")
                elif 'result' in result:
                    formatted.append(f"{tool.upper()} RESULT: {result['result']}")
            elif isinstance(result, str):
                formatted.append(f"{tool.upper()}: {result}")
        
        return "\n".join(formatted) if formatted else "No usable tool data"
    
    def _extract_recent_phrases(self, chat_history: List[Dict]) -> List[str]:
        """Extract recent phrases to avoid repetition"""
        if not chat_history:
            return []
        
        phrases = []
        for msg in chat_history[-4:]:
            if msg.get('role') == 'assistant' or 'response' in msg:
                content = msg.get('content', msg.get('response', ''))
                # Get first 30 chars of each sentence
                sentences = content.split('.')
                for sentence in sentences[:2]:
                    if sentence.strip():
                        phrases.append(sentence.strip()[:30])
        
        return phrases[-5:] if phrases else []
    
    def _clean_json_response(self, response: str) -> str:
        """Clean LLM response for JSON parsing"""
        response = response.strip()
        
        # Remove thinking tags
        if '<think>' in response:
            end_idx = response.find('</think>')
            if end_idx != -1:
                response = response[end_idx + 8:].strip()
        
        # Remove markdown blocks
        if response.startswith('```'):
            lines = response.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            response = '\n'.join(lines)
        
        return response
    
    def _clean_response(self, response: str) -> str:
        """Clean final response for display"""
        # Remove any system thinking
        response = self._clean_json_response(response)
        
        # Fix formatting for display
        response = response.replace('•', '-')
        response = response.strip()
        
        return response


class StreamlitOptimizedApp:
    """Streamlit integration for optimized agent"""
    
    def __init__(self, llm_client, tool_manager):
        self.agent = OptimizedAgent(llm_client, tool_manager)
        logger.info("StreamlitOptimizedApp initialized")
    
    async def process_message(self, message: str, chat_history: List[Dict], user_id: str = None) -> Dict[str, Any]:
        """Process message with optimized agent"""
        
        logger.info(f"📨 Processing message: '{message[:50]}...'")
        
        # Process with optimized agent
        result = await self.agent.process_query(message, chat_history, user_id)
        
        # Log performance metrics
        if result.get('success') and 'processing_time' in result:
            times = result['processing_time']
            logger.info(f"⚡ Performance Metrics:")
            logger.info(f"   Analysis: {times['analysis']:.2f}s")
            logger.info(f"   Tools: {times['tools']:.2f}s")
            logger.info(f"   Response: {times['response']:.2f}s")
            logger.info(f"   TOTAL: {times['total']:.2f}s")
            
            # Alert if too slow
            if times['total'] > 3.0:
                logger.warning(f"⚠️ Slow response detected: {times['total']:.2f}s")
        
        return result


# Example integration
async def main():
    """Example usage"""
    from your_llm_client import LLMClient  # Your existing LLM client
    from your_tool_manager import ToolManager  # Your existing tool manager
    
    # Initialize
    llm_client = LLMClient(api_key="your_key")
    tool_manager = ToolManager()
    
    # Create optimized app
    app = StreamlitOptimizedApp(llm_client, tool_manager)
    
    # Process message
    result = await app.process_message(
        "I'm struggling with customer support",
        chat_history=[],
        user_id="user123"
    )
    
    print(f"Response: {result['response']}")
    print(f"Business Opportunity: {result.get('business_opportunity', {}).get('detected')}")
    print(f"Processing Time: {result.get('processing_time', {}).get('total')}s")


if __name__ == "__main__":
    asyncio.run(main())