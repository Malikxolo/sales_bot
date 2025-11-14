"""
Web Search Agent for Perplexity API integration
FIXED VERSION - Enhanced error handling and logging
"""

from openai import AsyncOpenAI
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from os import getenv
import aiohttp
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client for OpenRouter API
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv('OPENROUTER_API_KEY')
)

async def search_perplexity(query: str, model: str = 'perplexity/sonar') -> str:
    """
    Search using Perplexity models via OpenRouter API
    
    Args:
        query: The search query
        model: The Perplexity model to use (default: perplexity/sonar)
    
    Returns:
        The search response as a string
    """
    try:
        logger.info(f"Searching with model {model} for query: {query[:100]}...")
        modified_query = f"""{query}

        Structure your response clearly:

        1. DIRECT ANSWER: Provide the key information that answers the query

        2. CONDITIONAL INFO: If your answer depends on information not provided by the user (like their location, specific brand/model, budget, dates), explicitly state: "This information requires: [what's needed]"

        3. ALTERNATIVES: Suggest alternatives if applicable

        Be concise and well-organized."""
        # Validate API key
        if not getenv('OPENROUTER_API_KEY'):
            raise Exception("OPENROUTER_API_KEY not found in environment variables")
        
        # Validate model
        valid_models = [
            'perplexity/sonar',
            'perplexity/sonar-deep-research', 
            'perplexity/sonar-pro',
            'perplexity/sonar-reasoning-pro'
        ]
        
        if model not in valid_models:
            logger.warning(f"Model {model} not in validated list, proceeding anyway")
        
        messages = [{
            "role": "user",
            "content": modified_query
        }]
        
        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4000,
            temperature=0.3
        )
        
        response = completion.choices[0].message.content
        urls = list({
            ann.url_citation.url
            for choice in getattr(completion, "choices", [])
            if (message := getattr(choice, "message", None))
            for ann in getattr(message, "annotations", [])
            if getattr(ann, "type", "") == "url_citation"
            and hasattr(ann, "url_citation")
            and getattr(ann.url_citation, "url", None)
        })

        logger.info(f"✅ Search completed successfully with {len(response)} characters")

        # Format sources cleanly at the end
        if urls:
            source_section = "\n\n---\nSources:\n" + "\n".join(f"- {url}" for url in urls)
            return response + source_section
        else:
            return response
        
    except Exception as e:
        error_msg = f"Exception occurred in web-search agent: {e}"
        logger.error(error_msg)
        
        # Return a structured error response instead of failing
        return f"Search failed: {str(e)}. Please check your OPENROUTER_API_KEY and try again."

async def search_llmlayer(query: str, api_key: str, api_url: str) -> str:
    """
    Search using LLMLayer API - returns pre-formatted answer
    
    Args:
        query: Search query (can be comma-separated for multiple queries)
        api_key: LLMLayer API key
        api_url: LLMLayer API endpoint
    
    Returns:
        Pre-formatted text response
    """
    try:
        logger.info(f"🌐 LLMLayer search for query: {query}")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Answer API payload - query breaking + search + scrape + summarize
        payload = {
            "query": query,
            "model": "groq/openai-gpt-oss-20b",
            "return_sources": True,
            "location": "in"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                api_url, 
                headers=headers, 
                json=payload, 
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"LLMLayer API error {response.status}: {error_text[:200]}")
                
                data = await response.json()
                answer = data.get("answer", "")
                
                if not answer:
                    raise Exception("No answer received from LLMLayer")
                
                logger.info(f"✅ LLMLayer search completed: {len(answer)} characters")
                return answer
                
    except Exception as e:
        error_msg = f"LLMLayer search failed: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

async def test_search_functionality():
    """Test function to verify search functionality"""
    try:
        test_query = "What are the latest developments in AI research?"
        test_model = "perplexity/sonar"
        
        logger.info("Testing search functionality...")
        result = await search_perplexity(test_query, test_model)
        
        if "Search failed:" in result:
            logger.error("❌ Search test failed")
            return False
        else:
            logger.info("✅ Search test successful")
            return True
            
    except Exception as e:
        logger.error(f"❌ Search test failed with exception: {e}")
        return False

async def main():
    """Main function for testing"""
    # Test with different models
    test_models = [
        'perplexity/sonar',
        'perplexity/sonar-deep-research', 
        'perplexity/sonar-pro',
        'perplexity/sonar-reasoning-pro'
    ]
    
    query = 'Best laptop brands in India 2024'
    
    logger.info("Testing different Perplexity models...")
    
    for model in test_models:
        logger.info(f"\n--- Testing {model} ---")
        try:
            result = await search_perplexity(query, model)
            logger.info(f"Result length: {len(result)} characters")
            print(f"\n{model} Result:")
            print(result[:200] + "..." if len(result) > 200 else result)
        except Exception as e:
            logger.error(f"Failed to test {model}: {e}")

if __name__ == '__main__':
    asyncio.run(main())