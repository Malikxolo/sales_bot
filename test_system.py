#!/usr/bin/env python3
"""
Test script for Brain-Heart Deep Research System
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def test_system():
    """Test basic system functionality"""
    
    try:
        from core import Config, LLMClient, BrainAgent, HeartAgent, ToolManager
        print("✅ Core imports successful")
        
        config = Config()
        print(f"✅ Configuration loaded - Providers: {', '.join(config.get_available_providers())}")
        
        tool_configs = config.get_tool_configs()
        available_tools = [name for name, cfg in tool_configs.items() if cfg.get('enabled')]
        print(f"✅ Available tools: {', '.join(available_tools)}")
        
        providers = config.get_available_providers()
        if providers:
            provider = providers[0]
            models = config.get_available_models(provider)
            if models:
                llm_config = config.create_llm_config(provider, models[0], temperature=0.5)
                llm_client = LLMClient(llm_config)
                print(f"✅ LLM client created - {provider}/{models[0]}")
                
                async with llm_client:
                    response = await llm_client.generate([
                        {"role": "user", "content": "Say 'System test successful' if you can read this."}
                    ])
                    print(f"✅ LLM response: {response[:50]}...")
                
                tool_manager = ToolManager(config, llm_client)
                print(f"✅ Tool manager initialized with {len(tool_manager.get_available_tools())} tools")
                
                calc_result = await tool_manager.execute_tool("calculator", expression="2+2")
                if calc_result.get("success"):
                    print(f"✅ Calculator tool working: 2+2 = {calc_result.get('result')}")
                
                brain_agent = BrainAgent(llm_client, tool_manager)
                print("✅ Brain Agent initialized")
                
                heart_agent = HeartAgent(llm_client)
                print("✅ Heart Agent initialized")
                
                print("\n🎉 All system tests passed!")
                print("\n🚀 System ready for use!")
                print("\nNext steps:")
                print("1. streamlit run app.py")
                print("2. Open http://localhost:8501")
                
            else:
                print(f"❌ No models available for provider {provider}")
        else:
            print("❌ No LLM providers configured")
            print("Add API keys to .env file")
    
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Run: pip install -r requirements.txt")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    print("🧠❤️ Testing Brain-Heart Deep Research System")
    print("=" * 50)
    asyncio.run(test_system())