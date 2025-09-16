"""
Brain-Heart Deep Research System - Streamlit Application
REAL VERSION - Pure LLM-driven with actual agents and tools
NO HARDCODING - Everything dynamically decided by LLMs
"""

import streamlit as st
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional

# Import core system
try:
    from core import (
        Config, LLMClient, BrainAgent, HeartAgent, 
        ToolManager, BrainHeartException
    )
    SYSTEM_AVAILABLE = True
except ImportError as e:
    st.error(f"Core system import failed: {e}")
    st.markdown("""
    **Fix Dependencies:**
    ```bash
    python fix_dependencies.py
    # OR manually:
    pip install aiohttp python-dotenv streamlit requests pandas numpy
    ```
    """)
    SYSTEM_AVAILABLE = False

if not SYSTEM_AVAILABLE:
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Brain-Heart Deep Research System",
    page_icon="🧠❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def initialize_config():
    """Initialize configuration"""
    try:
        config = Config()
        return {"config": config, "status": "success"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

async def create_agents_async(config, brain_model_config, heart_model_config):
    """Create Brain and Heart agents with selected models"""
    try:
        # Create LLM clients
        brain_llm = LLMClient(brain_model_config)
        heart_llm = LLMClient(heart_model_config)
        
        # Create tool manager
        tool_manager = ToolManager(config, brain_llm)
        
        # Create agents
        brain_agent = BrainAgent(brain_llm, tool_manager)
        heart_agent = HeartAgent(heart_llm)
        
        return {
            "brain_agent": brain_agent,
            "heart_agent": heart_agent,
            "tool_manager": tool_manager,
            "status": "success"
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

def display_model_selector(config, agent_name: str, default_temp: float):
    """Display model selection interface"""
    
    providers = config.get_available_providers()
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        provider = st.selectbox(
            f"{agent_name} Provider:",
            providers,
            key=f"{agent_name}_provider"
        )
    
    with col2:
        models = config.get_available_models(provider)
        model = st.selectbox(
            f"{agent_name} Model:",
            models,
            key=f"{agent_name}_model"
        )
    
    with col3:
        temperature = st.slider(
            f"{agent_name} Temp:",
            0.0, 1.0, default_temp,
            key=f"{agent_name}_temp",
            help="Higher = more creative"
        )
    
    return config.create_llm_config(provider, model, temperature)

async def process_query_real(query: str, brain_agent, heart_agent, style: str) -> Dict[str, Any]:
    """Process query through REAL Brain-Heart system - NO HARDCODING"""
    
    try:
        start_time = time.time()
        
        # Phase 1: Brain Agent Processing (REAL LLM-driven orchestration)
        st.info("🧠 Brain Agent analyzing query and selecting tools...")
        brain_result = await brain_agent.process_query(query)
        
        brain_time = time.time() - start_time
        
        # Phase 2: Heart Agent Synthesis (REAL LLM-driven synthesis)
        if brain_result.get("success"):
            st.info("❤️ Heart Agent synthesizing optimal response...")
            heart_result = await heart_agent.synthesize_response(
                brain_result, query, style
            )
            
            total_time = time.time() - start_time
            
            return {
                "success": True,
                "brain_result": brain_result,
                "heart_result": heart_result,
                "brain_time": brain_time,
                "total_time": total_time
            }
        else:
            return {
                "success": False,
                "error": brain_result.get("error", "Brain processing failed"),
                "brain_result": brain_result
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"System processing failed: {str(e)}"
        }

def display_real_results(result: Dict[str, Any], query: str):
    """Display REAL results from Brain-Heart processing"""
    
    if not result.get("success"):
        st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
        with st.expander("🔍 Debug Information"):
            st.json(result)
        return
    
    brain_result = result["brain_result"]
    heart_result = result["heart_result"]
    
    # Main response from Heart Agent (REAL LLM synthesis)
    st.markdown("## 💎 Final Response")
    st.markdown(heart_result.get("response", "No response generated"))
    
    # Performance metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🧠 Brain Time", f"{result.get('brain_time', 0):.2f}s")
    with col2:
        st.metric("⏱️ Total Time", f"{result.get('total_time', 0):.2f}s")
    with col3:
        tools_used = brain_result.get("tools_used", [])
        st.metric("🛠️ Tools Used", len(tools_used))
    
    # Detailed analysis tabs
    tab1, tab2, tab3 = st.tabs(["🧠 Brain Analysis", "❤️ Heart Synthesis", "🔍 Raw Data"])
    
    with tab1:
        st.markdown("### 🧠 Brain Agent Execution Details")
        
        plan = brain_result.get("plan", {})
        if plan:
            st.markdown("**LLM-Generated Plan:**")
            st.markdown(f"- **Approach:** {plan.get('approach', 'Unknown')}")
            st.markdown(f"- **Reasoning:** {plan.get('reasoning', 'Not provided')}")
            
            if tools_used:
                st.markdown("**Tools Selected by Brain Agent:**")
                for tool in tools_used:
                    st.markdown(f"- {tool}")
        
        # Show actual execution results
        execution_results = brain_result.get("execution_results", {})
        if execution_results:
            st.markdown("**Actual Tool Execution Results:**")
            for step_key, step_result in execution_results.items():
                with st.expander(f"Step {step_key.split('_')[1]} - {step_result.get('tool', 'Unknown')}"):
                    if step_result.get('result'):
                        result_data = step_result['result']
                        if isinstance(result_data, dict):
                            # Show formatted results for specific tools
                            if result_data.get('tool_name') == 'calculator':
                                if result_data.get('success'):
                                    st.success(f"Calculation Result: {result_data.get('formatted_result', result_data.get('result'))}")
                                else:
                                    st.error(f"Calculation Error: {result_data.get('error')}")
                            elif result_data.get('tool_name') == 'web_search':
                                if result_data.get('success'):
                                    st.success(f"Found {result_data.get('total_results', 0)} results")
                                    for i, search_result in enumerate(result_data.get('results', [])):
                                        st.markdown(f"**{i+1}. {search_result.get('title')}**")
                                        st.markdown(f"Link:  {search_result.get('link')}")
                                        st.markdown(f"   {search_result.get('snippet')}")
                                else:
                                    st.error(f"Search Error: {result_data.get('error')}")
                            else:
                                st.json(result_data)
                        else:
                            st.write(result_data)
                    else:
                        st.json(step_result)
    
    with tab2:
        st.markdown("### ❤️ Heart Agent Synthesis Details")
        
        approach = heart_result.get("synthesis_approach", {})
        if approach:
            st.markdown("**LLM-Determined Synthesis Strategy:**")
            st.markdown(f"- **Approach Type:** {approach.get('approach_type', 'Unknown')}")
            st.markdown(f"- **Tone:** {approach.get('tone', 'Professional')}")
            
            if approach.get("key_messages"):
                st.markdown("**Key Messages Identified:**")
                for msg in approach["key_messages"]:
                    st.markdown(f"- {msg}")
            
            if approach.get("structure"):
                st.markdown("**Response Structure:**")
                for section in approach["structure"]:
                    st.markdown(f"- {section}")
    
    with tab3:
        st.markdown("### 🔍 Complete Raw Data")
        st.markdown("**Full Brain Result:**")
        st.json(brain_result)
        st.markdown("**Full Heart Result:**")
        st.json(heart_result)

def main():
    """Main Streamlit application - REAL Brain-Heart system"""
    
    # Header
    st.markdown("# 🧠❤️ Brain-Heart Deep Research System")
    st.markdown("### Pure LLM Architecture - Agents Think, Tools Execute")
    
    # Initialize configuration
    config_result = initialize_config()
    
    if config_result["status"] == "error":
        st.error(f"❌ Configuration failed: {config_result['error']}")
        st.markdown("""
        **Setup Required:**
        1. Run: `python fix_dependencies.py`
        2. Copy `.env.example` to `.env`
        3. Add at least one LLM provider API key
        4. Restart the application
        """)
        st.stop()
    
    config = config_result["config"]
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("## 🎛️ Model Configuration")
        
        # Show available providers
        available_providers = config.get_available_providers()
        st.success(f"✅ Available: {', '.join(available_providers)}")
        
        # Brain Agent Configuration
        st.markdown("### 🧠 Brain Agent (Orchestrator)")
        st.caption("Analyzes queries and selects tools")
        brain_model_config = display_model_selector(config, "Brain", 0.3)
        
        st.markdown("---")
        
        # Heart Agent Configuration
        st.markdown("### ❤️ Heart Agent (Synthesizer)")
        st.caption("Creates final user-focused response")
        heart_model_config = display_model_selector(config, "Heart", 0.7)
        
        st.markdown("---")
        
        # Communication style
        st.markdown("### 💬 Response Style")
        style = st.selectbox(
            "Heart Agent Communication:",
            ["professional", "executive", "technical", "creative"],
            help="How the Heart Agent presents final response"
        )
        
        # Tool status
        st.markdown("### 🛠️ Available Tools")
        tool_configs = config.get_tool_configs()
        for tool_name, tool_config in tool_configs.items():
            status = "✅" if tool_config.get("enabled", False) else "❌"
            st.markdown(f"{status} {tool_name}")
    
    # Main interface
    st.markdown("## 🎯 Research Query")
    st.caption("The Brain Agent will analyze your query and dynamically select appropriate tools")
    
    # Query input
    query = st.text_area(
        "Enter your research query:",
        height=120,
        placeholder="Examples:\n• Calculate compound interest on $50,000 at 6% for 10 years\n• Research renewable energy market trends\n• Analyze competitive landscape for AI startups"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        submit_button = st.button("🚀 Start Brain-Heart Processing", type="primary", use_container_width=True)
    with col2:
        if st.button("🔄 Clear Results", use_container_width=True):
            st.rerun()
    with col3:
        show_debug = st.checkbox("Debug Mode")
    
    # Process query with REAL agents
    if submit_button and query.strip():
        # Display model information
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"🧠 Brain: {brain_model_config.provider}/{brain_model_config.model} (T={brain_model_config.temperature})")
        with col2:
            st.info(f"❤️ Heart: {heart_model_config.provider}/{heart_model_config.model} (T={heart_model_config.temperature})")
        
        # Create agents and process
        with st.spinner("Creating agents and processing query..."):
            try:
                # Run async agent creation and processing
                agents_result = asyncio.run(create_agents_async(config, brain_model_config, heart_model_config))
                
                if agents_result["status"] == "error":
                    st.error(f"❌ Agent creation failed: {agents_result['error']}")
                    if show_debug:
                        st.json(agents_result)
                else:
                    # Process query with real agents
                    result = asyncio.run(process_query_real(
                        query, 
                        agents_result["brain_agent"], 
                        agents_result["heart_agent"], 
                        style
                    ))
                    
                    # Display real results
                    display_real_results(result, query)
                    
            except Exception as e:
                st.error(f"❌ Processing failed: {str(e)}")
                if show_debug:
                    import traceback
                    st.code(traceback.format_exc())
    
    elif submit_button:
        st.warning("⚠️ Please enter a research query.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
        <strong>Brain-Heart Deep Research System v2.0</strong><br>
        🧠 Pure LLM Orchestration • ❤️ Dynamic Synthesis • 🛠️ Real Tool Execution
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()