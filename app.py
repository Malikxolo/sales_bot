"""
Brain-Heart Deep Research System - Streamlit Application
FIXED VERSION - Proper model selection flow
SECURITY: Path sanitization to prevent directory traversal attacks
"""

import streamlit as st
import asyncio
import json
import time
import uuid
import os
import shutil
import threading
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import sys
import logging.config
import requests
from chroma_log_handler import ChromaLogHandler
from dotenv import load_dotenv
import aiohttp
from core.google_drive_integration import render_multi_account_drive_picker, cleanup_multi_account_session
from core.path_security import (
    sanitize_path_component,
    sanitize_filename,
    validate_safe_path,
    create_safe_user_path
)

load_dotenv()

def upload_local_files(user_id: str, collection_name: str, files):

    url = "http://localhost:8030/api/collections/create/local"
    form_data = {
        "user_id": user_id,
        "collection_name": collection_name,
    }
    files_data = [
        ("files", (uploaded_file.name, uploaded_file.read(), uploaded_file.type or "application/octet-stream"))
        for uploaded_file in files
    ]

    response = requests.post(url, data=form_data, files=files_data)
    return response.json()

async def mog_query(user_id: str, chat_history:List, query: str):
    url = "http://localhost:8030/api/chat"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    payload = {
        "userid": user_id,
        "chat_history": chat_history,
        "user_query": query
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            response_data = await response.json()
            return response_data


# Initialize user_id FIRST (before logging)
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('error.log', encoding='utf-8', errors='replace')
    ],
    force=True  # Override any existing configuration
)

# Set specific loggers
logging.getLogger('core.google_drive_integration').setLevel(logging.INFO)
logging.getLogger('core.knowledge_base').setLevel(logging.INFO)

logger = logging.getLogger(__name__)

def cleanup_expired_sessions():
    """Clean up expired sessions - REMOVES ENTIRE FOLDERS"""
    from core.session_manager import SessionTokenManager
    
    sessions_dir = "sessions"
    if not os.path.exists(sessions_dir):
        logger.info("No sessions to clean")
        return
    
    logger.info("Starting session cleanup...")
    current_time = datetime.now(timezone.utc)
    cleaned_count = 0
    
    for user_id in os.listdir(sessions_dir):
        user_path = os.path.join(sessions_dir, user_id)
        if not os.path.isdir(user_path):
            continue
        
        logger.info(f"🔍 Checking user: {user_id}")
        
        # Get all session files
        session_files = [f for f in os.listdir(user_path) if f.startswith('session_') and f.endswith('.json')]
        
        if not session_files:
            # Empty folder - remove it
            logger.info(f"🗑️ Empty user folder found: {user_id}")
            try:
                shutil.rmtree(user_path)
                cleaned_count += 1
                logger.info(f"✅ Removed empty user folder: {user_id}")
            except Exception as e:
                logger.error(f"❌ Failed to remove empty folder {user_id}: {e}")
            continue
        
        # Check each session file
        user_expired = False
        for filename in session_files:
            account_id = filename.replace('session_', '').replace('.json', '')
            session_file = os.path.join(user_path, filename)
            
            logger.info(f"🔍 Checking session: {user_id}/{account_id}")
            
            try:
                # Read session file in binary mode
                with open(session_file, 'rb') as f:
                    encrypted_data = f.read()
                
                # Try to decrypt (TTL check)
                session_mgr = SessionTokenManager(user_id, account_id)
                decrypted_data = session_mgr._decrypt_data(encrypted_data)
                
                if not decrypted_data:
                    # TTL expired - mark for deletion
                    logger.info(f"🗑️ Session TTL expired: {user_id}/{account_id}")
                    user_expired = True
                    break
                else:
                    # Session still valid - keep user
                    logger.info(f"✅ Session valid: {user_id}/{account_id}")
                    user_expired = False
                    break
            
            except Exception as e:
                logger.info(f"🗑️ Session read error (expired): {user_id}/{account_id} - {e}")
                user_expired = True
                break
        
        # Delete user folder if expired
        if user_expired:
            try:
                logger.info(f"🗑️ Removing expired user folder: {user_id}")
                
                # Try to revoke tokens first
                try:
                    for filename in session_files:
                        account_id = filename.replace('session_', '').replace('.json', '')
                        session_mgr = SessionTokenManager(user_id, account_id)
                        session_mgr.revoke_google_tokens()
                        logger.info(f"🔒 Revoked tokens: {user_id}/{account_id}")
                except Exception as revoke_error:
                    logger.warning(f"Token revocation failed (continuing): {revoke_error}")
                
                # Delete the folder
                shutil.rmtree(user_path)
                cleaned_count += 1
                logger.info(f"✅ Deleted expired user folder: {user_id}")
                
            except Exception as delete_error:
                logger.error(f"❌ Failed to delete {user_id}: {delete_error}")
        else:
            logger.info(f"✅ Keeping user: {user_id}")
    
    logger.info(f"Cleanup completed: {cleaned_count} users removed")




@st.cache_resource
def start_agent_background_worker(_agent):
    """Start agent's background memory worker - FIXED VERSION"""
    try:
        # Create a dedicated event loop for the background worker thread
        # This loop will persist and won't interfere with main thread loops
        def run_worker():
            # Create a new loop for this thread
            logger.info("Starting agent background worker thread...")
            worker_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(worker_loop)
            
            try:
                worker_loop.run_until_complete(_agent.background_task_worker())
            except Exception as e:
                logger.error(f"Background worker error: {e}")
            finally:
                # Clean up this thread's loop
                try:
                    pending = asyncio.all_tasks(worker_loop)
                    for task in pending:
                        task.cancel()
                    if pending:
                        worker_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                except Exception as cleanup_error:
                    logger.warning(f"Worker cleanup warning: {cleanup_error}")
                finally:
                    worker_loop.close()
        
        # Start the worker in a daemon thread
        thread = threading.Thread(target=run_worker, daemon=True)
        thread.start()
        logger.info("✅ Agent background worker started with isolated event loop")
        return "started"
    
    except Exception as e:
        logger.error(f"Failed to start agent worker: {e}")
        return None


def get_user_id():
    """Get or create user ID"""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
        logger.info(f"New user session: {st.session_state.user_id}")
    return st.session_state.user_id

def get_user_collections(user_id: str) -> List[str]:
    """Get user's collections"""
    try:
        user_path = f"db_collection/{user_id}"
        if os.path.exists(user_path):
            return [d for d in os.listdir(user_path) if os.path.isdir(os.path.join(user_path, d))]
        return []
    except Exception as e:
        logger.error(f"Error getting collections: {e}")
        return []
    
def display_collection_name(user_id: str, namespaced_name: str) -> str:
    """Show clean collection names to users"""
    from core.knowledge_base import get_display_collection_name
    return get_display_collection_name(user_id, namespaced_name)

def create_collection(user_id: str, collection_name: str, files: List):
    """Create new collection from uploaded files - ENHANCED VERSION WITH CLOUD CLEANUP"""
    try:
        from core.knowledge_base import kb_manager
        from core.knowledge_base import create_knowledge_base
        
        # SECURITY: Sanitize user inputs
        safe_user_id = sanitize_path_component(user_id)
        safe_collection_name = sanitize_path_component(collection_name)
        
        # Create user directory with secure path
        user_path = create_safe_user_path("db_collection", safe_user_id)
        os.makedirs(user_path, exist_ok=True)
        
        # STEP 1: Delete ALL existing Chroma Cloud collections for this user
        try:
            client = kb_manager._get_chroma_client()
            all_collections = client.list_collections()
            user_prefix = f"{safe_user_id}_"
            
            for collection in all_collections:
                if collection.name.startswith(user_prefix):
                    client.delete_collection(name=collection.name)
                    logger.info(f"Deleted remote collection: {collection.name}")
        except Exception as e:
            logger.warning(f"Could not clean up remote collections: {e}")
        
        # STEP 2: Remove existing local collections 
        for existing in os.listdir(user_path):
            existing_path = validate_safe_path(user_path, existing)
            if os.path.isdir(existing_path):
                shutil.rmtree(existing_path)
                logger.info(f"Removed local collection: {existing}")
        
        # STEP 3: Create new collection directory and save files with secure paths
        collection_path = create_safe_user_path("db_collection", safe_user_id, safe_collection_name)
        os.makedirs(collection_path, exist_ok=True)
        
        # Save uploaded files with sanitized filenames
        file_paths = []
        for file in files:
            # SECURITY: Sanitize uploaded filename
            safe_filename = sanitize_filename(file.name)
            file_path = validate_safe_path(collection_path, safe_filename)
            
            with open(file_path, "wb") as f:
                f.write(file.getvalue())
            file_paths.append(file_path)
        
        logger.info(f"Saved {len(files)} files for processing")
        
        # STEP 4: Create vector database using your knowledge_base.py logic
        result = create_knowledge_base(safe_user_id, safe_collection_name, file_paths)
        
        if result["success"]:
            logger.info(f"Knowledge base created successfully: {safe_collection_name}")
            return True
        else:
            logger.error(f"Knowledge base creation failed: {result.get('error', 'Unknown error')}")
            return False
    
    except ValueError as ve:
        # Security validation error
        logger.error(f"Security validation failed: {ve}")
        st.error(f"Invalid input: {str(ve)}")
        return False
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        return False

def render_rag_sidebar():
    """Render RAG UI in sidebar - ONLY ADDITION"""
    user_id = get_user_id()
    collections = get_user_collections(user_id)
    
    st.markdown("### 📚 RAG Configuration")
    
    if collections:
        collection_name = collections[0]
        clean_name = display_collection_name(user_id, collection_name)
        st.success(f"Active: {clean_name}")
        
        # Show file count
        try:
            # SECURITY: Use sanitized paths
            safe_user_id = sanitize_path_component(user_id)
            safe_collection_name = sanitize_path_component(collection_name)
            
            metadata_path = validate_safe_path(
                "db_collection",
                safe_user_id,
                safe_collection_name,
                "metadata.json"
            )
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                st.info(f"Files: {metadata.get('file_count', 0)}")
        except ValueError as ve:
            logger.error(f"Security validation failed: {ve}")
        except Exception as e:
            logger.error(f"Error reading metadata: {e}")
            
        if st.button("✏️ Edit Collection"):
            st.session_state.show_edit = True
            st.session_state.show_upload = False
            
        if st.button("🔄 Replace Collection"):
            st.session_state.show_upload = True
            st.session_state.show_edit = False
    else:
        st.info("No collection loaded")
        st.warning("RAG tool disabled")
        if st.button("📁 Create Collection"):
            st.session_state.show_upload = True
            st.session_state.show_edit = False




# Import core system
try:
    from core import (
        Config, LLMClient, OptimizedAgent, 
        ToolManager, BrainHeartException
    )
    SYSTEM_AVAILABLE = True
except ImportError as e:
    st.error(f"Core system import failed: {e}")
    st.markdown("""
    **Fix Dependencies:**
    ```
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


async def create_agents_async(config, brain_agent_config, heart_agent_config, web_model_config, use_premium_search=False):
    """Create Optimized Agent with selected models"""
    try:
        # Create LLM clients (use same config for both brain and heart)
        brain_llm = LLMClient(brain_agent_config)
        heart_llm = LLMClient(heart_agent_config)
        
        # Create tool manager
        tool_manager = ToolManager(config, brain_llm, web_model_config, use_premium_search)
        
        # Create optimized agent (uses same LLM for both brain and heart functions)
        optimized_agent = OptimizedAgent(brain_llm, heart_llm, tool_manager)
        
        return {
            "optimized_agent": optimized_agent,
            "tool_manager": tool_manager,
            "status": "success"
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}



def display_model_selector(config, agent_name: str, key_prefix: str):
    """Display model selection interface with unique keys"""
    
    providers = config.get_available_providers()
    
    col1, col2 = st.columns([2, 2])
    
    with col1:
        provider = st.selectbox(
            f"{agent_name} Provider:",
            providers,
            key=f"{key_prefix}_provider"
        )
    
    with col2:
        models = config.get_available_models(provider)
        model = st.selectbox(
            f"{agent_name} Model:",
            models,
            key=f"{key_prefix}_model"
        )
    
    return config.create_llm_config(provider, model)


def display_web_model_selector(config, agent_name: str):
    """Display web model selection interface"""
    
    col1 = st.columns([2])[0]
    
    with col1:
        models = config.get_available_web_models()
        model = st.selectbox(
            f"{agent_name} Model:",
            models,
            key=f"{agent_name}_model"
        )
    
    return model


async def process_query_real(query: str, optimized_agent, style: str, user_id: str = None, chat_history: List = None) -> Dict[str, Any]:
    """Process query through Optimized Agent"""
    
    try:
        start_time = time.time()
        
        # Single agent processing
        st.info("🚀 Optimized Agent processing query...")
        result = await mog_query(user_id, chat_history, query)
        
        total_time = time.time() - start_time
        
        if result.get("success"):
            result["total_time"] = total_time
            return result
        else:
            return {
                "success": False,
                "error": result.get("error", "Processing failed")
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"System processing failed: {str(e)}"
        }


def display_real_results(result: Dict[str, Any], query: str):
    """Display results from Optimized Agent"""
    
    if not result.get("success"):
        st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
        with st.expander("🔍 Debug Information"):
            st.json(result)
        return
    
    # Main response from Optimized Agent
    st.markdown("## 💎 Final Response")
    response = result.get("response", "No response generated")
    if response:
        st.markdown(response)
    else:
        st.warning("No response generated by Optimized Agent")
    
    # Performance metrics
    processing_time = result.get("processing_time", {})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("⚡ Analysis Time", f"{processing_time.get('analysis', 0):.2f}s")
    with col2:
        st.metric("⏱️ Total Time", f"{processing_time.get('total', 0):.2f}s")
    with col3:
        tools_used = result.get("tools_used", [])
        st.metric("🛠️ Tools Used", len(tools_used))
    
    # Detailed analysis tabs
    tab1, tab2, tab3 = st.tabs(["🚀 Analysis", "🛠️ Tool Results", "🔍 Raw Data"])
    
    with tab1:
        st.markdown("### 🚀 Optimized Agent Analysis")
        
        analysis = result.get("analysis", {})
        if analysis:
            st.markdown("**Semantic Intent:**")
            st.markdown(f"- {analysis.get('semantic_intent', 'Unknown')}")
            
            business_opp = analysis.get('business_opportunity', {})
            if business_opp.get('detected'):
                st.markdown("**Business Opportunity Detected:**")
                st.markdown(f"- Score: {business_opp.get('composite_confidence', 0)}/100")
                st.markdown(f"- Pain Points: {', '.join(business_opp.get('pain_points', []))}")
            
            if tools_used:
                st.markdown("**Tools Selected:**")
                for tool in tools_used:
                    st.markdown(f"- {tool}")
    
    with tab2:
        st.markdown("### 🛠️ Tool Execution Results")
        
        tool_results = result.get("tool_results", {})
        if tool_results:
            for tool_name, tool_result in tool_results.items():
                with st.expander(f"🔧 {tool_name.title()} Results"):
                    if isinstance(tool_result, dict) and 'error' not in tool_result:
                        # Handle different tool result formats
                        if 'results' in tool_result and isinstance(tool_result['results'], list):
                            # Web search results
                            st.success(f"Found {len(tool_result['results'])} results")
                            for i, search_result in enumerate(tool_result['results'][:3]):
                                st.markdown(f"**{i+1}. {search_result.get('title', 'No title')}**")
                                st.markdown(f"Link: {search_result.get('link', 'No link')}")
                                st.markdown(f"   {search_result.get('snippet', 'No snippet')}")
                        elif 'retrieved' in tool_result:
                            # RAG results
                            st.success("Knowledge base retrieved")
                            st.markdown(tool_result['retrieved'][:300] + "..." if len(tool_result['retrieved']) > 300 else tool_result['retrieved'])
                        else:
                            # Generic result
                            st.json(tool_result)
                    else:
                        st.error(f"Tool error: {tool_result.get('error', 'Unknown error')}")
        else:
            st.info("No tools were used for this query")
    
    with tab3:
        st.markdown("### 🔍 Complete Raw Data")
        st.json(result)



def main():
    """Main Streamlit application - REAL Brain-Heart system"""
    
    if 'app_initialized' not in st.session_state:
        # start_background_cleanup()
        st.session_state.app_initialized = True
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        logger.info(f"Initialized chat history for user: {get_user_id()}")
    
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
        if st.button("🗑️ Clear Chat History"):
            st.session_state['chat_history'] = []  
            st.success("Chat cleared!")
            st.rerun()
            
        st.markdown("## 🎛️ Model Configuration")
        
        available_providers = config.get_available_providers()
        st.success(f"✅ Available: {', '.join(available_providers)}")
        
        # 🆕 Brain Agent Configuration
        st.markdown("### 🧠 Brain Agent")
        st.caption("Analysis, planning, and orchestration")
        brain_model_config = display_model_selector(config, "Brain", "brain")
        
        st.markdown("---")
        
        # 🆕 Heart Agent Configuration
        st.markdown("### ❤️ Heart Agent")
        st.caption("Response synthesis and communication")
        heart_model_config = display_model_selector(config, "Heart", "heart")

        st.markdown("---")

        # Web Agent Configuration  
        st.markdown("### 🌐 Web Search Agent")
        st.caption("Handles web search operations")
        # Premium search toggle
        use_premium_search = st.checkbox("🚀 Enable Premium Search (Perplexity)", value=False)
        
        # Show model selector only if premium enabled
        if use_premium_search:
            web_model_config = display_web_model_selector(config, "Web")
        else:
            web_model_config = None  # Will use Serper/ValueSerp
        
        st.markdown("---")
        st.markdown("### 🧪 Debug Cleanup")
        if st.button("🧪 Manual Cleanup Test"):
            logger.info("🧪 MANUAL TEST: Starting cleanup...")
            cleanup_expired_sessions()
            logger.info("🧪 MANUAL TEST: Cleanup completed")
            st.success("Check terminal logs!")

        if st.button("📂 Show Sessions"):
            if os.path.exists("sessions"):
                for user_dir in os.listdir("sessions"):
                    st.write(f"👤 {user_dir}")
                    user_path = os.path.join("sessions", user_dir)
                    if os.path.isdir(user_path):
                        for f in os.listdir(user_path):
                            st.write(f"  📄 {f}")
                # Communication style
        st.markdown("### 💬 Response Style")
        style = st.selectbox(
            "Heart Agent Communication:",
            ["professional", "executive", "technical", "creative"],
            help="How the Heart Agent presents final response"
        )
        
        # RAG Configuration - RIGHT BELOW RESPONSE STYLE - ONLY ADDITION
        render_rag_sidebar()
        
        st.markdown("---")  # ONLY ADDITION
        
        # Tool status
        st.markdown("### 🛠️ Available Tools")
        tool_configs = config.get_tool_configs(web_model=web_model_config, use_premium_search=use_premium_search)
        for tool_name, tool_config in tool_configs.items():
            status = "✅" if tool_config.get("enabled", False) else "❌"
            st.markdown(f"{status} {tool_name}")
            if tool_name == "web_search" and tool_config.get("enabled"):
                st.caption(f"   Model: {web_model_config}")
    
    # FILE UPLOAD INTERFACE - ONLY ADDITION  
    if st.session_state.get('show_upload', False):
        st.markdown("---")
        st.markdown("## 📤 Upload Documents for RAG")
        
        collection_name = st.text_input("Collection Name:", value="")
        
        # File source tabs
        tab1, tab2 = st.tabs(["💻 Local Files", "📁 Google Drive"])
        
        file_paths = None
        
        with tab1:
            uploaded_files = st.file_uploader(
                "Upload Files:",
                type=[
                    'txt', 'md', 'rtf', 'pdf', 'doc', 'docx', 'docm', 'dot', 'dotx', 'dotm', 'odt',
                    'csv', 'tsv', 'json', 'xlsx', 'xls', 'xlsm', 'xlsb', 'xltx', 'xltm', 'ods',
                    'ppt', 'pptx', 'pptm', 'potx', 'potm', 'ppsx', 'ppsm', 'odp',
                    'mdb', 'accdb', 'html', 'htm', 'xml', 'msg', 'eml', 'epub'
                ],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                file_paths = "local_files"  # Flag for local files
        
        with tab2:
            # Multi-account Google Drive picker with NO premature download
            selected_files = render_multi_account_drive_picker(get_user_id(), download_files=False)
            
            if selected_files and len(selected_files) > 0:
                file_paths = selected_files  # File metadata, not actual files
                
                # Show selected files summary (NO download yet)
                st.success(f"✅ Selected {len(selected_files)} files from Google Drive")
                
                # Show file details
                for file_info in selected_files:
                    st.write(f"📄 {file_info['name']} ({file_info.get('account_alias', 'Unknown account')})")
                
                # Group files by account using metadata
                account_summary = {}
                for file_info in selected_files:
                    account_id = file_info.get('account_id', 'unknown')
                    account_summary[account_id] = account_summary.get(account_id, 0) + 1
                
                if len(account_summary) > 1:
                    st.info(f"📊 Files from {len(account_summary)} accounts: " + 
                        ", ".join([f"{acc}: {count}" for acc, count in account_summary.items()]))
            else:
                file_paths = None

        
        # Create collection button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Create Collection", key="upload_local_files"):
                if not collection_name.strip():
                    st.error("❌ Please enter collection name")
                else:
                    # ✅ ROBUST: Check files based on what's actually available
                    has_local_files = uploaded_files and len(uploaded_files) > 0
                    has_drive_files = file_paths and file_paths != "local_files" and len(file_paths) > 0
                    
                    if not has_local_files and not has_drive_files:
                        st.error("❌ Please select files from either Local Files or Google Drive tab")
                    elif has_local_files and has_drive_files:
                        st.error("❌ Please select files from only ONE tab (Local OR Google Drive)")
                    else:
                        
                        start_time = time.time()
                        
                        with st.spinner("🔄 Creating knowledge base..."):
                            # Handle local vs Drive files
                            if has_local_files:
                                # Local files - existing logic
                                st.info("📁 Processing local files...")
                                res = upload_local_files(get_user_id(), collection_name, uploaded_files)
                                if res:
                                    elapsed = time.time() - start_time
                                    st.success(f"✅ Collection created in {elapsed:.1f}s!")
                                    st.success("✅ Collection created from local files!")
                                    st.session_state.show_upload = False
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to create collection")
                            
                            elif has_drive_files:
                                # Google Drive files - DOWNLOAD ONLY WHEN CREATING
                                st.info("📥 Processing Google Drive files...")
                                downloaded_files = []
                                success = False
                                
                                try:
                                    # STEP 1: Download selected files NOW
                                    st.info("📥 Downloading selected files...")
                                    from core.google_drive_integration import MultiAccountGoogleDriveManager
                                    drive_manager = MultiAccountGoogleDriveManager(get_user_id())
                                    
                                    downloaded_files = drive_manager.download_files_with_conflict_resolution(file_paths)
                                    
                                    if not downloaded_files:
                                        st.error("❌ Failed to download files")
                                    else:
                                        st.success(f"✅ Downloaded {len(downloaded_files)} files")
                                        
                                        # STEP 2: Create knowledge base
                                        st.info("🔄 Creating knowledge base...")
                                        from core.knowledge_base import create_knowledge_base
                                        
                                        # Detect if multi-account
                                        has_multiple_accounts = len(set(f.get('account_id') for f in file_paths)) > 1
                                        
                                        if has_multiple_accounts:
                                            result = create_knowledge_base(
                                                get_user_id(), 
                                                collection_name, 
                                                downloaded_files, 
                                                account_info="multi_account"
                                            )
                                        else:
                                            result = create_knowledge_base(get_user_id(), collection_name, downloaded_files)
                                        
                                        if result["success"]:
                                            elapsed = time.time() - start_time 
                                            st.success(f"✅ Collection created in {elapsed:.1f}s!")
                                            st.success("✅ Collection created from Google Drive!")
                                            success = True
                                        else:
                                            st.error(f"❌ Failed to create collection: {result.get('error')}")
                                
                                except Exception as e:
                                    st.error(f"❌ Error during collection creation: {e}")
                                    logger.error(f"Collection creation error: {e}")
                                
                                finally:
                                    # STEP 3: ALWAYS CLEANUP
                                    st.info("🧹 Cleaning up...")
                                    try:
                                        # Clean up downloaded files
                                        for file_path in downloaded_files:
                                            if os.path.exists(file_path):
                                                os.remove(file_path)
                                        
                                        # Revoke Google sessions for security
                                        cleanup_multi_account_session(get_user_id(), keep_connection=True)
                                        
                                    except Exception as cleanup_error:
                                        logger.error(f"Cleanup error: {cleanup_error}")
                                    
                                    # If successful, close upload UI
                                    if success:
                                        st.session_state.show_upload = False
                                        st.rerun()

        with col2:
            if st.button("❌ Cancel", key="upload_cancel"):
                # AUTOMATIC REVOKE FOR SECURITY (Fixed)
                from core.google_drive_integration import MultiAccountGoogleDriveManager
                drive_manager = MultiAccountGoogleDriveManager(get_user_id())
                
                # Always revoke for security when cancelling
                with st.spinner("🔒 Cleaning up and revoking access..."):
                    revoked = drive_manager.security_disconnect_all()
                    if revoked > 0:
                        st.success(f"✅ Revoked {revoked} Google accounts for security")
                    else:
                        # Fallback cleanup if no accounts to revoke
                        cleanup_multi_account_session(get_user_id(), keep_connection=False)
                
                st.session_state.show_upload = False
                st.rerun()
    # EDIT COLLECTION INTERFACE
    if st.session_state.get('show_edit', False):
        st.markdown("---")
        st.markdown("## ✏️ Edit Collection")
        
        user_id = get_user_id()
        collections = get_user_collections(user_id)
        
        if collections:
            collection_name = collections[0]
            
            # Get metadata
            try:
                metadata_path = f"db_collection/{user_id}/{collection_name}/knowledge_base_metadata.json"
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                file_names = metadata.get('file_names', [])
                
                if file_names:
                    st.markdown(f"### 📄 Current Files ({len(file_names)})")
                    
                    # Display files with delete buttons
                    for filename in file_names:
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.write(f"📄 {filename}")
                        with col2:
                            if st.button("🗑️", key=f"delete_{filename}"):
                                with st.spinner(f"Deleting {filename}..."):
                                    from core.knowledge_base import delete_file
                                    result = delete_file(user_id, collection_name, filename)
                                    
                                    if result["success"]:
                                        st.success(f"✅ Deleted {filename}")
                                        st.rerun()
                                    else:
                                        st.error(f"❌ Failed: {result.get('error')}")
                else:
                    st.warning("No files in collection")
                
            except Exception as e:
                st.error(f"Error loading files: {e}")
            
            st.markdown("---")
            st.markdown("### ➕ Add New Files")
            
            # Add files uploader
            new_files = st.file_uploader(
                "Upload files to add:",
                type=[
                    'txt', 'md', 'rtf', 'pdf', 'doc', 'docx', 'docm', 'dot', 'dotx', 'dotm', 'odt',
                    'csv', 'tsv', 'json', 'xlsx', 'xls', 'xlsm', 'xlsb', 'xltx', 'xltm', 'ods',
                    'ppt', 'pptx', 'pptm', 'potx', 'potm', 'ppsx', 'ppsm', 'odp',
                    'mdb', 'accdb', 'html', 'htm', 'xml', 'msg', 'eml', 'epub'
                ],
                accept_multiple_files=True,
                key="edit_file_uploader"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("➕ Add Files") and new_files:
                    with st.spinner(f"Adding {len(new_files)} files..."):
                        # Save uploaded files temporarily
                        temp_paths = []
                        for file in new_files:
                            temp_path = file.name
                            with open(temp_path, "wb") as f:
                                f.write(file.getvalue())
                            temp_paths.append(temp_path)
                        
                        try:
                            from core.knowledge_base import add_files
                            result = add_files(user_id, collection_name, temp_paths)
                            
                            if result["success"]:
                                st.success(f"✅ Added {len(new_files)} files!")
                                st.rerun()
                            else:
                                st.error(f"❌ Failed: {result.get('error')}")
                        
                        finally:
                            # Cleanup temp files
                            for temp_path in temp_paths:
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
            
            with col2:
                if st.button("✅ Done"):
                    st.session_state.show_edit = False
                    st.rerun()
        
        else:
            st.error("No collection found")
            if st.button("❌ Close"):
                st.session_state.show_edit = False
                st.rerun()
                

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
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"🧠 Brain: {brain_model_config.provider}/{brain_model_config.model}")
        with col2:
            st.info(f"❤️ Heart: {heart_model_config.provider}/{heart_model_config.model}")
        with col3:
            search_type = f"Premium: {web_model_config}" if use_premium_search else "Standard: ScrapingDog/ValueSerp"
            st.info(f"🌐 Web: {search_type}")

        # Create agents and process
        with st.spinner("Creating agents and processing query..."):
            try:
                # Run async agent creation and processing
                agents_result = asyncio.run(create_agents_async(config, brain_model_config, heart_model_config, web_model_config, use_premium_search))
                
                if agents_result["status"] == "error":
                    st.error(f"❌ Agent creation failed: {agents_result['error']}")
                    logger.error(f"Agent creation failed: {agents_result}")
                    if show_debug:
                        st.json(agents_result)
                else:
                    # Process query with optimized agent
                    
                    if not st.session_state.get("agent_worker_started", False):
                        try:
                            # Call the cached start function with the actual agent
                            start_agent_background_worker(agents_result["optimized_agent"])
                            st.session_state["agent_worker_started"] = True
                            logger.info("Agent background worker requested (st.session_state set).")
                        except Exception as e:
                            logger.error(f"Failed to start background worker: {e}")
                            # Optionally show to user
                            if show_debug:
                                st.warning(f"Background worker start error: {e}")
                                
                    result = asyncio.run(process_query_real(
                        query, 
                        agents_result["optimized_agent"], 
                        style,
                        user_id=get_user_id(),
                        chat_history=st.session_state.chat_history
                    ))
                    
                    if result.get("success"):
                        # Update chat history
                        st.session_state.chat_history.extend([
                            {"role": "user", "content": query},
                            {"role": "assistant", "content": result.get("response", "")}
                        ])
                        logger.info(f"Updated chat history. Total messages: {len(st.session_state.chat_history)}")
                    
                    # Display real results
                    display_real_results(result, query)
                    
            except Exception as e:
                st.error(f"❌ Processing failed: {str(e)}")
                logger.error(f"Processing failed: {str(e)}")  # ONLY ADDITION
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
