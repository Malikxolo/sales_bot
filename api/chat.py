from fastapi import FastAPI, APIRouter, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse
from fastapi import Body
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from redis.asyncio import Redis
from contextlib import asynccontextmanager
from langchain_core.messages import HumanMessage, AIMessage
import logging
router = APIRouter()
import time
import os
from core import (
    LLMClient, HeartAgent, 
    ToolManager, Config, BrainAgent
)
from core.optimized_agent import OptimizedAgent
from core.logging_security import (
    safe_log_response,
    safe_log_user_data,
    safe_log_error,
    safe_log_query
)
import json
import shutil
import asyncio

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    logging.info("⚡ Starting app lifespan...")
    
    global optimizedAgent
    config = Config()

    redis_client = Redis(host=os.getenv('REDIS_HOST'), port=os.getenv('REDIS_PORT'), decode_responses=os.getenv('REDIS_DECODE_RESPONSES'), username=os.getenv('REDIS_USERNAME'), password=os.getenv('REDIS_PASSWORD'))
    await FastAPILimiter.init(redis_client)
    brain_model_config = config.create_llm_config(
        provider=settings.brain_provider,
        model=settings.brain_model,
        max_tokens=1000
    )
    heart_model_config = config.create_llm_config(
        provider=settings.heart_provider,
        model=settings.heart_model,
        max_tokens=1000
    )
    web_model_config = config.get_tool_configs(
        web_model=settings.web_model,
        use_premium_search=settings.use_premium_search
    )

    brain_llm = LLMClient(brain_model_config)
    heart_llm = LLMClient(heart_model_config)
    tool_manager = ToolManager(config, brain_llm, web_model_config, settings.use_premium_search)

    optimizedAgent = OptimizedAgent(brain_llm, heart_llm, tool_manager)

    
    optimizedAgent.worker_task = asyncio.create_task(
        optimizedAgent.background_task_worker()
    )
    logging.info("✅ OptimizedAgent background worker started")

    try:
        yield
    finally:
        
        logging.info("⚡ Shutting down app lifespan...")
        optimizedAgent.worker_task.cancel()
        try:
            await optimizedAgent.worker_task
        except asyncio.CancelledError:
            logging.info("✅ OptimizedAgent worker cancelled cleanly")


from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class QueryMessage(BaseModel):
    role: str
    content: str

class UserQuery(BaseModel):
    messages: List[QueryMessage]

class ChatMessage(BaseModel):
    userid: str
    chat_history: list[dict] = []
    user_query: str
    
from .global_config import settings

class UpdateAgentsRequest(BaseModel):
    brain_provider: Optional[str]
    brain_model: Optional[str]
    heart_provider: Optional[str]
    heart_model: Optional[str]
    use_premium_search: Optional[bool]
    web_model: Optional[str]


def list_files_in_directory(directory: str) -> List[str]:
    """List all files in a given directory"""
    if not os.path.exists(directory):
        return []
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    
def get_collection_metadata(user_id: str, collection_name: str):
    metadata_path = f"db_collection/{user_id}/{collection_name}/knowledge_base_metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return json.load(f)
    return {"file_count": 0}

def list_user_collections(user_id: str):
    base_path = f"db_collection/{user_id}"
    if not os.path.exists(base_path):
        return []
    return [c for c in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, c))]

def get_users():
    base_path = "db_collection"
    if not os.path.exists(base_path):
        return []
    return [u for u in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, u))]
    
    
    
    
async def create_agents_async(config, brain_model_config, heart_model_config, web_model_config, use_premium_search=False):
    """Create Brain and Heart agents with selected models"""
    try:
        
        brain_llm = LLMClient(brain_model_config)
        heart_llm = LLMClient(heart_model_config)
        
        tool_manager = ToolManager(config, brain_llm, web_model_config, use_premium_search)
        
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
    
async def process_query_real(query: str, brain_agent, heart_agent, style: str, user_id: str = None) -> Dict[str, Any]:
    """Process query through REAL Brain-Heart system - NO HARDCODING"""
    
    try:
        start_time = time.time()
        
        # Phase 1: Brain Agent Processing (REAL LLM-driven orchestration)
        logging.info("🧠 Brain Agent analyzing query and selecting tools...")
        brain_result = await brain_agent.process_query(query, user_id=user_id)
        
        brain_time = time.time() - start_time
        
        # Phase 2: Heart Agent Synthesis (REAL LLM-driven synthesis)
        if brain_result.get("success"):
            logging.info("❤️ Heart Agent synthesizing optimal response...")
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
        
        
def convert_pydantic_messages_to_dict(pydantic_messages: List[QueryMessage]) -> List[Dict[str, str]]:
    """Convert Pydantic QueryMessage objects to dict format expected by Brain Agent"""
    
    return [{"role": msg.role, "content": msg.content} for msg in pydantic_messages]

def convert_langchain_messages_to_dict(lc_messages: List) -> List[Dict[str, str]]:
    """Convert LangChain messages to dict format expected by Brain Agent"""
    
    messages = []
    for msg in lc_messages:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})
        else:
            # Handle other message types if needed
            messages.append({"role": "unknown", "content": str(msg.content)})
    
    return messages


@router.post("/set_agents")
async def set_brain_heart_agents(request: UpdateAgentsRequest):
    """
    Update global Brain-Heart agents and web search configuration.
    """
    if request.brain_provider:
        settings.brain_provider = request.brain_provider
    if request.brain_model:
        settings.brain_model = request.brain_model
    if request.heart_provider:
        settings.heart_provider = request.heart_provider
    if request.heart_model:
        settings.heart_model = request.heart_model
    if request.use_premium_search is not None:
        settings.use_premium_search = request.use_premium_search
    if request.web_model:
        settings.web_model = request.web_model

    return JSONResponse(content={
        "status": "success",
        "current_settings": {
            "brain_provider": settings.brain_provider,
            "brain_model": settings.brain_model,
            "heart_provider": settings.heart_provider,
            "heart_model": settings.heart_model,
            "use_premium_search": settings.use_premium_search,
            "web_model": settings.web_model
        }
    })


@router.post("/chat", dependencies=[Depends(RateLimiter(times=6, seconds=60))])
async def chat_brain_heart_system(request: ChatMessage = Body(...)):
    """New endpoint specifically for Brain-Heart system with messages support"""
    
    try:
        user_id = request.userid
        user_query = request.user_query
        chat_history = request.chat_history if hasattr(request, 'chat_history') else []
        
        
        safe_log_user_data(user_id, 'brain_heart_chat', message_count=len(user_query))
        
        brain_provider = settings.brain_provider or os.getenv("BRAIN_LLM_PROVIDER")
        brain_model = settings.brain_model or os.getenv("BRAIN_LLM_MODEL")
        heart_provider = settings.heart_provider or os.getenv("HEART_LLM_PROVIDER")
        heart_model = settings.heart_model or os.getenv("HEART_LLM_MODEL")
        use_premium_search = settings.use_premium_search or os.getenv("USE_PREMIUM_SEARCH", "false").lower() == "true"
        web_model = settings.web_model or os.getenv("WEB_LLM_MODEL", "")
        
    
        
        config = Config()
        
        brain_model_config = config.create_llm_config(
            provider=brain_provider,
            model=brain_model,
            max_tokens=1000
        )
        heart_model_config = config.create_llm_config(
            provider=heart_provider,
            model=heart_model,
            max_tokens=1000
        )
        web_model_config = config.get_tool_configs(
            web_model=web_model,
            use_premium_search=use_premium_search
        )
        
        tool_manager = ToolManager(
            config,
            brain_model_config,
            web_model,
            use_premium_search
        )
        
        brain_llm = LLMClient(brain_model_config)
        heart_llm = LLMClient(heart_model_config)
        
        
        optimizedAgent = OptimizedAgent(
            brain_llm,
            heart_llm,
            tool_manager
        )
        
        
        
        result = await optimizedAgent.process_query(user_query, chat_history, user_id)
        
        if result["success"]:
            
            safe_log_response(result, level='info')
            return JSONResponse(content=result, status_code=200)
        else:
            return JSONResponse(
                content={"error": result["error"]}, 
                status_code=500
            )
            
    except Exception as e:
        logging.error(f"❌ Brain-Heart chat endpoint failed: {str(e)}")
        return JSONResponse(
            content={"error": f"Brain-Heart processing failed: {str(e)}"}, 
            status_code=500
        )

# Additional utility endpoints



@router.get("/chat/memory/{user_id}")
async def get_user_memory(user_id: str):
    """Get memory summary for a specific user"""
    
    try:
        # You'll need to access the brain agent instance
        # This assumes it's available through some dependency injection or global state
        agents = await create_agents_async(
            config=None,  # Your config here
            brain_model_config=None,  # Your brain model config
            heart_model_config=None,  # Your heart model config
            web_model_config=None,  # Your web model config
            use_premium_search=False
        )
        
        if agents["status"] != "success":
            return JSONResponse(
                content={"error": f"Failed to create agents: {agents['error']}"}, 
                status_code=500
            )
        
        memory_summary = agents["brain_agent"].get_memory_summary()
        
        return JSONResponse(content={
            "user_id": user_id,
            "memory_summary": memory_summary
        }, status_code=200)
        
    except Exception as e:
        logging.error(f"❌ Memory retrieval failed: {str(e)}")
        return JSONResponse(
            content={"error": f"Memory retrieval failed: {str(e)}"}, 
            status_code=500
        )

@router.post("/chat/single-query")
async def chat_single_query_legacy(query: str = Body(...), user_id: str = Body(...)):
    """Legacy endpoint for single query processing (backward compatibility)"""
    
    try:
        # Convert single query to messages format
        messages = [{"role": "user", "content": query}]
        
        # Create a mock ChatMessage request
        class MockUserQuery:
            def __init__(self, messages):
                self.messages = [QueryMessage(role=msg["role"], content=msg["content"]) for msg in messages]
        
        class MockChatMessage:
            def __init__(self, userid, user_query):
                self.userid = userid
                self.user_query = user_query
        
        mock_request = MockChatMessage(user_id, MockUserQuery(messages))
        
        # Process through the Brain-Heart system
        return await chat_brain_heart_system(mock_request)
        
    except Exception as e:
        safe_log_error(e, context={'endpoint': 'single_query_processing'})
        return JSONResponse(
            content={"error": "Single query processing failed"}, 
            status_code=500
        )
        


@router.get("/get-collections")
async def get_collections():
    """List all user IDs with collections"""
    try:
        users = get_users()
        data = []

        for user in users:
            collections = list_user_collections(user)
            for c in collections:
                meta = get_collection_metadata(user, c)
                data.append({"name": c, "metadata": [meta]})
        return JSONResponse(content={"collections": data}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


        
@router.get("/collections/{user_id}")
async def list_collections(user_id: str):
    """List all collections for a given user"""
    try:
        collections = list_user_collections(user_id)
        data = []
        for c in collections:
            meta = get_collection_metadata(user_id, c)
            data.append({"name": c, "metadata": meta})
        return JSONResponse(content={"collections": data}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/collections/create/local")
async def create_collection_local(
    user_id: str = Form(...),
    collection_name: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """Create RAG collection from uploaded local files"""
    try:
        from core.knowledge_base import create_knowledge_base

        upload_dir = f"temp_uploads/{user_id}/{collection_name}"
        os.makedirs(upload_dir, exist_ok=True)
        file_paths = []

        # Save uploaded files
        for file in files:
            file_path = os.path.join(upload_dir, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())
            file_paths.append(file_path)

        result = create_knowledge_base(user_id, collection_name, file_paths)

        # Cleanup
        shutil.rmtree(upload_dir, ignore_errors=True)

        if result.get("success"):
            return JSONResponse(content={"message": "✅ Collection created!"}, status_code=200)
        else:
            return JSONResponse(content={"error": result.get("error")}, status_code=500)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/collections/create/drive")
async def create_collection_drive(user_id: str = Form(...), collection_name: str = Form(...), files: List[dict] = Body(...)):
    """Create RAG collection from Google Drive (multi-account support)"""
    try:
        from core.google_drive_integration import MultiAccountGoogleDriveManager
        from core.knowledge_base import create_knowledge_base

        drive_manager = MultiAccountGoogleDriveManager(user_id)

        # Download only when creating
        downloaded_files = drive_manager.download_files_with_conflict_resolution(files)

        if not downloaded_files:
            return JSONResponse(content={"error": "Failed to download files"}, status_code=500)

        has_multiple_accounts = len(set(f.get('account_id') for f in files)) > 1
        if has_multiple_accounts:
            result = create_knowledge_base(user_id, collection_name, downloaded_files, account_info="multi_account")
        else:
            result = create_knowledge_base(user_id, collection_name, downloaded_files)

        # Cleanup local files and revoke sessions
        for fp in downloaded_files:
            if os.path.exists(fp):
                os.remove(fp)
        drive_manager.security_disconnect_all()

        if result.get("success"):
            return JSONResponse(content={"message": "✅ Collection created from Google Drive!"}, status_code=200)
        else:
            return JSONResponse(content={"error": result.get("error")}, status_code=500)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.delete("/collections/{user_id}/{collection_name}")
async def delete_collection(user_id: str, collection_name: str):
    """Delete an existing collection"""
    try:
        collection_path = f"db_collection/{user_id}/{collection_name}"
        if os.path.exists(collection_path):
            shutil.rmtree(collection_path)
            return JSONResponse(content={"message": f"✅ Collection {collection_name} deleted"}, status_code=200)
        else:
            return JSONResponse(content={"error": "Collection not found"}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
