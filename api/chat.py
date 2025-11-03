from fastapi import FastAPI, APIRouter, UploadFile, File, Form, Depends, Body
from fastapi.responses import JSONResponse
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from redis.asyncio import Redis
from contextlib import asynccontextmanager


# Updated imports for new knowledge base manager
from core.knowledge_base import (
    initialize_kb_manager,
    create_collection,
    upload_documents,
    query_documents,
    delete_collection,
    delete_documents,
    list_collections,
    get_collection_stats,
    update_collection,
    get_document,
    update_document,
    search_all_collections,
    batch_upload_documents,
    get_documents_by_metadata,
    export_collection,
    set_org_cache,
    set_collection_cache,
    get_org_cache,
    get_collection_cache,
    kb_manager
)

from core.organization_manager import (
    initialize_org_manager,
    create_organization,
    join_organization,
    get_organization,
    delete_organization,
    org_manager
)

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
import chromadb
from pymongo import MongoClient

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    logging.info("⚡ Starting app lifespan...")
    
    global optimizedAgent, org_manager, kb_manager
    config = Config()

    redis_client = Redis(
        host=os.getenv('REDIS_HOST'), 
        port=os.getenv('REDIS_PORT'), 
        decode_responses=os.getenv('REDIS_DECODE_RESPONSES'), 
        username=os.getenv('REDIS_USERNAME'), 
        password=os.getenv('REDIS_PASSWORD')
    )
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
    
    # Initialize Organization Manager
    mongo_client = MongoClient(os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'))
    org_manager = initialize_org_manager(mongo_client, database_name="knowledge_base")
    
    # Initialize Knowledge Base Manager
    chroma_client = chromadb.Client()
    kb_manager = initialize_kb_manager(
        chroma_client=chroma_client,
        mongo_client=mongo_client,
        org_manager=org_manager,
        redis_client=redis_client,
        database_name="knowledge_base"
    )
    await kb_manager.create_global_collection()
    
    logging.info("✅ Organization Manager and Knowledge Base Manager initialized")

    optimizedAgent.worker_task = asyncio.create_task(
        optimizedAgent.background_task_worker()
    )
    logging.info("OptimizedAgent background worker started")

    try:
        yield
    finally:
        logging.info("⚡ Shutting down app lifespan...")
        optimizedAgent.worker_task.cancel()
        try:
            await optimizedAgent.worker_task
        except asyncio.CancelledError:
            logging.info("OptimizedAgent worker cancelled cleanly")


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


# ============================================================================
# KNOWLEDGE BASE / COLLECTION ENDPOINTS (NEW IMPLEMENTATION)
# ============================================================================

@router.post("/organizations/{org_id}/collections/create")
async def create_collection_endpoint(
    org_id: str,
    collection_name: str = Body(..., embed=True),
    description: str = Body(..., embed=True),
    user_id: str = Body(..., embed=True),
    team_id: Optional[str] = Body(None, embed=True),
    metadata: Optional[Dict[str, Any]] = Body(None, embed=True)
):
    """
    Create a new knowledge base collection within an organization.
    """
    try:
        result = await create_collection(
            org_id=org_id,
            collection_name=collection_name,
            description=description,
            user_id=user_id,
            team_id=team_id,
            metadata=metadata
        )
        
        if result.get("success"):
            res = await set_collection_cache(user_id, collection_name)
            logging.info(f"Collection cached for user {user_id}: {res}")
            return JSONResponse(
                content={
                    "message": f"Collection '{collection_name}' created successfully!",
                    "collection_id": result.get("collection_id")
                }, 
                status_code=200
            )
        else:
            return JSONResponse(
                content={"error": result.get("error")}, 
                status_code=400
            )
            
    except Exception as e:
        logging.error(f"Error creating collection: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )


@router.get("/organizations/{org_id}/collections")
async def get_collections_endpoint(
    org_id: str,
    user_id: str,
    team_id: Optional[str] = None
):
    """
    List all collections accessible to a user in an organization.
    """
    try:
        result = await list_collections(
            org_id=org_id,
            user_id=user_id,
            team_id=team_id
        )
        
        if result.get("success"):
            return JSONResponse(
                content={
                    "collections": result.get("collections"),
                    "count": result.get("count")
                }, 
                status_code=200
            )
        else:
            return JSONResponse(
                content={"error": result.get("error")}, 
                status_code=400
            )
            
    except Exception as e:
        logging.error(f"Error getting collections: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )


@router.get("/organizations/{org_id}/collections/{collection_name}/stats")
async def get_collection_stats_endpoint(
    org_id: str,
    collection_name: str,
    user_id: str
):
    """
    Get detailed statistics for a collection.
    """
    try:
        result = await get_collection_stats(
            org_id=org_id,
            collection_name=collection_name,
            user_id=user_id
        )
        
        if result.get("success"):
            return JSONResponse(
                content={"stats": result.get("stats")}, 
                status_code=200
            )
        else:
            return JSONResponse(
                content={"error": result.get("error")}, 
                status_code=400
            )
            
    except Exception as e:
        logging.error(f"Error getting collection stats: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )


@router.post("/organizations/{org_id}/collections/{collection_name}/upload")
async def upload_documents_endpoint(
    org_id: str,
    collection_name: str,
    user_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    Upload documents/files to a collection.
    Processes files and stores them as searchable documents.
    """
    try:
        documents = []
        metadatas = []
        
        # Process each uploaded file
        for file in files:
            content = await file.read()
            
            # Handle different file types
            if file.filename.endswith('.txt'):
                text = content.decode('utf-8')
            elif file.filename.endswith('.pdf'):
                # You'll need to implement PDF extraction
                from PyPDF2 import PdfReader
                import io
                pdf_reader = PdfReader(io.BytesIO(content))
                text = "\n".join([page.extract_text() for page in pdf_reader.pages])
            elif file.filename.endswith('.json'):
                text = content.decode('utf-8')
            else:
                text = content.decode('utf-8', errors='ignore')
            
            # Chunk the document if it's too large
            # Simple chunking - you may want more sophisticated chunking
            max_chunk_size = 1000
            chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
            
            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append({
                    "filename": file.filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "file_type": file.filename.split('.')[-1]
                })
        
        # Upload to knowledge base
        result = await upload_documents(
            org_id=org_id,
            collection_name=collection_name,
            documents=documents,
            user_id=user_id,
            metadatas=metadatas
        )
        
        if result.get("success"):
            return JSONResponse(
                content={
                    "message": f"Successfully uploaded {len(files)} file(s) with {result.get('count')} chunks",
                    "document_ids": result.get("document_ids"),
                    "total_chunks": result.get("count")
                }, 
                status_code=200
            )
        else:
            return JSONResponse(
                content={"error": result.get("error")}, 
                status_code=400
            )
            
    except Exception as e:
        logging.error(f"Error uploading documents: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )


@router.post("/organizations/{org_id}/collections/{collection_name}/upload-text")
async def upload_text_documents_endpoint(
    org_id: str,
    collection_name: str,
    user_id: str = Body(..., embed=True),
    documents: List[str] = Body(..., embed=True),
    metadatas: Optional[List[Dict[str, Any]]] = Body(None, embed=True)
):
    """
    Upload text documents directly to a collection.
    """
    try:
        result = await upload_documents(
            org_id=org_id,
            collection_name=collection_name,
            documents=documents,
            user_id=user_id,
            metadatas=metadatas
        )
        
        if result.get("success"):
            return JSONResponse(
                content={
                    "message": f"Successfully uploaded {result.get('count')} documents",
                    "document_ids": result.get("document_ids")
                }, 
                status_code=200
            )
        else:
            return JSONResponse(
                content={"error": result.get("error")}, 
                status_code=400
            )
            
    except Exception as e:
        logging.error(f"Error uploading text documents: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )


@router.post("/organizations/{org_id}/collections/{collection_name}/batch-upload")
async def batch_upload_documents_endpoint(
    org_id: str,
    collection_name: str,
    user_id: str = Body(..., embed=True),
    documents: List[str] = Body(..., embed=True),
    batch_size: int = Body(100, embed=True),
    metadatas: Optional[List[Dict[str, Any]]] = Body(None, embed=True)
):
    """
    Upload large number of documents in batches for better performance.
    """
    try:
        result = await batch_upload_documents(
            org_id=org_id,
            collection_name=collection_name,
            user_id=user_id,
            documents=documents,
            batch_size=batch_size,
            metadatas=metadatas
        )
        
        if result.get("success"):
            return JSONResponse(
                content={
                    "message": f"Successfully uploaded {result.get('total_uploaded')} documents in {result.get('batches')} batches",
                    "total_uploaded": result.get("total_uploaded"),
                    "batches": result.get("batches")
                }, 
                status_code=200
            )
        else:
            return JSONResponse(
                content={"error": result.get("error")}, 
                status_code=400
            )
            
    except Exception as e:
        logging.error(f"Error batch uploading documents: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )


@router.post("/organizations/{org_id}/collections/{collection_name}/query")
async def query_collection_endpoint(
    org_id: str,
    collection_name: str,
    user_id: str = Body(..., embed=True),
    query_text: str = Body(..., embed=True),
    n_results: int = Body(5, embed=True),
    where: Optional[Dict[str, Any]] = Body(None, embed=True)
):
    """
    Query a specific collection using semantic search.
    Returns top N most relevant documents.
    """
    try:
        result = await query_documents(
            org_id=org_id,
            collection_name=collection_name,
            query_text=query_text,
            user_id=user_id,
            n_results=n_results,
            where=where
        )
        
        if result.get("success"):
            return JSONResponse(
                content={
                    "query": result.get("query"),
                    "results": result.get("results"),
                    "count": result.get("count")
                }, 
                status_code=200
            )
        else:
            return JSONResponse(
                content={"error": result.get("error")}, 
                status_code=400
            )
            
    except Exception as e:
        logging.error(f"Error querying collection: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )


@router.post("/organizations/{org_id}/search-all")
async def search_all_collections_endpoint(
    org_id: str,
    user_id: str = Body(..., embed=True),
    query_text: str = Body(..., embed=True),
    n_results: int = Body(5, embed=True),
    team_id: Optional[str] = Body(None, embed=True)
):
    """
    Search across all accessible collections in an organization.
    """
    try:
        result = await search_all_collections(
            org_id=org_id,
            query_text=query_text,
            user_id=user_id,
            n_results=n_results,
            team_id=team_id
        )
        
        if result.get("success"):
            return JSONResponse(
                content={
                    "query": result.get("query"),
                    "results": result.get("results"),
                    "collections_searched": result.get("collections_searched")
                }, 
                status_code=200
            )
        else:
            return JSONResponse(
                content={"error": result.get("error")}, 
                status_code=400
            )
            
    except Exception as e:
        logging.error(f"Error searching all collections: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )


@router.put("/organizations/{org_id}/collections/{collection_name}")
async def update_collection_endpoint(
    org_id: str,
    collection_name: str,
    user_id: str = Body(..., embed=True),
    new_name: Optional[str] = Body(None, embed=True),
    new_description: Optional[str] = Body(None, embed=True),
    new_metadata: Optional[Dict[str, Any]] = Body(None, embed=True)
):
    """
    Update collection metadata (name, description, custom metadata).
    """
    try:
        result = await update_collection(
            org_id=org_id,
            collection_name=collection_name,
            user_id=user_id,
            new_name=new_name,
            new_description=new_description,
            new_metadata=new_metadata
        )
        
        if result.get("success"):
            return JSONResponse(
                content={"message": "Collection updated successfully"}, 
                status_code=200
            )
        else:
            return JSONResponse(
                content={"error": result.get("error")}, 
                status_code=400
            )
            
    except Exception as e:
        logging.error(f"Error updating collection: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )


@router.delete("/organizations/{org_id}/collections/{collection_name}")
async def delete_collection_endpoint(
    org_id: str,
    collection_name: str,
    user_id: str
):
    """
    Delete an entire collection and all its documents.
    """
    try:
        result = await delete_collection(
            org_id=org_id,
            collection_name=collection_name,
            user_id=user_id
        )
        
        if result.get("success"):
            return JSONResponse(
                content={"message": f"Collection '{collection_name}' deleted successfully!"}, 
                status_code=200
            )
        else:
            return JSONResponse(
                content={"error": result.get("error")}, 
                status_code=400
            )
            
    except Exception as e:
        logging.error(f"Error deleting collection: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )


@router.delete("/organizations/{org_id}/collections/{collection_name}/documents")
async def delete_documents_endpoint(
    org_id: str,
    collection_name: str,
    user_id: str = Body(..., embed=True),
    document_ids: List[str] = Body(..., embed=True)
):
    """
    Delete specific documents from a collection.
    """
    try:
        result = await delete_documents(
            org_id=org_id,
            collection_name=collection_name,
            document_ids=document_ids,
            user_id=user_id
        )
        
        if result.get("success"):
            return JSONResponse(
                content={
                    "message": f"Successfully deleted {result.get('deleted_count')} documents",
                    "deleted_count": result.get("deleted_count")
                }, 
                status_code=200
            )
        else:
            return JSONResponse(
                content={"error": result.get("error")}, 
                status_code=400
            )
            
    except Exception as e:
        logging.error(f"Error deleting documents: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )


@router.get("/organizations/{org_id}/documents/{document_id}")
async def get_document_endpoint(
    org_id: str,
    document_id: str,
    user_id: str
):
    """
    Get a specific document by ID.
    """
    try:
        result = await get_document(
            org_id=org_id,
            document_id=document_id,
            user_id=user_id
        )
        
        if result.get("success"):
            return JSONResponse(
                content={"document": result.get("document")}, 
                status_code=200
            )
        else:
            return JSONResponse(
                content={"error": result.get("error")}, 
                status_code=400
            )
            
    except Exception as e:
        logging.error(f"Error getting document: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )


@router.put("/organizations/{org_id}/documents/{document_id}")
async def update_document_endpoint(
    org_id: str,
    document_id: str,
    user_id: str = Body(..., embed=True),
    new_text: Optional[str] = Body(None, embed=True),
    new_metadata: Optional[Dict[str, Any]] = Body(None, embed=True)
):
    """
    Update a document's text or metadata.
    """
    try:
        result = await update_document(
            org_id=org_id,
            document_id=document_id,
            user_id=user_id,
            new_text=new_text,
            new_metadata=new_metadata
        )
        
        if result.get("success"):
            return JSONResponse(
                content={"message": "Document updated successfully"}, 
                status_code=200
            )
        else:
            return JSONResponse(
                content={"error": result.get("error")}, 
                status_code=400
            )
            
    except Exception as e:
        logging.error(f"Error updating document: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )


@router.post("/organizations/{org_id}/collections/{collection_name}/query-by-metadata")
async def get_documents_by_metadata_endpoint(
    org_id: str,
    collection_name: str,
    user_id: str = Body(..., embed=True),
    metadata_filter: Dict[str, Any] = Body(..., embed=True),
    limit: int = Body(100, embed=True)
):
    """
    Get documents filtered by metadata criteria.
    """
    try:
        result = await get_documents_by_metadata(
            org_id=org_id,
            collection_name=collection_name,
            user_id=user_id,
            metadata_filter=metadata_filter,
            limit=limit
        )
        
        if result.get("success"):
            return JSONResponse(
                content={
                    "documents": result.get("documents"),
                    "count": result.get("count")
                }, 
                status_code=200
            )
        else:
            return JSONResponse(
                content={"error": result.get("error")}, 
                status_code=400
            )
            
    except Exception as e:
        logging.error(f"Error getting documents by metadata: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )


@router.get("/organizations/{org_id}/collections/{collection_name}/export")
async def export_collection_endpoint(
    org_id: str,
    collection_name: str,
    user_id: str
):
    """
    Export all documents from a collection for backup/migration.
    """
    try:
        result = await export_collection(
            org_id=org_id,
            collection_name=collection_name,
            user_id=user_id
        )
        
        if result.get("success"):
            return JSONResponse(
                content={
                    "documents": result.get("documents"),
                    "metadata": result.get("metadata")
                }, 
                status_code=200
            )
        else:
            return JSONResponse(
                content={"error": result.get("error")}, 
                status_code=400
            )
            
    except Exception as e:
        logging.error(f"Error exporting collection: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )


# ============================================================================
# ORGANIZATION ENDPOINTS
# ============================================================================

@router.get("/organizations/{org_id}")
async def get_organization_info(org_id: str):
    """
    Get organization information by org_id.
    """
    try:
        org_data = await get_organization(org_id)
        
        if org_data:
            return JSONResponse(
                content={"organization": org_data}, 
                status_code=200
            )
        else:
            return JSONResponse(
                content={"error": "Organization not found"}, 
                status_code=404
            )
            
    except Exception as e:
        logging.error(f"Error getting organization info: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )
        

from fastapi import Query, Body, Path

# ============================================================================
# ORGANIZATION ENDPOINTS
# ============================================================================

@router.post("/organizations/create")
async def create_organization_endpoint(
    org_name: str = Body(..., embed=True),
    user_name: str = Body(..., embed=True),
    user_id: str = Body(..., embed=True)
):
    """
    Create a new organization.
    """
    try:
        result = await create_organization(org_name, user_name, user_id)
        
        if result.get("success"):
            logging.info(f"Organization created with ID: {result.get('org_id')}")
            org_data = await set_org_cache(user_id, result.get("org_id"))
            logging.info(f"Organization cached for user {user_id}: {org_data}")
            return JSONResponse(
                content={
                    "organization_id": result.get("org_id"),
                    "invite_code": result.get("invite_code"),
                    "is_existing": result.get("is_existing", False),
                    "message": result.get("message")
                }, 
                status_code=200
            )
        else:
            return JSONResponse(
                content={"error": result.get("error")}, 
                status_code=400
            )
            
    except Exception as e:
        logging.error(f"Error creating organization: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )


@router.post("/organizations/join")
async def join_organization_endpoint(
    invite_code: str = Body(..., embed=True),
    user_name: str = Body(..., embed=True),
    user_id: str = Body(..., embed=True)
):
    """
    Join an existing organization using an invite code.
    """
    try:
        result = await join_organization(invite_code, user_name, user_id)
        
        if result.get("success"):
            # Cache the organization for the user
            await set_org_cache(user_id, result.get("org_id"))
            logging.info(f"User {user_id} joined organization {result.get('org_id')}")
            
            return JSONResponse(
                content={
                    "organization_id": result.get("org_id"),
                    "message": "Successfully joined organization"
                }, 
                status_code=200
            )
        else:
            return JSONResponse(
                content={"error": result.get("error")}, 
                status_code=400
            )
            
    except Exception as e:
        logging.error(f"Error joining organization: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )


@router.delete("/organizations/{org_id}")
async def remove_organization_endpoint(
    org_id: str = Path(..., description="Organization ID to delete")
):
    """
    Delete an organization by org_id.
    """
    try:
        logging.info(f"Attempting to delete organization: {org_id}")
        result = await delete_organization(org_id)
        
        if result.get("success"):
            return JSONResponse(
                content={"message": "Organization deleted successfully"}, 
                status_code=200
            )
        else:
            return JSONResponse(
                content={"error": result.get("error")}, 
                status_code=400
            )
            
    except Exception as e:
        logging.error(f"Error deleting organization: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )


@router.get("/organizations/get-org/{user_id}")
async def get_active_organization_endpoint(
    user_id: str = Path(..., description="User ID to get organization for")
):
    """
    Get organization information for a user.
    """
    try:
        logging.info(f"Fetching organization data from cache for user {user_id}")
        org_data = await get_org_cache(user_id)
        logging.info(f"Retrieved organization data from cache for user {user_id}: {org_data}")
        
        if org_data:
            return JSONResponse(
                content={"organization": org_data},
                status_code=200
            )
        else:
            return JSONResponse(
                content={"error": "Organization not found for user"}, 
                status_code=404
            )
            
    except Exception as e:
        logging.error(f"Error getting organization for user: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )


@router.patch("/organizations/set")
async def set_active_organization_endpoint(
    org_id: str = Body(..., embed=True),
    user_id: str = Body(..., embed=True)
):
    """
    Set active organization for a user in cache.
    """
    try:
        logging.info(f"Setting organization {org_id} for user {user_id}")
        
        if not org_id:
            return JSONResponse(
                content={"error": "Organization ID is required"}, 
                status_code=400
            )
        
        await set_org_cache(user_id, org_id)
        return JSONResponse(
            content={
                "message": "Organization set in cache successfully",
                "org_id": org_id
            },
            status_code=200
        )
            
    except Exception as e:
        logging.error(f"Error setting organization in cache: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )


# ============================================================================
# COLLECTION ENDPOINTS
# ============================================================================

@router.get("/collections/get-collection/{user_id}")
async def get_active_collection_endpoint(
    user_id: str = Path(..., description="User ID to get collection for")
):
    """
    Get collection information for a user.
    """
    try:
        logging.info(f"Fetching collection data from cache for user {user_id}")
        collection_data = await get_collection_cache(user_id)
        logging.info(f"Retrieved collection data from cache for user {user_id}: {collection_data}")
        
        if collection_data:
            return JSONResponse(
                content={"collection": collection_data},
                status_code=200
            )
        else:
            return JSONResponse(
                content={"error": "Collection not found for user"}, 
                status_code=404
            )
            
    except Exception as e:
        logging.error(f"Error getting collection for user: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )


@router.patch("/collections/set")
async def set_active_collection_endpoint(
    collection_name: str = Body(..., embed=True),
    user_id: str = Body(..., embed=True)
):
    """
    Set active collection for a user in cache.
    """
    try:
        logging.info(f"Setting collection {collection_name} for user {user_id}")
        
        if not collection_name:
            return JSONResponse(
                content={"error": "Collection name is required"}, 
                status_code=400
            )
        
        await set_collection_cache(user_id, collection_name)
        return JSONResponse(
            content={
                "message": "Collection set in cache successfully",
                "collection_name": collection_name
            },
            status_code=200
        )
            
    except Exception as e:
        logging.error(f"Error setting collection in cache: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )


# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get("/health/knowledge-base")
async def health_check_kb():
    """
    Health check for knowledge base systems (ChromaDB + MongoDB).
    """
    try:
        from core.knowledge_base import kb_manager
        
        if kb_manager is None:
            return JSONResponse(
                content={"error": "Knowledge Base Manager not initialized"}, 
                status_code=503
            )
        
        health_status = {
            "chromadb": "connected",
            "mongodb": "connected",
            "timestamp": time.time()
        }
        
        # Test ChromaDB
        try:
            kb_manager.chroma_client.heartbeat()
        except Exception as e:
            health_status["chromadb"] = f"error: {str(e)}"
        
        # Test MongoDB
        try:
            kb_manager.mongo_client.admin.command('ping')
        except Exception as e:
            health_status["mongodb"] = f"error: {str(e)}"
        
        status_code = 200 if (health_status["chromadb"] == "connected" and 
                              health_status["mongodb"] == "connected") else 503
        
        return JSONResponse(content=health_status, status_code=status_code)
        
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )