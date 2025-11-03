"""
Knowledge Base Manager with Organization Integration
Handles document storage, retrieval, and permissions using ChromaDB and MongoDB
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4
from datetime import datetime, timezone
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from pymongo import MongoClient
import hashlib
import logging
from os import getenv, path
from redis.asyncio import Redis
logger = logging.getLogger(__name__)


class KnowledgeBaseManager:
    """
    Manages knowledge base operations with organization and team-level access control
    Integrates ChromaDB for vector storage and MongoDB for metadata/permissions
    """
    
    def __init__(
        self,
        chroma_client: chromadb.Client,
        mongo_client: MongoClient,
        org_manager,
        redis_client: Redis,
        embedding_function=None,
        database_name: str = "knowledge_base"
    ):
        """
        Initialize the Knowledge Base Manager
        
        Args:
            chroma_client: ChromaDB client instance
            mongo_client: MongoDB client instance
            redis_client: Redis client instance
            org_manager: OrganizationManager instance
            embedding_function: Embedding function for ChromaDB (default: sentence-transformers)
            database_name: MongoDB database name
        """
        self.chroma_client = chroma_client
        self.mongo_client = mongo_client
        self.org_manager = org_manager
        self.redis_client = redis_client
        self.db = mongo_client[database_name]
        
        # MongoDB collections
        self.documents_collection = self.db.documents
        self.collections_metadata = self.db.collection_metadata
        
        # Default embedding function
        self.embedding_function = embedding_function or embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self._setup_indexes()
        
    
    def _setup_indexes(self):
        """Create MongoDB indexes for efficient queries"""
        # Index on org_id and team_id for fast lookups
        self.documents_collection.create_index([("org_id", 1), ("team_id", 1)])
        self.documents_collection.create_index("doc_id", unique=True)
        self.documents_collection.create_index("collection_name")
        
        self.collections_metadata.create_index([("org_id", 1), ("team_id", 1)])
        self.collections_metadata.create_index([("org_id", 1), ("collection_name", 1)], unique=True)
    
    def _get_collection_name(self, org_id: str, team_id: Optional[str] = None) -> str:
        """
        Generate a unique collection name for ChromaDB
        
        Args:
            org_id: Organization ID
            team_id: Team ID (optional, for team-specific collections)
        
        Returns:
            Unique collection name
        """
        if team_id:
            return f"{org_id}_{team_id}"
        return f"{org_id}_global"
    
    async def _set_org_cache(self, org_id:str, user_id:str):
        """Set organization data in Redis cache"""
        return await self.redis_client.set(user_id, org_id)
        
    async def _get_org_cache(self, user_id:str) -> Optional[str]:
        """Get organization data from Redis cache"""
        return await self.redis_client.get(user_id)
    
    async def _set_collection_cache(self, collection_name:str, user_id:str):
        """Set collection data in Redis cache"""
        return await self.redis_client.set(user_id, collection_name)
        
    async def _get_collection_cache(self, user_id:str) -> Optional[str]:
        """Get collection data from Redis cache"""
        return await self.redis_client.get(user_id)
    
    async def create_global_collection(self):
        """
        Create a global collection for all organizations if it doesn't exist
        """
        try:
            collection_name = "global_knowledge_base"
            existing = await asyncio.to_thread(
                self.collections_metadata.find_one,
                {"org_id": "global", "collection_name": collection_name}
            )
            
            if not existing:
                chroma_collection = await asyncio.to_thread(
                    self.chroma_client.get_or_create_collection,
                    name=collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"org_id": "global"}
                )
                
                collection_doc = {
                    "collection_id": str(uuid4()),
                    "collection_name": collection_name,
                    "description": "Global knowledge base accessible to all organizations",
                    "org_id": "global",
                    "team_id": None,
                    "chroma_collection_name": collection_name,
                    "created_by": "system",
                    "created_at": datetime.now(timezone.utc),
                    "metadata": {},
                    "document_count": 0
                }
                
                await asyncio.to_thread(
                    self.collections_metadata.insert_one,
                    collection_doc
                )
        except Exception as e:
            print(f"Error creating global collection: {e}")
    
    async def create_collection(
        self,
        org_id: str,
        collection_name: str,
        description: str,
        user_id: str,
        team_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new knowledge base collection or return existing if created by same user
        
        Args:
            org_id: Organization ID
            collection_name: Human-readable collection name
            description: Collection description
            user_id: User creating the collection
            team_id: Team ID (if team-specific collection)
            metadata: Additional metadata
        
        Returns:
            {
                "success": bool, 
                "collection_id": str, 
                "chroma_collection_name": str,
                "is_existing": bool,
                "message": str,
                "error": str
            }
        """
        try:
            # Check permission
            action = "upload_documents"
            has_permission = await self.org_manager.check_permission(org_id, user_id, action)
            
            if not has_permission:
                return {"success": False, "error": "Permission denied"}
            
            # Verify team access if team_id provided
            if team_id:
                org = await self.org_manager.get_organization(org_id)
                if not org:
                    return {"success": False, "error": "Organization not found"}
                
                user_team_id = org.get("members", {}).get(user_id, {}).get("team_id")
                if user_team_id != team_id and org.get("owner_id") != user_id:
                    return {"success": False, "error": "Access denied to this team"}
            
            # Check if collection already exists
            existing = await asyncio.to_thread(
                self.collections_metadata.find_one,
                {"org_id": org_id, "collection_name": collection_name}
            )
            
            if existing:
                # Check if the user is the creator
                if existing.get("created_by") == user_id:
                    # User created this collection, return existing details
                    return {
                        "success": True,
                        "collection_id": existing["collection_id"],
                        "collection_name": existing["collection_name"],
                        "description": existing.get("description", ""),
                        "chroma_collection_name": existing["chroma_collection_name"],
                        "team_id": existing.get("team_id"),
                        "document_count": existing.get("document_count", 0),
                        "created_at": existing.get("created_at").isoformat() if existing.get("created_at") else None,
                        "is_existing": True,
                        "message": f"Collection '{collection_name}' already exists and you are the creator"
                    }
                else:
                    # Someone else created this collection
                    creator_name = existing.get("created_by", "another user")
                    return {
                        "success": False,
                        "error": f"Collection '{collection_name}' already exists in this organization (created by {creator_name})"
                    }
            
            # Collection doesn't exist, create new one
            # Generate ChromaDB collection name
            chroma_collection_name = self._get_collection_name(org_id, team_id)
            
            # Create ChromaDB collection
            chroma_collection = await asyncio.to_thread(
                self.chroma_client.get_or_create_collection,
                name=chroma_collection_name,
                embedding_function=self.embedding_function,
                metadata={"org_id": org_id, "team_id": team_id or "global"}
            )
            
            # Store metadata in MongoDB
            collection_id = str(uuid4())
            collection_doc = {
                "collection_id": collection_id,
                "collection_name": collection_name,
                "description": description,
                "org_id": org_id,
                "team_id": team_id,
                "chroma_collection_name": chroma_collection_name,
                "created_by": user_id,
                "created_at": datetime.now(timezone.utc),
                "metadata": metadata or {},
                "document_count": 0
            }
            
            await asyncio.to_thread(
                self.collections_metadata.insert_one,
                collection_doc
            )
            
            return {
                "success": True,
                "collection_id": collection_id,
                "collection_name": collection_name,
                "description": description,
                "chroma_collection_name": chroma_collection_name,
                "team_id": team_id,
                "document_count": 0,
                "is_existing": False,
                "message": f"Collection '{collection_name}' created successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return {"success": False, "error": str(e)}
    
    async def upload_documents(
        self,
        org_id: str,
        collection_name: str,
        documents: List[str],
        user_id: str,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Upload documents to a knowledge base collection
        
        Args:
            org_id: Organization ID
            collection_name: Collection name
            documents: List of document texts
            user_id: User uploading documents
            metadatas: Optional metadata for each document
            ids: Optional custom IDs for documents
        
        Returns:
            {"success": bool, "document_ids": List[str], "error": str}
        """
        try:
            # Check permission
            has_permission = await self.org_manager.check_permission(org_id, user_id, "upload_documents")
            
            if not has_permission:
                return {"success": False, "error": "Permission denied"}
            
            # Get collection metadata
            collection_meta = await asyncio.to_thread(
                self.collections_metadata.find_one,
                {"org_id": org_id, "collection_name": collection_name}
            )
            
            if not collection_meta:
                return {"success": False, "error": "Collection not found"}
            
            # Verify team access
            team_id = collection_meta.get("team_id")
            if team_id:
                org = await self.org_manager.get_organization(org_id)
                user_team_id = org.get("members", {}).get(user_id, {}).get("team_id")
                if user_team_id != team_id and org.get("owner_id") != user_id:
                    return {"success": False, "error": "Access denied to this team's collection"}
            
            # Generate document IDs if not provided
            if ids is None:
                ids = [str(uuid4()) for _ in documents]
            
            # Prepare metadatas
            if metadatas is None:
                metadatas = [{} for _ in documents]
            
            # Add system metadata
            for i, meta in enumerate(metadatas):
                meta.update({
                    "org_id": org_id,
                    "team_id": team_id,
                    "uploaded_by": user_id,
                    "uploaded_at": datetime.now(timezone.utc).isoformat(),
                    "doc_id": ids[i]
                })
            
            # Get ChromaDB collection
            chroma_collection_name = collection_meta["chroma_collection_name"]
            chroma_collection = await asyncio.to_thread(
                self.chroma_client.get_collection,
                name=chroma_collection_name,
                embedding_function=self.embedding_function
            )
            
            # Add documents to ChromaDB
            await asyncio.to_thread(
                chroma_collection.add,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            # Store document metadata in MongoDB
            doc_records = []
            for i, doc_id in enumerate(ids):
                doc_records.append({
                    "doc_id": doc_id,
                    "collection_id": collection_meta["collection_id"],
                    "collection_name": collection_name,
                    "org_id": org_id,
                    "team_id": team_id,
                    "uploaded_by": user_id,
                    "uploaded_at": datetime.now(timezone.utc),
                    "document_text": documents[i],
                    "metadata": metadatas[i],
                    "document_hash": hashlib.sha256(documents[i].encode()).hexdigest()
                })
            
            await asyncio.to_thread(
                self.documents_collection.insert_many,
                doc_records
            )
            
            # Update document count
            await asyncio.to_thread(
                self.collections_metadata.update_one,
                {"collection_id": collection_meta["collection_id"]},
                {"$inc": {"document_count": len(documents)}}
            )
            
            return {
                "success": True,
                "document_ids": ids,
                "count": len(documents)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
        
    
    async def query_documents(
        self,
        org_id: str,
        collection_name: str,
        query_text: str,
        user_id: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query documents using semantic search
        
        Args:
            org_id: Organization ID
            collection_name: Collection name
            query_text: Query text
            user_id: User making the query
            n_results: Number of results to return
            where: Optional metadata filters
        
        Returns:
            {"success": bool, "results": List[Dict], "error": str}
        """
        try:
            # Verify user is in organization
            org = await self.org_manager.get_organization(org_id)
            if not org:
                return {"success": False, "error": "Organization not found"}
            
            if user_id not in org.get("members", {}):
                return {"success": False, "error": "User not in organization"}
            
            # Get collection metadata
            collection_meta = await asyncio.to_thread(
                self.collections_metadata.find_one,
                {"org_id": org_id, "collection_name": collection_name}
            )
            
            if not collection_meta:
                return {"success": False, "error": "Collection not found"}
            
            # Verify team access
            team_id = collection_meta.get("team_id")
            if team_id:
                user_team_id = org.get("members", {}).get(user_id, {}).get("team_id")
                user_role = org.get("members", {}).get(user_id, {}).get("role")
                
                # Allow access if user is in the team, is owner, or can select teams
                can_access = (
                    user_team_id == team_id or
                    org.get("owner_id") == user_id or
                    user_role == "owner"
                )
                
                if not can_access:
                    return {"success": False, "error": "Access denied to this team's collection"}
            
            # Get ChromaDB collection
            chroma_collection_name = collection_meta["chroma_collection_name"]
            chroma_collection = await asyncio.to_thread(
                self.chroma_client.get_collection,
                name=chroma_collection_name,
                embedding_function=self.embedding_function
            )
            
            # Perform query
            results = await asyncio.to_thread(
                chroma_collection.query,
                query_texts=[query_text],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results and results.get("ids") and len(results["ids"]) > 0:
                for i in range(len(results["ids"][0])):
                    formatted_results.append({
                        "id": results["ids"][0][i],
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i]
                    })
            
            return {
                "success": True,
                "results": formatted_results,
                "query": query_text,
                "count": len(formatted_results)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def delete_documents(
        self,
        org_id: str,
        collection_name: str,
        document_ids: List[str],
        user_id: str
    ) -> Dict[str, Any]:
        """
        Delete documents from a collection
        
        Args:
            org_id: Organization ID
            collection_name: Collection name
            document_ids: List of document IDs to delete
            user_id: User deleting documents
        
        Returns:
            {"success": bool, "deleted_count": int, "error": str}
        """
        try:
            # Check permission
            has_permission = await self.org_manager.check_permission(org_id, user_id, "delete_collection")
            
            if not has_permission:
                return {"success": False, "error": "Permission denied"}
            
            # Get collection metadata
            collection_meta = await asyncio.to_thread(
                self.collections_metadata.find_one,
                {"org_id": org_id, "collection_name": collection_name}
            )
            
            if not collection_meta:
                return {"success": False, "error": "Collection not found"}
            
            # Verify team access
            team_id = collection_meta.get("team_id")
            if team_id:
                org = await self.org_manager.get_organization(org_id)
                user_team_id = org.get("members", {}).get(user_id, {}).get("team_id")
                if user_team_id != team_id and org.get("owner_id") != user_id:
                    return {"success": False, "error": "Access denied to this team's collection"}
            
            # Get ChromaDB collection
            chroma_collection_name = collection_meta["chroma_collection_name"]
            chroma_collection = await asyncio.to_thread(
                self.chroma_client.get_collection,
                name=chroma_collection_name,
                embedding_function=self.embedding_function
            )
            
            # Delete from ChromaDB
            await asyncio.to_thread(
                chroma_collection.delete,
                ids=document_ids
            )
            
            # Delete from MongoDB
            result = await asyncio.to_thread(
                self.documents_collection.delete_many,
                {"doc_id": {"$in": document_ids}, "org_id": org_id}
            )
            
            # Update document count
            await asyncio.to_thread(
                self.collections_metadata.update_one,
                {"collection_id": collection_meta["collection_id"]},
                {"$inc": {"document_count": -result.deleted_count}}
            )
            
            return {
                "success": True,
                "deleted_count": result.deleted_count
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def delete_collection(
    self,
    org_id: str,
    collection_name: str,
    user_id: str
    ) -> Dict[str, Any]:
        """
        Delete an entire collection
        
        Args:
            org_id: Organization ID
            collection_name: Collection name
            user_id: User deleting the collection
        
        Returns:
            {"success": bool, "error": str}
        """
        try:
            # Check permission
            has_permission = await self.org_manager.check_permission(org_id, user_id, "delete_collection")
            logger.info("Existing collections: %s", self.chroma_client.list_collections())

            if not has_permission:
                return {"success": False, "error": "Permission denied"}
            
            # Get collection metadata
            collection_meta = await asyncio.to_thread(
                self.collections_metadata.find_one,
                {"org_id": org_id, "collection_name": collection_name}
            )
            
            if not collection_meta:
                return {"success": False, "error": "Collection not found"}
            
            # Verify team access
            team_id = collection_meta.get("team_id")
            if team_id:
                org = await self.org_manager.get_organization(org_id)
                user_team_id = org.get("members", {}).get(user_id, {}).get("team_id")
                if user_team_id != team_id and org.get("owner_id") != user_id:
                    return {"success": False, "error": "Access denied to this team's collection"}
            
            # Delete ChromaDB collection (ignore if not exists)
            chroma_collection_name = collection_meta["chroma_collection_name"]
            try:
                await asyncio.to_thread(
                    self.chroma_client.delete_collection,
                    name=chroma_collection_name
                )
            except Exception as chroma_error:
                # Only ignore if it's a 'not found' kind of error
                error_str = str(chroma_error).lower()
                if "not found" in error_str or "does not exist" in error_str or "no such collection" in error_str:
                    logger.warning(f"Chroma collection '{chroma_collection_name}' does not exist, skipping deletion.")
                else:
                    raise  # re-raise unexpected errors
            
            # Delete all documents from MongoDB
            await asyncio.to_thread(
                self.documents_collection.delete_many,
                {"collection_id": collection_meta["collection_id"]}
            )
            
            # Delete collection metadata
            await asyncio.to_thread(
                self.collections_metadata.delete_one,
                {"collection_id": collection_meta["collection_id"]}
            )
            
            return {"success": True}
            
        except Exception as e:
            logger.exception("Error deleting collection: %s", e)
            return {"success": False, "error": str(e)}

    
    async def list_collections(
        self,
        org_id: str,
        user_id: str,
        team_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List all collections accessible to a user
        
        Args:
            org_id: Organization ID
            user_id: User requesting the list
            team_id: Optional team filter
        
        Returns:
            {"success": bool, "collections": List[Dict], "error": str}
        """
        try:
            # Verify user is in organization
            org = await self.org_manager.get_organization(org_id)
            if not org:
                return {"success": False, "error": "Organization not found"}
            
            if user_id not in org.get("members", {}):
                return {"success": False, "error": "User not in organization"}
            
            # Build query
            query = {"org_id": org_id}
            
            # Filter by team if specified
            if team_id:
                user_team_id = org.get("members", {}).get(user_id, {}).get("team_id")
                if user_team_id != team_id and org.get("owner_id") != user_id:
                    return {"success": False, "error": "Access denied to this team"}
                query["team_id"] = team_id
            else:
                # If no team specified, show collections user has access to
                user_team_id = org.get("members", {}).get(user_id, {}).get("team_id")
                user_role = org.get("members", {}).get(user_id, {}).get("role")
                
                if user_role == "owner":
                    # Owner sees all collections
                    pass
                else:
                    # Non-owners see global collections and their team's collections
                    query["$or"] = [
                        {"team_id": None},
                        {"team_id": user_team_id}
                    ]
            
            # Get collections
            collections = await asyncio.to_thread(
                lambda: list(self.collections_metadata.find(query))
            )
            
            # Format results
            formatted_collections = []
            for col in collections:
                formatted_collections.append({
                    "collection_id": col["collection_id"],
                    "collection_name": col["collection_name"],
                    "description": col.get("description", ""),
                    "team_id": col.get("team_id"),
                    "document_count": col.get("document_count", 0),
                    "created_by": col.get("created_by"),
                    "created_at": col.get("created_at").isoformat() if col.get("created_at") else None
                })
            
            return {
                "success": True,
                "collections": formatted_collections,
                "count": len(formatted_collections)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_collection_stats(
        self,
        org_id: str,
        collection_name: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Get statistics for a collection
        
        Args:
            org_id: Organization ID
            collection_name: Collection name
            user_id: User requesting stats
        
        Returns:
            {"success": bool, "stats": Dict, "error": str}
        """
        try:
            # Verify user access
            org = await self.org_manager.get_organization(org_id)
            if not org or user_id not in org.get("members", {}):
                return {"success": False, "error": "Access denied"}
            
            # Get collection metadata
            collection_meta = await asyncio.to_thread(
                self.collections_metadata.find_one,
                {"org_id": org_id, "collection_name": collection_name}
            )
            
            if not collection_meta:
                return {"success": False, "error": "Collection not found"}
            
            # Verify team access
            team_id = collection_meta.get("team_id")
            if team_id:
                user_team_id = org.get("members", {}).get(user_id, {}).get("team_id")
                if user_team_id != team_id and org.get("owner_id") != user_id:
                    return {"success": False, "error": "Access denied"}
            
            # Get document stats from MongoDB
            doc_count = await asyncio.to_thread(
                self.documents_collection.count_documents,
                {"collection_id": collection_meta["collection_id"]}
            )
            
            # Get recent uploads
            recent_docs = await asyncio.to_thread(
                lambda: list(self.documents_collection.find(
                    {"collection_id": collection_meta["collection_id"]}
                ).sort("uploaded_at", -1).limit(5))
            )
            
            stats = {
                "collection_id": collection_meta["collection_id"],
                "collection_name": collection_name,
                "description": collection_meta.get("description", ""),
                "team_id": team_id,
                "document_count": doc_count,
                "created_by": collection_meta.get("created_by"),
                "created_at": collection_meta.get("created_at").isoformat() if collection_meta.get("created_at") else None,
                "recent_uploads": [
                    {
                        "doc_id": doc["doc_id"],
                        "uploaded_by": doc.get("uploaded_by"),
                        "uploaded_at": doc.get("uploaded_at").isoformat() if doc.get("uploaded_at") else None
                    }
                    for doc in recent_docs
                ]
            }
            
            return {
                "success": True,
                "stats": stats
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def update_collection(
        self,
        org_id: str,
        collection_name: str,
        user_id: str,
        new_name: Optional[str] = None,
        new_description: Optional[str] = None,
        new_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Update collection metadata
        
        Args:
            org_id: Organization ID
            collection_name: Current collection name
            user_id: User updating the collection
            new_name: New collection name (optional)
            new_description: New description (optional)
            new_metadata: New metadata (optional)
        
        Returns:
            {"success": bool, "error": str}
        """
        try:
            # Check permission
            has_permission = await self.org_manager.check_permission(org_id, user_id, "edit_collection")
            
            if not has_permission:
                return {"success": False, "error": "Permission denied"}
            
            # Get collection metadata
            collection_meta = await asyncio.to_thread(
                self.collections_metadata.find_one,
                {"org_id": org_id, "collection_name": collection_name}
            )
            
            if not collection_meta:
                return {"success": False, "error": "Collection not found"}
            
            # Verify team access
            team_id = collection_meta.get("team_id")
            if team_id:
                org = await self.org_manager.get_organization(org_id)
                user_team_id = org.get("members", {}).get(user_id, {}).get("team_id")
                if user_team_id != team_id and org.get("owner_id") != user_id:
                    return {"success": False, "error": "Access denied to this team's collection"}
            
            # Build update document
            update_doc = {}
            if new_name:
                # Check if new name already exists
                existing = await asyncio.to_thread(
                    self.collections_metadata.find_one,
                    {"org_id": org_id, "collection_name": new_name, "collection_id": {"$ne": collection_meta["collection_id"]}}
                )
                if existing:
                    return {"success": False, "error": "Collection name already exists"}
                update_doc["collection_name"] = new_name
            
            if new_description:
                update_doc["description"] = new_description
            
            if new_metadata:
                update_doc["metadata"] = new_metadata
            
            if not update_doc:
                return {"success": False, "error": "No updates provided"}
            
            update_doc["updated_at"] = datetime.now(timezone.utc)
            update_doc["updated_by"] = user_id
            
            # Update in MongoDB
            await asyncio.to_thread(
                self.collections_metadata.update_one,
                {"collection_id": collection_meta["collection_id"]},
                {"$set": update_doc}
            )
            
            # Update documents if name changed
            if new_name:
                await asyncio.to_thread(
                    self.documents_collection.update_many,
                    {"collection_id": collection_meta["collection_id"]},
                    {"$set": {"collection_name": new_name}}
                )
            
            return {"success": True}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_document(
        self,
        org_id: str,
        document_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Get a specific document by ID
        
        Args:
            org_id: Organization ID
            document_id: Document ID
            user_id: User requesting the document
        
        Returns:
            {"success": bool, "document": Dict, "error": str}
        """
        try:
            # Verify user is in organization
            org = await self.org_manager.get_organization(org_id)
            if not org or user_id not in org.get("members", {}):
                return {"success": False, "error": "Access denied"}
            
            # Get document
            doc = await asyncio.to_thread(
                self.documents_collection.find_one,
                {"doc_id": document_id, "org_id": org_id}
            )
            
            if not doc:
                return {"success": False, "error": "Document not found"}
            
            # Verify team access
            team_id = doc.get("team_id")
            if team_id:
                user_team_id = org.get("members", {}).get(user_id, {}).get("team_id")
                if user_team_id != team_id and org.get("owner_id") != user_id:
                    return {"success": False, "error": "Access denied to this team's document"}
            
            # Format document
            formatted_doc = {
                "doc_id": doc["doc_id"],
                "collection_id": doc.get("collection_id"),
                "collection_name": doc.get("collection_name"),
                "document_text": doc.get("document_text"),
                "metadata": doc.get("metadata", {}),
                "uploaded_by": doc.get("uploaded_by"),
                "uploaded_at": doc.get("uploaded_at").isoformat() if doc.get("uploaded_at") else None
            }
            
            return {
                "success": True,
                "document": formatted_doc
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def update_document(
        self,
        org_id: str,
        document_id: str,
        user_id: str,
        new_text: Optional[str] = None,
        new_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Update a document's text or metadata
        
        Args:
            org_id: Organization ID
            document_id: Document ID
            user_id: User updating the document
            new_text: New document text (optional)
            new_metadata: New metadata (optional)
        
        Returns:
            {"success": bool, "error": str}
        """
        try:
            # Check permission
            has_permission = await self.org_manager.check_permission(org_id, user_id, "upload_documents")
            
            if not has_permission:
                return {"success": False, "error": "Permission denied"}
            
            # Get document
            doc = await asyncio.to_thread(
                self.documents_collection.find_one,
                {"doc_id": document_id, "org_id": org_id}
            )
            
            if not doc:
                return {"success": False, "error": "Document not found"}
            
            # Verify team access
            team_id = doc.get("team_id")
            if team_id:
                org = await self.org_manager.get_organization(org_id)
                user_team_id = org.get("members", {}).get(user_id, {}).get("team_id")
                if user_team_id != team_id and org.get("owner_id") != user_id:
                    return {"success": False, "error": "Access denied to this team's document"}
            
            # Build update
            mongo_update = {}
            chroma_update = {}
            
            if new_text:
                mongo_update["document_text"] = new_text
                mongo_update["document_hash"] = hashlib.sha256(new_text.encode()).hexdigest()
                chroma_update["documents"] = [new_text]
            
            if new_metadata:
                # Merge with existing metadata
                merged_metadata = doc.get("metadata", {}).copy()
                merged_metadata.update(new_metadata)
                mongo_update["metadata"] = merged_metadata
                chroma_update["metadatas"] = [merged_metadata]
            
            if not mongo_update and not chroma_update:
                return {"success": False, "error": "No updates provided"}
            
            mongo_update["updated_at"] = datetime.now(timezone.utc)
            mongo_update["updated_by"] = user_id
            
            # Update in MongoDB
            await asyncio.to_thread(
                self.documents_collection.update_one,
                {"doc_id": document_id},
                {"$set": mongo_update}
            )
            
            # Update in ChromaDB if needed
            if chroma_update:
                collection_meta = await asyncio.to_thread(
                    self.collections_metadata.find_one,
                    {"collection_id": doc["collection_id"]}
                )
                
                if collection_meta:
                    chroma_collection = await asyncio.to_thread(
                        self.chroma_client.get_collection,
                        name=collection_meta["chroma_collection_name"],
                        embedding_function=self.embedding_function
                    )
                    
                    await asyncio.to_thread(
                        chroma_collection.update,
                        ids=[document_id],
                        **chroma_update
                    )
            
            return {"success": True}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def search_all_collections(
        self,
        org_id: str,
        query_text: str,
        user_id: str,
        n_results: int = 5,
        team_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search across all accessible collections
        
        Args:
            org_id: Organization ID
            query_text: Query text
            user_id: User making the query
            n_results: Number of results per collection
            team_id: Optional team filter
        
        Returns:
            {"success": bool, "results": Dict[str, List], "error": str}
        """
        try:
            # Get accessible collections
            collections_result = await self.list_collections(org_id, user_id, team_id)
            
            if not collections_result["success"]:
                return collections_result
            
            # Search each collection
            all_results = {}
            for collection in collections_result["collections"]:
                collection_name = collection["collection_name"]
                search_result = await self.query_documents(
                    org_id=org_id,
                    collection_name=collection_name,
                    query_text=query_text,
                    user_id=user_id,
                    n_results=n_results
                )
                
                if search_result["success"] and search_result["results"]:
                    all_results[collection_name] = search_result["results"]
            
            return {
                "success": True,
                "results": all_results,
                "collections_searched": len(collections_result["collections"]),
                "query": query_text
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def batch_upload_documents(
        self,
        org_id: str,
        collection_name: str,
        user_id: str,
        documents: List[str],
        batch_size: int = 100,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Upload documents in batches for better performance
        
        Args:
            org_id: Organization ID
            collection_name: Collection name
            user_id: User uploading documents
            documents: List of document texts
            batch_size: Number of documents per batch
            metadatas: Optional metadata for each document
        
        Returns:
            {"success": bool, "total_uploaded": int, "batches": int, "error": str}
        """
        try:
            total_docs = len(documents)
            batches = (total_docs + batch_size - 1) // batch_size
            uploaded_count = 0
            
            for i in range(0, total_docs, batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_metas = metadatas[i:i + batch_size] if metadatas else None
                
                result = await self.upload_documents(
                    org_id=org_id,
                    collection_name=collection_name,
                    documents=batch_docs,
                    user_id=user_id,
                    metadatas=batch_metas
                )
                
                if result["success"]:
                    uploaded_count += result["count"]
                else:
                    return {
                        "success": False,
                        "error": f"Failed at batch {i//batch_size + 1}: {result.get('error')}",
                        "uploaded_before_error": uploaded_count
                    }
            
            return {
                "success": True,
                "total_uploaded": uploaded_count,
                "batches": batches
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_documents_by_metadata(
        self,
        org_id: str,
        collection_name: str,
        user_id: str,
        metadata_filter: Dict[str, Any],
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get documents by metadata filter
        
        Args:
            org_id: Organization ID
            collection_name: Collection name
            user_id: User requesting documents
            metadata_filter: MongoDB-style metadata filter
            limit: Maximum number of documents to return
        
        Returns:
            {"success": bool, "documents": List[Dict], "error": str}
        """
        try:
            # Verify user access
            org = await self.org_manager.get_organization(org_id)
            if not org or user_id not in org.get("members", {}):
                return {"success": False, "error": "Access denied"}
            
            # Get collection metadata
            collection_meta = await asyncio.to_thread(
                self.collections_metadata.find_one,
                {"org_id": org_id, "collection_name": collection_name}
            )
            
            if not collection_meta:
                return {"success": False, "error": "Collection not found"}
            
            # Verify team access
            team_id = collection_meta.get("team_id")
            if team_id:
                user_team_id = org.get("members", {}).get(user_id, {}).get("team_id")
                if user_team_id != team_id and org.get("owner_id") != user_id:
                    return {"success": False, "error": "Access denied"}
            
            # Build query
            query = {
                "collection_id": collection_meta["collection_id"],
                "org_id": org_id
            }
            
            # Add metadata filters
            for key, value in metadata_filter.items():
                query[f"metadata.{key}"] = value
            
            # Get documents
            docs = await asyncio.to_thread(
                lambda: list(self.documents_collection.find(query).limit(limit))
            )
            
            # Format documents
            formatted_docs = []
            for doc in docs:
                formatted_docs.append({
                    "doc_id": doc["doc_id"],
                    "document_text": doc.get("document_text"),
                    "metadata": doc.get("metadata", {}),
                    "uploaded_by": doc.get("uploaded_by"),
                    "uploaded_at": doc.get("uploaded_at").isoformat() if doc.get("uploaded_at") else None
                })
            
            return {
                "success": True,
                "documents": formatted_docs,
                "count": len(formatted_docs)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def export_collection(
        self,
        org_id: str,
        collection_name: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Export all documents from a collection
        
        Args:
            org_id: Organization ID
            collection_name: Collection name
            user_id: User exporting the collection
        
        Returns:
            {"success": bool, "documents": List[Dict], "metadata": Dict, "error": str}
        """
        try:
            # Verify user access
            org = await self.org_manager.get_organization(org_id)
            if not org or user_id not in org.get("members", {}):
                return {"success": False, "error": "Access denied"}
            
            # Get collection metadata
            collection_meta = await asyncio.to_thread(
                self.collections_metadata.find_one,
                {"org_id": org_id, "collection_name": collection_name}
            )
            
            if not collection_meta:
                return {"success": False, "error": "Collection not found"}
            
            # Verify team access
            team_id = collection_meta.get("team_id")
            if team_id:
                user_team_id = org.get("members", {}).get(user_id, {}).get("team_id")
                if user_team_id != team_id and org.get("owner_id") != user_id:
                    return {"success": False, "error": "Access denied"}
            
            # Get all documents
            docs = await asyncio.to_thread(
                lambda: list(self.documents_collection.find(
                    {"collection_id": collection_meta["collection_id"]}
                ))
            )
            
            # Format for export
            export_docs = []
            for doc in docs:
                export_docs.append({
                    "doc_id": doc["doc_id"],
                    "text": doc.get("document_text"),
                    "metadata": doc.get("metadata", {}),
                    "uploaded_by": doc.get("uploaded_by"),
                    "uploaded_at": doc.get("uploaded_at").isoformat() if doc.get("uploaded_at") else None
                })
            
            export_metadata = {
                "collection_name": collection_name,
                "description": collection_meta.get("description", ""),
                "team_id": team_id,
                "created_by": collection_meta.get("created_by"),
                "created_at": collection_meta.get("created_at").isoformat() if collection_meta.get("created_at") else None,
                "exported_by": user_id,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "document_count": len(export_docs)
            }
            
            return {
                "success": True,
                "documents": export_docs,
                "metadata": export_metadata
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# Global instance
kb_manager = None


def initialize_kb_manager(
    chroma_client: chromadb.Client,
    mongo_client: MongoClient,
    org_manager,
    redis_client,
    embedding_function=None,
    database_name: str = "knowledge_base"
) -> KnowledgeBaseManager:
    """
    Initialize the global knowledge base manager
    
    Args:
        chroma_client: ChromaDB client
        mongo_client: MongoDB client
        org_manager: OrganizationManager instance
        redis_client: Redis client
        embedding_function: Optional embedding function
        database_name: MongoDB database name
    
    Returns:
        KnowledgeBaseManager instance
    """
    global kb_manager
    kb_manager = KnowledgeBaseManager(
        chroma_client=chroma_client,
        mongo_client=mongo_client,
        org_manager=org_manager,
        redis_client=redis_client,
        embedding_function=embedding_function,
        database_name=database_name
    )
    
    return kb_manager


# Convenience functions
async def create_collection(
    org_id: str,
    collection_name: str,
    description: str,
    user_id: str,
    team_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a new collection"""
    if kb_manager is None:
        raise Exception("KnowledgeBaseManager not initialized. Call initialize_kb_manager() first.")
    return await kb_manager.create_collection(org_id, collection_name, description, user_id, team_id, metadata)


async def upload_documents(
    org_id: str,
    collection_name: str,
    documents: List[str],
    user_id: str,
    metadatas: Optional[List[Dict[str, Any]]] = None,
    ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Upload documents to a collection"""
    if kb_manager is None:
        raise Exception("KnowledgeBaseManager not initialized. Call initialize_kb_manager() first.")
    return await kb_manager.upload_documents(org_id, collection_name, documents, user_id, metadatas, ids)


async def query_documents(
    org_id: str,
    collection_name: str,
    query_text: str,
    user_id: str,
    n_results: int = 5,
    where: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Query documents"""
    if kb_manager is None:
        raise Exception("KnowledgeBaseManager not initialized. Call initialize_kb_manager() first.")
    return await kb_manager.query_documents(org_id, collection_name, query_text, user_id, n_results, where)


async def list_collections(
    org_id: str,
    user_id: str,
    team_id: Optional[str] = None
) -> Dict[str, Any]:
    """List accessible collections"""
    if kb_manager is None:
        raise Exception("KnowledgeBaseManager not initialized. Call initialize_kb_manager() first.")
    return await kb_manager.list_collections(org_id, user_id, team_id)


async def delete_collection(
    org_id: str,
    collection_name: str,
    user_id: str
) -> Dict[str, Any]:
    """Delete a collection"""
    if kb_manager is None:
        raise Exception("KnowledgeBaseManager not initialized. Call initialize_kb_manager() first.")
    return await kb_manager.delete_collection(org_id, collection_name, user_id)


async def delete_documents(
    org_id: str,
    collection_name: str,
    document_ids: List[str],
    user_id: str
) -> Dict[str, Any]:
    """Delete specific documents from a collection"""
    if kb_manager is None:
        raise Exception("KnowledgeBaseManager not initialized. Call initialize_kb_manager() first.")
    return await kb_manager.delete_documents(org_id, collection_name, document_ids, user_id)


async def get_collection_stats(
    org_id: str,
    collection_name: str,
    user_id: str
) -> Dict[str, Any]:
    """Get collection statistics"""
    if kb_manager is None:
        raise Exception("KnowledgeBaseManager not initialized. Call initialize_kb_manager() first.")
    return await kb_manager.get_collection_stats(org_id, collection_name, user_id)


async def update_collection(
    org_id: str,
    collection_name: str,
    user_id: str,
    new_name: Optional[str] = None,
    new_description: Optional[str] = None,
    new_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Update collection metadata"""
    if kb_manager is None:
        raise Exception("KnowledgeBaseManager not initialized. Call initialize_kb_manager() first.")
    return await kb_manager.update_collection(org_id, collection_name, user_id, new_name, new_description, new_metadata)


async def get_document(
    org_id: str,
    document_id: str,
    user_id: str
) -> Dict[str, Any]:
    """Get a specific document by ID"""
    if kb_manager is None:
        raise Exception("KnowledgeBaseManager not initialized. Call initialize_kb_manager() first.")
    return await kb_manager.get_document(org_id, document_id, user_id)


async def update_document(
    org_id: str,
    document_id: str,
    user_id: str,
    new_text: Optional[str] = None,
    new_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Update a document's text or metadata"""
    if kb_manager is None:
        raise Exception("KnowledgeBaseManager not initialized. Call initialize_kb_manager() first.")
    return await kb_manager.update_document(org_id, document_id, user_id, new_text, new_metadata)


async def search_all_collections(
    org_id: str,
    query_text: str,
    user_id: str,
    n_results: int = 5,
    team_id: Optional[str] = None
) -> Dict[str, Any]:
    """Search across all accessible collections"""
    if kb_manager is None:
        raise Exception("KnowledgeBaseManager not initialized. Call initialize_kb_manager() first.")
    return await kb_manager.search_all_collections(org_id, query_text, user_id, n_results, team_id)


async def batch_upload_documents(
    org_id: str,
    collection_name: str,
    user_id: str,
    documents: List[str],
    batch_size: int = 100,
    metadatas: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Upload documents in batches"""
    if kb_manager is None:
        raise Exception("KnowledgeBaseManager not initialized. Call initialize_kb_manager() first.")
    return await kb_manager.batch_upload_documents(org_id, collection_name, user_id, documents, batch_size, metadatas)


async def get_org_cache(user_id: str) -> Optional[str]:
    """Get organization ID from cache for a user"""
    if kb_manager is None:
        raise Exception("KnowledgeBaseManager not initialized. Call initialize_kb_manager() first.")
    return await kb_manager._get_org_cache(f"org_{user_id}")

async def set_org_cache(user_id: str, org_id: str) -> None:
    """Set organization ID in cache for a user"""
    if kb_manager is None:
        raise Exception("KnowledgeBaseManager not initialized. Call initialize_kb_manager() first.")
    return await kb_manager._set_org_cache(org_id, f"org_{user_id}")
    
async def get_collection_cache(user_id: str) -> Optional[str]:
    """Get collection name from cache for a user"""
    if kb_manager is None:
        raise Exception("KnowledgeBaseManager not initialized. Call initialize_kb_manager() first.")
    return await kb_manager._get_collection_cache(f"collection_{user_id}")

async def set_collection_cache(user_id: str, collection_name: str) -> None:
    """Set collection name in cache for a user"""
    if kb_manager is None:
        raise Exception("KnowledgeBaseManager not initialized. Call initialize_kb_manager() first.")
    return await kb_manager._set_collection_cache(collection_name, f"collection_{user_id}")



async def get_documents_by_metadata(
    org_id: str,
    collection_name: str,
    user_id: str,
    metadata_filter: Dict[str, Any],
    limit: int = 100
) -> Dict[str, Any]:
    """Get documents by metadata filter"""
    if kb_manager is None:
        raise Exception("KnowledgeBaseManager not initialized. Call initialize_kb_manager() first.")
    return await kb_manager.get_documents_by_metadata(org_id, collection_name, user_id, metadata_filter, limit)


async def export_collection(
    org_id: str,
    collection_name: str,
    user_id: str
) -> Dict[str, Any]:
    """Export all documents from a collection"""
    if kb_manager is None:
        raise Exception("KnowledgeBaseManager not initialized. Call initialize_kb_manager() first.")
    return await kb_manager.export_collection(org_id, collection_name, user_id)