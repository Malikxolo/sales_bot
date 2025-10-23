"""
Knowledge Base Creation - Adapted for Brain-Heart Deep Research System
User-specific vector database creation using ChromaDB + LangChain
ENHANCED: PDF + DOCX Support + Chroma Cloud Support + Multi-User Isolation
SECURITY: Path sanitization to prevent directory traversal attacks
"""

import os
import shutil
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
import uuid

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.schema import Document

# Import path security utilities
from core.path_security import (
    sanitize_path_component,
    create_safe_user_path,
    validate_safe_path,
    sanitize_filename
)

# Multi-format document loaders with graceful fallback
try:
    from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
    PDF_SUPPORT = True
    DOCX_SUPPORT = True
except ImportError:
    PyPDFLoader = None
    UnstructuredWordDocumentLoader = None
    PDF_SUPPORT = False
    DOCX_SUPPORT = False
    
load_dotenv()

# Setup logger
logger = logging.getLogger(__name__)

class UTF8TextLoader(TextLoader):
    """A custom TextLoader that enforces UTF-8 encoding."""
    def __init__(self, file_path: str, **kwargs):
        super().__init__(file_path, encoding='utf-8', **kwargs)

class KnowledgeBaseManager:
    """Manages user-specific knowledge base creation following Brain-Heart patterns"""
    
    def __init__(self, base_path: str = "db_collection"):
        self.base_path = base_path
        self.embeddings_model = "text-embedding-3-small"
        self.chunk_size = 1000
        self.chunk_overlap = 100
        
        logger.info("KnowledgeBaseManager initialized")
        if PDF_SUPPORT:
            logger.info("PDF support available")
        if DOCX_SUPPORT:
            logger.info("DOCX support available")
    
    def _get_chroma_client(self, path: str = None):
        """Get ChromaDB client - Cloud if API key exists, else local"""
        if os.getenv('CHROMA_API_KEY'):
            logger.info("Using Chroma Cloud client")
            return chromadb.CloudClient(
                api_key=os.getenv('CHROMA_API_KEY'),
                tenant=os.getenv('CHROMA_TENANT'), 
                database=os.getenv('CHROMA_DATABASE')
            )
        else:
            logger.info(f"Using local ChromaDB client: {path}")
            return chromadb.PersistentClient(path=path)
    
    def _get_namespaced_collection_name(self, user_id: str, collection_name: str) -> str:
        """Create user-namespaced collection name to prevent cross-user contamination"""
        return f"{user_id}_{collection_name}"
    
    def create_user_knowledge_base(
        self, 
        user_id: str, 
        collection_name: str, 
        file_paths: List[str]
    ) -> Dict[str, Any]:
        """Create user-specific knowledge base from uploaded files"""
        try:
            logger.info(f"Starting knowledge base creation for user {user_id}, collection: {collection_name}")
            
            # SECURITY: Sanitize user inputs to prevent directory traversal
            safe_user_id = sanitize_path_component(user_id)
            safe_collection_name = sanitize_path_component(collection_name)
            
            # Extract file names from file paths
            file_names = [os.path.basename(file_path) for file_path in file_paths]
            
            # 1. Setup user-specific paths with security validation
            user_path = create_safe_user_path(self.base_path, safe_user_id)
            collection_path = create_safe_user_path(self.base_path, safe_user_id, safe_collection_name)
            chroma_db_path = validate_safe_path(collection_path, "chroma_db")
            
            # Ensure directories exist
            os.makedirs(chroma_db_path, exist_ok=True)
            logger.info(f"Created directories: {chroma_db_path}")
            
            # 2. Load documents from uploaded files
            logger.info(f"Loading {len(file_paths)} documents...")
            documents = self._load_documents_from_files(file_paths, collection_path)
            
            if not documents:
                return {
                    "success": False,
                    "error": "No documents could be loaded",
                    "details": {
                        "file_count": len(file_paths),
                        "file_names": file_names
                    }
                }
            
            logger.info(f"Successfully loaded {len(documents)} documents")
            
            # 3. Split documents into chunks
            logger.info("Splitting documents into chunks...")
            texts = self._split_documents(documents)
            logger.info(f"Split into {len(texts)} chunks")
            
            # 4. Create embeddings and vector store
            logger.info("Creating embeddings and ChromaDB vector store...")
            result = self._create_vector_store(texts, chroma_db_path, safe_collection_name, safe_user_id)
            
            if result["success"]:
                # 5. Store metadata
                metadata = {
                    "user_id": safe_user_id,
                    "collection_name": safe_collection_name,
                    "snippet": texts[0].page_content[:50] if texts else "",
                    "file_count": len(file_paths),
                    "file_names": file_names,
                    "document_count": len(documents),
                    "chunk_count": len(texts),
                    "embedding_model": self.embeddings_model,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "created_at": result["created_at"],
                    "chroma_db_path": chroma_db_path
                }
                
                self._save_metadata(collection_path, metadata)
                
                logger.info(f"Knowledge base created successfully: {safe_collection_name}")
                return {
                    "success": True,
                    "message": f"Knowledge base '{safe_collection_name}' created successfully",
                    "metadata": metadata
                }
            else:
                return result
                
        except ValueError as ve:
            # Security validation error
            logger.error(f"Security validation failed: {ve}")
            return {
                "success": False,
                "error": f"Invalid input: {str(ve)}"
            }
        except Exception as e:
            logger.error(f"Failed to create knowledge base: {e}")
            return {
                "success": False,
                "error": f"Knowledge base creation failed: {str(e)}"
            }
    
    def _load_documents_from_files(self, file_paths: List[str], collection_path: str) -> List:
        """Load documents from uploaded files - ENHANCED with comprehensive format support"""
        documents = []
        
        # Import unstructured for comprehensive format support
        try:
            from unstructured.partition.auto import partition
            UNSTRUCTURED_AVAILABLE = True
            logger.info("Unstructured package available")
        except ImportError:
            UNSTRUCTURED_AVAILABLE = False
            logger.warning("Unstructured not available. Limited format support.")
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
                
            filename = os.path.basename(file_path)
            file_extension = os.path.splitext(filename)[1].lower()
            
            logger.info(f"Processing file: {filename} (type: {file_extension})")
            
            try:
                # Use unstructured for ALL supported formats
                if UNSTRUCTURED_AVAILABLE and file_extension in [
                    # Text & Document formats
                    '.md', '.rtf',
                    # Data formats 
                    '.csv', '.tsv', '.json',
                    # Excel formats
                    '.xlsx', '.xls', '.xlsm', '.xlsb', '.xltx', '.xltm',
                    # Word formats (handled by unstructured)
                    '.docm', '.dotx', '.dotm', '.dot',
                    # PowerPoint formats
                    '.ppt', '.pptx', '.pptm', '.potx', '.potm', '.ppsx', '.ppsm',
                    # Web & Markup formats  
                    '.html', '.htm', '.xml',
                    # OpenDocument formats
                    '.odt', '.ods', '.odp',
                    # Database formats
                    '.mdb', '.accdb',
                    # Communication formats
                    '.epub', '.msg', '.eml'
                ]:
                    elements = partition(filename=file_path)
                    content = "\n\n".join([str(element) for element in elements])
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": os.path.basename(file_path),
                            "file_type": file_extension,
                            "elements_count": len(elements)
                        }
                    )
                    documents.append(doc)
                    logger.info(f"Loaded {file_extension} file: {filename} ({len(elements)} elements)")
                    
                elif file_extension == '.txt':
                    loader = UTF8TextLoader(file_path)
                    file_docs = loader.load()
                    for doc in file_docs:
                        doc.metadata["source"] = os.path.basename(file_path)
                    documents.extend(file_docs)
                    logger.info(f"Loaded text file: {filename}")
                    
                elif file_extension == '.pdf':
                    if PDF_SUPPORT:
                        loader = PyPDFLoader(file_path)
                        file_docs = loader.load()
                        for doc in file_docs:
                            doc.metadata["source"] = os.path.basename(file_path)
                        documents.extend(file_docs)
                        logger.info(f"Loaded PDF: {filename} ({len(file_docs)} pages)")
                    else:
                        logger.error(f"PDF support not available")
                        
                elif file_extension in ['.doc', '.docx']:
                    if DOCX_SUPPORT:
                        loader = UnstructuredWordDocumentLoader(file_path)
                        file_docs = loader.load()
                        for doc in file_docs:
                            doc.metadata["source"] = os.path.basename(file_path)
                        documents.extend(file_docs)
                        logger.info(f"Loaded Word document: {filename}")
                    else:
                        logger.error(f"DOCX support not available")
                        
                else:
                    logger.warning(f"Unsupported file type: {file_extension} for file {filename}")
                    
            except Exception as e:
                error_msg = str(e).lower()
                
                # Detect encryption/password-protected files
                encryption_keywords = [
                    'encrypt', 'password', 'decrypt', 'protected', 
                    'badzipfile', 'pdfdecryptionerror', 'pdfreadeerror',
                    'bad magic number', 'file is encrypted', 'bad password',
                    'document is password protected', 'file has not been decrypted'
                ]
                
                if any(keyword in error_msg for keyword in encryption_keywords):
                    logger.error(f"❌ ENCRYPTED FILE: '{filename}' is password-protected or encrypted. Please provide an unprotected version.")
                else:
                    logger.error(f"Error processing file {filename}: {e}")
                
                continue
        
        logger.info(f"Successfully processed {len(documents)} documents from {len(file_paths)} files")
        return documents


    def _split_documents(self, documents: List) -> List:
        """Split documents into chunks using your existing approach"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        texts = text_splitter.split_documents(documents)
        return texts
    
    def _create_vector_store(self, texts: List, chroma_db_path: str, collection_name: str, user_id: str) -> Dict[str, Any]:
        """Create ChromaDB vector store with user isolation"""
        try:
            # Create embeddings
            embeddings = OpenAIEmbeddings(model=self.embeddings_model, openai_api_key=os.getenv('OPENAI_API_KEY'))
            logger.info(f"Created embeddings model: {self.embeddings_model}")
            
            # Use smart client selection
            db = self._get_chroma_client(path=chroma_db_path)
            
            # Create namespaced collection name for user isolation
            namespaced_name = self._get_namespaced_collection_name(user_id, collection_name)
            
            # Clean up existing user collection (only affects this user's data)
            if namespaced_name in [c.name for c in db.list_collections()]:
                db.delete_collection(name=namespaced_name)
                logger.info(f"Deleted existing user collection: {namespaced_name}")
            
            # Create new namespaced collection
            collection = db.get_or_create_collection(name=namespaced_name)
            
            # Extract content and metadata
            contents = [doc.page_content for doc in texts]
            metadatas = [doc.metadata for doc in texts]
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
            
            # Add documents to collection
            collection.add(
                documents=contents,
                metadatas=metadatas,
                ids=ids
            )
            
            # Verify collection
            count = collection.count()
            logger.info(f"Vector store created: {count} entries in collection '{namespaced_name}'")
            
            return {
                "success": True,
                "collection_count": count,
                "created_at": self._get_timestamp()
            }
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            return {
                "success": False,
                "error": f"Vector store creation failed: {str(e)}"
            }
    
    def _save_metadata(self, collection_path: str, metadata: Dict[str, Any]):
        """Save collection metadata"""
        import json
        from datetime import datetime
        
        metadata_file = os.path.join(collection_path, "knowledge_base_metadata.json")
        
        # Add creation timestamp
        metadata["kb_created_at"] = datetime.now().isoformat()
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved: {metadata_file}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_collection_info(self, user_id: str, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get information about user's collection"""
        try:
            # SECURITY: Sanitize inputs
            safe_user_id = sanitize_path_component(user_id)
            safe_collection_name = sanitize_path_component(collection_name)
            
            collection_path = create_safe_user_path(self.base_path, safe_user_id, safe_collection_name)
            metadata_file = validate_safe_path(collection_path, "knowledge_base_metadata.json")
            
            if os.path.exists(metadata_file):
                import json
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                return metadata
            
            return None
            
        except ValueError as ve:
            logger.error(f"Security validation failed: {ve}")
            return None
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return None
    
    def query_collection(
        self, 
        user_id: str, 
        collection_name: str, 
        query: str, 
        n_results: int = 5
    ) -> Dict[str, Any]:
        """Query user's knowledge base collection with user isolation"""
        try:
            # SECURITY: Sanitize inputs
            safe_user_id = sanitize_path_component(user_id)
            safe_collection_name = sanitize_path_component(collection_name)
            
            collection_path = create_safe_user_path(self.base_path, safe_user_id, safe_collection_name)
            chroma_db_path = validate_safe_path(collection_path, "chroma_db")
            
            # Use smart client selection
            db = self._get_chroma_client(path=chroma_db_path)
            
            # Use namespaced collection name for user isolation
            namespaced_name = self._get_namespaced_collection_name(safe_user_id, safe_collection_name)
            collection = db.get_collection(name=namespaced_name)
            
            # Perform query
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            logger.info(f"Query executed for user {safe_user_id}, collection {safe_collection_name}: {len(results['documents'][0])} results")
            
            return {
                "success": True,
                "query": query,
                "results": results['documents'][0] if results['documents'] else [],
                "metadatas": results['metadatas'][0] if results['metadatas'] else [],
                "distances": results['distances'][0] if results['distances'] else []
            }
            
        except ValueError as ve:
            logger.error(f"Security validation failed: {ve}")
            return {
                "success": False,
                "error": f"Invalid input: {str(ve)}",
                "results": []
            }
        except Exception as e:
            logger.error(f"Error querying collection: {e}")
            return {
                "success": False,
                "error": f"Query failed: {str(e)}",
                "results": []
            }

    def delete_file_from_collection(
        self, 
        user_id: str, 
        collection_name: str, 
        filename: str
    ) -> Dict[str, Any]:
        """Delete all chunks from a specific file"""
        try:
            collection_path = os.path.join(self.base_path, user_id, collection_name)
            chroma_db_path = os.path.join(collection_path, "chroma_db")
            
            db = self._get_chroma_client(path=chroma_db_path)
            namespaced_name = self._get_namespaced_collection_name(user_id, collection_name)
            collection = db.get_collection(name=namespaced_name)
            
            count_before = collection.count()
            
            # Get items to delete and delete by IDs
            items_to_delete = collection.get(where={"source": filename})
            if items_to_delete['ids']:
                collection.delete(ids=items_to_delete['ids'])
            
            count_after = collection.count()
            deleted_chunks = count_before - count_after
            
            logger.info(f"Deleted {deleted_chunks} chunks from file '{filename}'")
            
            self._update_metadata_after_deletion(collection_path, filename, deleted_chunks)
            
            return {
                "success": True,
                "deleted_file": filename,
                "deleted_chunks": deleted_chunks,
                "remaining_chunks": count_after
            }
            
        except Exception as e:
            logger.error(f"Error deleting file {filename}: {e}")
            return {
                "success": False,
                "error": f"Failed to delete file: {str(e)}"
            }

    def add_files_to_collection(
        self, 
        user_id: str, 
        collection_name: str, 
        file_paths: List[str]
    ) -> Dict[str, Any]:
        """Add new files to EXISTING collection"""
        try:
            logger.info(f"Adding {len(file_paths)} files to collection {collection_name}")
            
            collection_path = os.path.join(self.base_path, user_id, collection_name)
            chroma_db_path = os.path.join(collection_path, "chroma_db")
            
            # Check for duplicates
            metadata_file = os.path.join(collection_path, "knowledge_base_metadata.json")
            if os.path.exists(metadata_file):
                import json
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                existing_files = set(metadata.get("file_names", []))
                new_file_names = [os.path.basename(fp) for fp in file_paths]
                duplicates = [f for f in new_file_names if f in existing_files]
                
                if duplicates:
                    logger.warning(f"⚠️ Duplicate files detected: {duplicates}")
                    return {
                        "success": False,
                        "error": f"Files already exist: {', '.join(duplicates)}"
                    }
            
            # Load and process new files
            documents = self._load_documents_from_files(file_paths, collection_path)
            
            if not documents:
                return {
                    "success": False,
                    "error": "No documents could be loaded"
                }
            
            logger.info(f"Loaded {len(documents)} documents")
            
            # Split into chunks
            texts = self._split_documents(documents)
            logger.info(f"Split new files into {len(texts)} chunks")
            
            # Get existing collection
            db = self._get_chroma_client(path=chroma_db_path)
            namespaced_name = self._get_namespaced_collection_name(user_id, collection_name)
            collection = db.get_collection(name=namespaced_name)
            
            # Get current count
            current_count = collection.count()
            logger.info(f"Current collection has {current_count} chunks")
            
            # Prepare new data
            contents = [doc.page_content for doc in texts]
            metadatas = [doc.metadata for doc in texts]
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
            
            # Add to EXISTING collection
            collection.add(
                documents=contents,
                metadatas=metadatas,
                ids=ids
            )
            
            new_count = current_count + len(texts)
            
            # Update metadata JSON
            file_names = [os.path.basename(fp) for fp in file_paths]
            self._update_metadata_after_addition(collection_path, file_names, len(documents), len(texts))
            
            return {
                "success": True,
                "added_files": file_names,
                "added_chunks": len(texts),
                "total_chunks": new_count
            }
            
        except Exception as e:
            logger.error(f"Error adding files: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Failed to add files: {str(e)}"
            }

    def _update_metadata_after_deletion(self, collection_path: str, deleted_filename: str, deleted_chunks: int):
        """Update metadata JSON after file deletion"""
        try:
            import json
            from datetime import datetime
            metadata_file = os.path.join(collection_path, "knowledge_base_metadata.json")
            
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Update file list and counts
                if deleted_filename in metadata.get("file_names", []):
                    metadata["file_names"].remove(deleted_filename)
                    metadata["file_count"] = len(metadata["file_names"])
                
                metadata["chunk_count"] = metadata.get("chunk_count", 0) - deleted_chunks
                metadata["last_modified"] = datetime.now().isoformat()
                
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"Updated metadata after deletion")
        except Exception as e:
            logger.warning(f"Could not update metadata: {e}")

    def _update_metadata_after_addition(self, collection_path: str, new_filenames: List[str], doc_count: int, chunk_count: int):
        """Update metadata JSON after adding files"""
        try:
            import json
            from datetime import datetime
            metadata_file = os.path.join(collection_path, "knowledge_base_metadata.json")
            
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Update file list and counts
                existing_files = metadata.get("file_names", [])
                metadata["file_names"] = existing_files + new_filenames
                metadata["file_count"] = len(metadata["file_names"])
                metadata["document_count"] = metadata.get("document_count", 0) + doc_count
                metadata["chunk_count"] = metadata.get("chunk_count", 0) + chunk_count
                metadata["last_modified"] = datetime.now().isoformat()
                
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"Updated metadata after addition")
        except Exception as e:
            logger.warning(f"Could not update metadata: {e}")

# Initialize global knowledge base manager
kb_manager = KnowledgeBaseManager()

def create_knowledge_base(user_id: str, collection_name: str, file_paths: List[str]) -> Dict[str, Any]:
    """Convenience function for creating knowledge base"""
    return kb_manager.create_user_knowledge_base(user_id, collection_name, file_paths)

def query_knowledge_base(user_id: str, collection_name: str, query: str, n_results: int = 5) -> Dict[str, Any]:
    """Convenience function for querying knowledge base"""
    return kb_manager.query_collection(user_id, collection_name, query, n_results)

def get_user_collections(user_id: str) -> List[str]:
    """Get list of user's collections"""
    try:
        # SECURITY: Sanitize user_id
        safe_user_id = sanitize_path_component(user_id)
        
        user_path = create_safe_user_path("db_collection", safe_user_id)
        if os.path.exists(user_path):
            collections = [d for d in os.listdir(user_path) if os.path.isdir(os.path.join(user_path, d))]
            logger.info(f"Found {len(collections)} collections for user {safe_user_id}: {collections}")
            return collections
        else:
            logger.info(f"No directory found for user {safe_user_id}")
            return []
    except ValueError as ve:
        logger.error(f"Security validation failed: {ve}")
        return []
    except Exception as e:
        logger.error(f"Error getting collections for user {user_id}: {e}")
        return []

def get_display_collection_name(user_id: str, namespaced_name: str) -> str:
    """Convert namespaced collection name back to display name for UI"""
    prefix = f"{user_id}_"
    if namespaced_name.startswith(prefix):
        return namespaced_name[len(prefix):]
    return namespaced_name

def delete_file(user_id: str, collection_name: str, filename: str) -> Dict[str, Any]:
    """Convenience function for deleting a file"""
    return kb_manager.delete_file_from_collection(user_id, collection_name, filename)


def add_files(user_id: str, collection_name: str, file_paths: List[str]) -> Dict[str, Any]:
    """Convenience function for adding files to existing collection"""
    return kb_manager.add_files_to_collection(user_id, collection_name, file_paths)
