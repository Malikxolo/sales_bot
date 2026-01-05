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
from dotenv import load_dotenv
import aiohttp
from core.path_security import validate_safe_path, create_safe_user_path, sanitize_filename, sanitize_path_component
from core.google_drive_integration import render_multi_account_drive_picker, cleanup_multi_account_session
from core.organization_manager import (
    create_organization, 
    join_organization, 
    check_permission,
    get_organization,
    initialize_org_manager
)

org_manager = initialize_org_manager()

load_dotenv()

import requests



def upload_local_files(user_id: str, collection_name: str, file_paths: list[str]):
    """
    Uploads local files to the /api/collections/create/local endpoint.
    
    Args:
        user_id (str): The user ID.
        collection_name (str): The name of the collection.
        file_paths (list[str]): List of file paths to upload.

    Returns:
        dict: The JSON response from the server.
    """
    url = "http://localhost:8020/api/collections/create/local"

    
    form_data = {
        "user_id": user_id,
        "collection_name": collection_name,
    }

    
    files_data = []
    for file_path in file_paths:
        mime_type = "application/pdf" if file_path.endswith(".pdf") else "application/octet-stream"
        files_data.append(("files", (file_path.split("/")[-1], open(file_path, "rb"), mime_type)))

    headers = {
        "accept": "application/json",
    
    }

    response = requests.post(url, data=form_data, files=files_data, headers=headers)

    
    for _, file_tuple in files_data:
        file_tuple[1].close()

    return response.json()


async def mog_query(user_id: str, chat_history:List, query: str, source: str = "website"):
    url = "http://localhost:8020/api/chat"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    payload = {
        "userid": user_id,
        "chat_history": chat_history,
        "user_query": query,
        "source": source
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            response_data = await response.json()
            return response_data


# Initialize user_id FIRST (before logging)
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
    
if 'org_id' not in st.session_state:
    st.session_state.org_id = None
    
if 'org_role' not in st.session_state:
    st.session_state.org_role = None
    
if 'user_name' not in st.session_state:
    st.session_state.user_name = None
    
if 'show_create_org' not in st.session_state:
    st.session_state.show_create_org = False
    
if 'show_join_org' not in st.session_state:
    st.session_state.show_join_org = False
    
if 'show_transfer_admin' not in st.session_state:
    st.session_state.show_transfer_admin = False

if 'team_id' not in st.session_state:
    st.session_state.team_id = None  

if 'selected_team_id' not in st.session_state:
    st.session_state.selected_team_id = None  

if 'show_create_team_dialog' not in st.session_state:
    st.session_state.show_create_team_dialog = False
    
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

def sync_user_role():
    """Sync user role and team from organizations.json if in org"""
    if st.session_state.get('org_id'):
        org = get_organization(st.session_state.org_id)
        if org:
            user_id = get_user_id()
            members = org.get("members", {})
            
            if user_id in members:
                # Update session with latest role from JSON
                latest_role = members[user_id]["role"]
                if st.session_state.org_role != latest_role:
                    st.session_state.org_role = latest_role
                
                # UPDATE: Sync team_id
                latest_team_id = members[user_id].get("team_id")
                if st.session_state.team_id != latest_team_id:
                    st.session_state.team_id = latest_team_id
            else:
                # User removed from org - clear session
                st.session_state.org_id = None
                st.session_state.org_role = None
                st.session_state.user_name = None
                st.session_state.team_id = None  # ← ADD THIS
                st.session_state.selected_team_id = None  # ← ADD THIS



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
    from core.knowledge_base import get_active_collection
    return get_active_collection(user_id, namespaced_name)

def create_collection(user_id: str, collection_name: str, files: List):
    """Create new collection from uploaded files - ENHANCED VERSION WITH CLOUD CLEANUP"""
    try:
        from core.knowledge_base import get_kb_manager, create_knowledge_base
        kb_manager = get_kb_manager()
        
        safe_user_id = sanitize_path_component(user_id)
        safe_collection_name = sanitize_path_component(collection_name)
        
        
        user_path = create_safe_user_path("db_collection", safe_user_id)
        os.makedirs(user_path, exist_ok=True)
        
        
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

async def show_create_org_form():
    """Form to create new organization"""
    st.header("🏢 Create Organization")
    
    with st.form("create_org_form"):
        org_name = st.text_input(
            "Organization Name",
            placeholder="e.g., Acme Corporation"
        )
        
        user_name = st.text_input(
            "Your Name",
            placeholder="e.g., John Smith"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("Create", use_container_width=True)
        with col2:
            cancel = st.form_submit_button("Cancel", use_container_width=True)
        
        if cancel:
            st.session_state.show_create_org = False
            st.rerun()
        
        if submitted:
            if not org_name or not user_name:
                st.error("Please fill in all fields")
            else:
                user_id = get_user_id()
                result = await create_organization(org_name, user_name, user_id)
                
                if result["success"]:
                    st.session_state.org_id = result["org_id"]
                    st.session_state.org_role = result["role"]
                    st.session_state.user_name = user_name
                    st.session_state.show_create_org = False
                    
                    st.success(f"✅ Organization '{org_name}' created!")
                    st.info(f"🎟️ **Invite Code:** `{result['invite_code']}`")
                    st.caption("Share this code with your team")
                    
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error(f"❌ {result['error']}")


def show_join_org_form():
    """Form to join existing organization"""
    st.header("🔗 Join Organization")
    
    with st.form("join_org_form"):
        user_name = st.text_input(
            "Your Name",
            placeholder="e.g., Sarah Johnson"
        )
        
        invite_code = st.text_input(
            "Invite Code",
            placeholder="e.g., ACM5H2K9",
            max_chars=8
        ).upper()
        
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("Join", use_container_width=True)
        with col2:
            cancel = st.form_submit_button("Cancel", use_container_width=True)
        
        if cancel:
            st.session_state.show_join_org = False
            st.rerun()
        
        if submitted:
            if not user_name or not invite_code:
                st.error("Please fill in all fields")
            else:
                user_id = get_user_id()
                result = join_organization(invite_code, user_name, user_id)
                
                if result["success"]:
                    st.session_state.org_id = result["org_id"]
                    st.session_state.org_role = result["role"]
                    st.session_state.user_name = user_name
                    st.session_state.show_join_org = False
                    
                    st.success(f"✅ Joined '{result['org_name']}'!")
                    st.info(f"Your role: {result['role'].title()}")
                    
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error(f"❌ {result['error']}")


def upload_to_organization():
    """Upload documents to organization KB (owner/team admin)"""
    if not st.session_state.org_id:
        st.warning("Join an organization first")
        return
    
    user_id = get_user_id()
    
    # Determine upload target
    upload_target_id = None
    upload_team_name = None
    
    if check_permission(st.session_state.org_id, get_user_id(), "upload_documents"):
        # Owner selects team to upload to
        teams = org_manager.get_teams(st.session_state.org_id)
        if not teams:
            st.warning("⚠️ Create teams first in Team Management")
            if st.button("Go to Team Management"):
                st.session_state['show_team_management'] = True
                st.rerun()
            return
        
        st.subheader("📤 Upload to Team")
        team_options = {t["team_id"]: t["team_name"] for t in teams}
        
        selected_team = st.selectbox(
            "Select team:",
            list(team_options.keys()),
            format_func=lambda x: team_options[x],
            key="owner_upload_team_selector"
        )
        
        upload_target_id = f"{st.session_state.org_id}/{selected_team}"
        upload_team_name = team_options[selected_team]
        
    elif st.session_state.org_role == "team_admin":
        # Team admin uploads to their team
        if not st.session_state.team_id:
            st.error("You're not assigned to a team")
            return
        
        upload_target_id = f"{st.session_state.org_id}/{st.session_state.team_id}"
        
        # Get team name
        org = get_organization(st.session_state.org_id)
        teams = org.get("teams", {})
        if st.session_state.team_id in teams:
            upload_team_name = teams[st.session_state.team_id]["team_name"]
        else:
            upload_team_name = "Your Team"
        
        st.subheader(f"📤 Upload to {upload_team_name}")
    else:
        st.error("❌ Only owner and team admins can upload")
        return
    
    # Show target info
    st.info(f"📁 Uploading to: **{upload_team_name}**")
    
    collection_name = st.text_input(
        "Collection Name", 
        value="team_knowledge",
        key="org_collection_name"
    )
    
    uploaded_files = st.file_uploader(
        "Upload Documents",
        accept_multiple_files=True,
        type=["txt", "pdf", "docx", "doc", "md", "csv", "json"],
        key="org_file_uploader"
    )
    
    if uploaded_files and st.button("Upload to Organization", key="org_upload_btn"):
        with st.spinner(f"Uploading to {upload_team_name}..."):
            result = upload_local_files(
                user_id=upload_target_id,  # ← Uses org_id/team_id format!
                collection_name=collection_name,
                files=uploaded_files
            )
            
            if result.get("success"):
                st.success(f"✅ Uploaded {len(uploaded_files)} files to {upload_team_name}!")
                st.info(f"Team members of {upload_team_name} can now access these documents")
            else:
                st.error(f"❌ Upload failed: {result.get('error')}")

def render_rag_sidebar():
    """RAG Configuration - Fully integrated with organization features"""
    st.sidebar.markdown("---")
    
    # Determine mode and title
    if st.session_state.org_id:
        org = get_organization(st.session_state.org_id)
        if org:
            st.sidebar.subheader(f"📚 RAG ({org['org_name']})")
            st.sidebar.caption(f"Role: {st.session_state.org_role.title()}")
        else:
            st.sidebar.subheader("📚 RAG Configuration")
    else:
        st.sidebar.subheader("📚 RAG Configuration")
    
    # NEW - Match your UPLOAD and QUERY logic:
    if st.session_state.org_id:
        if st.session_state.team_id:
            # Team admin/member - use org_id/team_id
            target_id = f"{st.session_state.org_id}/{st.session_state.team_id}"
        elif st.session_state.org_role == "owner" and st.session_state.selected_team_id:
            # Owner with team selection
            target_id = f"{st.session_state.org_id}/{st.session_state.selected_team_id}"
        else:
            # No team context
            target_id = None  # Fallback to org level
    else:
        target_id = get_user_id()
    
    # Get collections using local file check (fast!)
    collections = get_user_collections(target_id)


    # Show collection status
    if collections:
        collection_name = collections[0]
        clean_name = display_collection_name(target_id, collection_name)
        mode_label = "org" if st.session_state.org_id else "personal"
        st.sidebar.success(f"✅ Active: {clean_name} ({mode_label})")
    else:
        if st.session_state.org_id:
            st.sidebar.info("No org collection loaded")
        else:
            st.sidebar.info("No collection loaded")

    # OWNER BUTTONS (Always visible - NOT dependent on collections)
    if st.session_state.org_id and check_permission(st.session_state.org_id, get_user_id(), "view_invite_code"):
        # CREATE TEAM - Always visible
        if st.sidebar.button("🏗️ Create Team", key="create_team_btn"):
            st.session_state['show_create_team_dialog'] = True
            st.rerun()
        
        # MANAGE TEAMS - Only if teams exist
        teams = org_manager.get_teams(st.session_state.org_id)
        if teams:
            if st.sidebar.button("👥 Manage Teams", key="manage_teams"):
                st.session_state['show_team_management'] = True
                st.rerun()

    # EDIT COLLECTION - Only if collection exists
    if collections:
        if st.session_state.org_id:
            if check_permission(st.session_state.org_id, get_user_id(), "edit_collection"):
                if st.sidebar.button("✏️ Edit Collection", key="edit_org"):
                    st.session_state['show_edit'] = True
                    st.rerun()
            else:
                st.sidebar.button("🔒 Edit (Owner/Team Admin Only)", disabled=True, key="edit_disabled")
        else:
            # Personal mode edit
            if st.sidebar.button("✏️ Edit Collection", key="edit_personal"):
                st.session_state['show_edit'] = True
                st.rerun()

    
    # Show org info if in organization
    if st.session_state.org_id:
        org = get_organization(st.session_state.org_id)
        if org:
            # Compact org details
            col1, col2 = st.sidebar.columns([2, 1])
            with col1:
                if check_permission(st.session_state.org_id, get_user_id(), "view_invite_code"):
                    with st.sidebar.expander("🔑 Invite Code"):
                        st.code(org["invite_code"])
                        st.caption("Share with team")
            with col2:
                member_count = org_manager.get_member_count(st.session_state.org_id)
                st.sidebar.write(f"👥 {member_count}")    
            # Show collection info
            if st.session_state.org_id:
                # Organization mode
                org = get_organization(st.session_state.org_id)
                if org:
                    org_name = org["org_name"]
                    st.sidebar.markdown(f"**🏢 {org_name}**")
                    st.sidebar.markdown(f"*Role: {st.session_state.org_role.title()}*")
                
                # ADD THIS: Team selector for owner
                if st.session_state.org_role == "owner":
                    org = get_organization(st.session_state.org_id)
                    teams = org_manager.get_teams(st.session_state.org_id)
                    
                    if teams:
                        st.sidebar.markdown("---")
                        st.sidebar.markdown("**Query Team:**")
                        
                        team_options = {team["team_id"]: team["team_name"] for team in teams}
                        team_ids = list(team_options.keys())
                        team_names = list(team_options.values())
                        
                        # Default to first team if not selected
                        if not st.session_state.selected_team_id and team_ids:
                            st.session_state.selected_team_id = team_ids[0]
                        
                        selected_index = team_ids.index(st.session_state.selected_team_id) if st.session_state.selected_team_id in team_ids else 0
                        
                        selected_team = st.sidebar.selectbox(
                            "Select team to query:",
                            team_names,
                            index=selected_index,
                            key="team_selector"
                        )
                        
                        # Update selected team
                        st.session_state.selected_team_id = team_ids[team_names.index(selected_team)]
                        
                        st.sidebar.info(f"✅ Querying: {selected_team}")
                    else:
                        st.sidebar.warning("No teams created yet")
                elif st.session_state.team_id:
                    # Team member/admin - show their team
                    org = get_organization(st.session_state.org_id)
                    teams = org.get("teams", {})
                    if st.session_state.team_id in teams:
                        team_name = teams[st.session_state.team_id]["team_name"]
                        st.sidebar.info(f"📁 Team: {team_name}")
                else:
                    # Unassigned member
                    st.sidebar.warning("⏳ Not assigned to a team")
                    
            # Leave organization button (different for admin vs viewer)
            if st.session_state.org_role == "owner":
                if st.sidebar.button("← Leave Organization", key="leave_org_admin"):
                    st.session_state['show_transfer_admin'] = True
                    st.rerun()
            else:
                # Viewer can leave directly
                if st.sidebar.button("← Leave Organization", key="leave_org_viewer"):
                    org_manager.remove_member(st.session_state.org_id, get_user_id())
                    st.session_state.org_id = None
                    st.session_state.org_role = None
                    st.session_state.user_name = None
                    st.success("Left organization")
                    time.sleep(1)
                    st.rerun()

            
            st.sidebar.markdown("---")
    
    # THREE MAIN BUTTONS (context-aware)
    # Button 1: Create Collection
    if st.session_state.org_id:
        # In organization mode
        if check_permission(st.session_state.org_id, get_user_id(), "upload_documents"):
            # Check if teams exist
            teams = org_manager.get_teams(st.session_state.org_id)
            
            if teams:
                # Teams exist - show enabled button
                button_label = "📤 Create Collection (Org)"
                button_disabled = False
                button_key = "create_org_collection"
            else:
                # No teams - show disabled button with message
                button_label = "📤 Create Collection (Create teams first)"
                button_disabled = True
                button_key = "create_disabled_noteams"
        else:
            button_label = "🔒 Create Collection (Owner/Team Admin Only)"
            button_disabled = True
            button_key = "create_disabled_viewer"


    else:
        # Personal mode
        button_label = "📁 Create Collection"
        button_disabled = False
        button_key = "create_personal_collection"
    
    if st.sidebar.button(button_label, disabled=button_disabled, key=button_key, use_container_width=True):
        st.session_state['show_upload'] = True
        st.rerun()
    
    # Button 2 & 3: Create/Join Organization (only if NOT in org)
    if not st.session_state.org_id:
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("🏢 Create Org", key="create_org_sidebar", use_container_width=True):
                st.session_state.show_create_org = True
                st.rerun()
        
        with col2:
            if st.button("🔗 Join Org", key="join_org_sidebar", use_container_width=True):
                st.session_state.show_join_org = True
                st.rerun()

# Import core system
try:
    from core import (
        Config, LLMClient, 
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


async def create_agents_async(config, brain_agent_config, heart_agent_config, router_agent_config, web_model_config, use_premium_search=False):
    """Create Optimized Agent with selected models"""
    try:
        # Create LLM clients
        brain_llm = LLMClient(brain_agent_config)
        heart_llm = LLMClient(heart_agent_config)
        router_llm = LLMClient(router_agent_config)
        
        # Create tool manager
        tool_manager = ToolManager(config, brain_llm, web_model_config, use_premium_search)
        
        return {
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


async def process_query_real(query: str, style: str, user_id: str = None, chat_history: List = None, source: str = "website") -> Dict[str, Any]:
    """Process query through Optimized Agent"""
    
    try:
        start_time = time.time()
        
        # Single agent processing
        st.info("🚀 Optimized Agent processing query...")
        result = await mog_query(user_id, chat_history, query, source)
        
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

def show_team_management_panel():
    """Team management panel for organization owner"""
    st.markdown("# 👥 Team Management")
    
    org = get_organization(st.session_state.org_id)
    if not org:
        st.error("Organization not found")
        return
    
    user_id = get_user_id()
    
    # Check if owner
    if org["owner_id"] != user_id:
        st.error("Only owner can manage teams")
        return
    
    # Close button
    if st.button("← Back to Chat"):
        st.session_state['show_team_management'] = False
        st.rerun()
    
    st.markdown("---")
    
    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📋 Teams", "👥 Unassigned Members", "🗑️ Delete Team"])
    
    # TAB 1: Create & Manage Teams
    with tab1:
        st.markdown("### Create New Team")
        with st.form("create_team_form"):
            team_name = st.text_input("Team Name", placeholder="DevOps Team")
            submit = st.form_submit_button("Create Team")
            
            if submit and team_name:
                result = org_manager.create_team(
                    st.session_state.org_id,
                    team_name,
                    user_id
                )
                if result["success"]:
                    st.success(f"✅ Team '{team_name}' created!")
                    st.rerun()
                else:
                    st.error(result["error"])
        
        st.markdown("---")
        st.markdown("### Existing Teams")
        
        teams = org_manager.get_teams(st.session_state.org_id)
        if teams:
            for team in teams:
                with st.expander(f"📁 {team['team_name']}", expanded=False):
                    team_members = org_manager.get_team_members(
                        st.session_state.org_id,
                        team["team_id"]
                    )
                    
                    # Team admin info
                    team_admin_id = team.get("team_admin_id")
                    if team_admin_id:
                        admin_name = next(
                            (m["name"] for m in team_members if m["user_id"] == team_admin_id),
                            "Unknown"
                        )
                        st.info(f"👑 Admin: {admin_name}")
                    else:
                        st.warning("⚠️ No team admin assigned")
                    
                    st.markdown(f"**Members ({len(team_members)}):**")
                    
                    if team_members:
                        for member in team_members:
                            col1, col2, col3 = st.columns([3, 2, 1])
                            with col1:
                                role_icon = "👑" if member["role"] == "team_admin" else "👤"
                                st.write(f"{role_icon} {member['name']}")
                            with col2:
                                st.caption(member['role'].replace('_', ' ').title())
                            with col3:
                                if st.button("Remove", key=f"remove_{team['team_id']}_{member['user_id']}"):
                                    result = org_manager.remove_member_from_team(
                                        st.session_state.org_id,
                                        team["team_id"],
                                        member["user_id"],
                                        user_id
                                    )
                                    if result["success"]:
                                        st.success("Member removed")
                                        st.rerun()
                                    else:
                                        st.error(result["error"])
                    else:
                        st.caption("No members yet")
                    
                    # Assign team admin
                    st.markdown("---")
                    st.markdown("**Assign Team Admin:**")
                    if team_members:
                        member_options = {m["user_id"]: m["name"] for m in team_members}
                        selected_admin = st.selectbox(
                            "Choose admin:",
                            list(member_options.keys()),
                            format_func=lambda x: member_options[x],
                            key=f"admin_select_{team['team_id']}"
                        )
                        if st.button("Set as Admin", key=f"set_admin_{team['team_id']}"):
                            result = org_manager.assign_team_admin(
                                st.session_state.org_id,
                                team["team_id"],
                                selected_admin,
                                user_id
                            )
                            if result["success"]:
                                st.success("Team admin assigned!")
                                st.rerun()
                            else:
                                st.error(result["error"])
        else:
            st.info("No teams created yet")
    
    # TAB 2: Unassigned Members
    with tab2:
        st.markdown("### Unassigned Members")
        unassigned = org_manager.get_unassigned_members(st.session_state.org_id)
        teams = org_manager.get_teams(st.session_state.org_id)
        
        if unassigned and teams:
            for member in unassigned:
                with st.container():
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"👤 {member['name']}")
                    with col2:
                        team_options = {t["team_id"]: t["team_name"] for t in teams}
                        selected_team = st.selectbox(
                            "Assign to:",
                            list(team_options.keys()),
                            format_func=lambda x: team_options[x],
                            key=f"assign_{member['user_id']}"
                        )
                        if st.button("Assign", key=f"assign_btn_{member['user_id']}"):
                            result = org_manager.add_member_to_team(
                                st.session_state.org_id,
                                selected_team,
                                member["user_id"],
                                user_id
                            )
                            if result["success"]:
                                st.success("Member assigned!")
                                st.rerun()
                            else:
                                st.error(result["error"])
                    st.markdown("---")
        elif not teams:
            st.warning("Create teams first")
        else:
            st.info("All members are assigned to teams")
    
    # TAB 3: Delete Team
    with tab3:
        st.markdown("### Delete Team")
        st.warning("⚠️ Teams can only be deleted if they have no members")
        
        teams = org_manager.get_teams(st.session_state.org_id)
        if teams:
            team_options = {t["team_id"]: t["team_name"] for t in teams}
            selected_team = st.selectbox(
                "Select team to delete:",
                list(team_options.keys()),
                format_func=lambda x: team_options[x],
                key="delete_team_select"
            )
            
            # Show team member count
            team_members = org_manager.get_team_members(
                st.session_state.org_id,
                selected_team
            )
            st.info(f"Members in team: {len(team_members)}")
            
            if st.button("🗑️ Delete Team", type="primary"):
                result = org_manager.delete_team(
                    st.session_state.org_id,
                    selected_team,
                    user_id
                )
                if result["success"]:
                    st.success("Team deleted!")
                    st.rerun()
                else:
                    st.error(result["error"])
        else:
            st.info("No teams to delete")

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
    
    # Sync user role with JSON file
    sync_user_role()
    
    # Show org forms if active
    if st.session_state.get("show_create_org"):
        show_create_org_form()
        st.stop()

    if st.session_state.get("show_join_org"):
        show_join_org_form()
        st.stop()
    # CREATE TEAM DIALOG
    if st.session_state.get("show_create_team_dialog"):
        st.markdown("## 🏗️ Create New Team")
        
        with st.form("quick_create_team_form"):
            team_name = st.text_input("Team Name", placeholder="e.g., DevOps Team")
            
            col1, col2 = st.columns(2)
            with col1:
                submit = st.form_submit_button("✅ Create Team", use_container_width=True)
            with col2:
                cancel = st.form_submit_button("❌ Cancel", use_container_width=True)
            
            if cancel:
                st.session_state['show_create_team_dialog'] = False
                st.rerun()
            
            if submit and team_name:
                result = org_manager.create_team(
                    st.session_state.org_id,
                    team_name,
                    get_user_id()
                )
                
                if result["success"]:
                    st.success(f"✅ Team '{team_name}' created successfully!")
                    st.session_state['show_create_team_dialog'] = False
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"❌ {result['error']}")
        
        st.stop()

    # Show team management panel
    if st.session_state.get("show_team_management"):
        show_team_management_panel()
        st.stop()
        
    # TRANSFER ADMIN ROLE DIALOG
    if st.session_state.get('show_transfer_admin'):
        st.markdown("---")
        
        org = get_organization(st.session_state.org_id)
        members = org.get("members", {})
        
        # Get other members (exclude current admin)
        current_user_id = get_user_id()
        other_members = {uid: info for uid, info in members.items() if uid != current_user_id}
        
        if not other_members:
            # Admin is alone - allow delete
            st.warning("⚠️ You are the only member in this organization")
            st.markdown("**You can:**")
            st.markdown("- Delete the organization (will remove all data)")
            st.markdown("- Cancel and stay as admin")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Delete Organization", key="delete_org_solo"):
                    org_id = st.session_state.org_id
                    
                    # Delete org files
                    import shutil
                    org_path = f"db_collection/{org_id}"
                    if os.path.exists(org_path):
                        shutil.rmtree(org_path)
                    
                    # Delete from JSON
                    org_manager.delete_organization(org_id)
                    
                    # Clear session
                    st.session_state.org_id = None
                    st.session_state.org_role = None
                    st.session_state.user_name = None
                    st.session_state['show_transfer_admin'] = False
                    
                    st.success("✅ Organization deleted")
                    time.sleep(2)
                    st.rerun()
            
            with col2:
                if st.button("❌ Cancel", key="cancel_delete_solo"):
                    st.session_state['show_transfer_admin'] = False
                    st.rerun()
        
        else:
            # Has members - must transfer admin
            st.info(f"👥 Transfer admin role to continue")
            st.markdown(f"**Organization:** {org['org_name']}")
            st.markdown(f"**Members:** {len(other_members)}")
            
            # Create member list for selection
            member_options = []
            member_ids = []
            for uid, info in other_members.items():
                member_options.append(f"{info['name']} (Viewer)")
                member_ids.append(uid)
            
            selected = st.selectbox("Select new admin:", member_options, key="select_new_admin")
            selected_index = member_options.index(selected)
            new_admin_id = member_ids[selected_index]
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("✅ Transfer & Leave", key="confirm_transfer"):
                    # Transfer admin role
                    org_manager.transfer_admin(
                        st.session_state.org_id, 
                        current_user_id, 
                        new_admin_id
                    )
                    
                    # Clear session
                    st.session_state.org_id = None
                    st.session_state.org_role = None
                    st.session_state.user_name = None
                    st.session_state['show_transfer_admin'] = False
                    
                    new_admin_name = selected.split(' (')[0]
                    st.success(f"✅ Admin transferred to {new_admin_name}")
                    st.info("You have left the organization")
                    time.sleep(2)
                    st.rerun()
            
            with col2:
                if st.button("❌ Cancel", key="cancel_transfer"):
                    st.session_state['show_transfer_admin'] = False
                    st.rerun()
        
        st.stop()

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
        # Determine mode and target
        if st.session_state.org_id:
            if st.session_state.team_id:
                target_id = f"{st.session_state.org_id}/{st.session_state.team_id}"
            elif st.session_state.org_role == "owner" and st.session_state.selected_team_id:
                target_id = f"{st.session_state.org_id}/{st.session_state.selected_team_id}"
            else:
                target_id = None
            mode = "Organization"
            
            # Permission check
            if not check_permission(st.session_state.org_id, get_user_id(), "upload_documents"):
                st.error("⛔ Only owner and team admins can edit organization collections")
                if st.button("Close", key="close_upload_error"):
                    st.session_state['show_upload'] = False
                    st.rerun()
                st.stop()
            
            st.markdown(f"## 📤 Create {mode} Collection")
            org = get_organization(st.session_state.org_id)
            st.info(f"Creating for: **{org['org_name']}**")
        else:
            target_id = get_user_id()
            mode = "Personal"
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
                                res = upload_local_files(target_id, collection_name, uploaded_files)
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
                                            result = create_knowledge_base(target_id, collection_name, downloaded_files)
                                        
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
        # Determine target (org or personal)
        if st.session_state.org_id:
            if st.session_state.team_id:
                target_id = f"{st.session_state.org_id}/{st.session_state.team_id}"
            elif st.session_state.org_role == "owner" and st.session_state.selected_team_id:
                target_id = f"{st.session_state.org_id}/{st.session_state.selected_team_id}"
            else:
                target_id = None
            mode = "Organization"
            
            # Permission check
            if not check_permission(st.session_state.org_id, get_user_id(), "edit_collection"):
                st.error("⛔ Only owner and team admins can upload to organization")
                if st.button("Close", key="close_edit_error"):
                    st.session_state['show_edit'] = False
                    st.rerun()
                st.stop()
            
            st.markdown(f"## ✏️ Edit {mode} Collection")
            org = get_organization(st.session_state.org_id)
            st.info(f"Editing: **{org['org_name']}** collection")
        else:
            target_id = get_user_id()
            mode = "Personal"
            st.markdown("## ✏️ Edit Collection")
        
        collections = get_user_collections(target_id)
        
        if collections:
            collection_name = collections[0]
            
            # Get metadata
            try:
                metadata_path = f"db_collection/{target_id}/{collection_name}/knowledge_base_metadata.json"
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
                                    result = delete_file(target_id, collection_name, filename)
                                    
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
                            result = add_files(target_id, collection_name, temp_paths)
                            
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
                # Create router config using brain provider/model with minimal tokens
                router_model_config = config.create_llm_config(
                    provider=brain_model_config.provider,
                    model=brain_model_config.model,
                    max_tokens=1000
                )
                # Run async agent creation and processing
                agents_result = asyncio.run(create_agents_async(config, brain_model_config, heart_model_config, router_model_config, web_model_config, use_premium_search))
                
                if agents_result["status"] == "error":
                    st.error(f"❌ Agent creation failed: {agents_result['error']}")
                    logger.error(f"Agent creation failed: {agents_result}")
                    if show_debug:
                        st.json(agents_result)
                else:
                    # Process query with optimized agent
                    
                    # Determine query target based on role and team
                    if st.session_state.org_id:
                        # Organization mode
                        if st.session_state.org_role == "owner":
                            # Owner uses selected_team_id from dropdown
                            if st.session_state.selected_team_id:
                                query_target_id = f"{st.session_state.org_id}/{st.session_state.selected_team_id}"
                            else:
                                query_target_id = f"viewer_{st.session_state.org_id}_{get_user_id()}" # No team selected - RAG will fail gracefully
                        elif st.session_state.team_id:
                            # Team admin/member uses their assigned team
                            query_target_id = f"{st.session_state.org_id}/{st.session_state.team_id}"
                        else:
                            # Viewer with no team
                            query_target_id = f"viewer_{st.session_state.org_id}_{get_user_id()}" # RAG will fail gracefully
                    else:
                        # Personal mode
                        query_target_id = get_user_id()

                                
                    result = asyncio.run(process_query_real(
                        query, 
                        style,
                        user_id=query_target_id,
                        chat_history=st.session_state.chat_history,
                        source="website" 
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