"""
Google Drive Integration with Multi-Account Session Token Management
Professional UI with enhanced security and file conflict handling
SECURITY: Path sanitization to prevent directory traversal attacks
"""

import os
import shutil
import logging
from typing import List, Dict, Any, Optional
import streamlit as st
from core.session_manager import SessionTokenManager, MultiAccountManager
from core.path_security import (
    sanitize_path_component,
    sanitize_filename,
    validate_safe_path,
    create_safe_user_path
)
import uuid
import hashlib

# Google Drive API imports
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    import io
    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False

logger = logging.getLogger(__name__)


SCOPES = [
    'https://www.googleapis.com/auth/drive.readonly',
    'https://www.googleapis.com/auth/userinfo.email',
    'openid'
]

class MultiAccountGoogleDriveManager:
    """Multi-account Google Drive manager with enhanced security and file conflict handling"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.credentials_file = "credentials.json"
        self.base_temp_path = "temp_drive_files"
        self.account_manager = MultiAccountManager(user_id)
        
        logger.debug(f"🚀 MultiAccountGoogleDriveManager initialized for user: {user_id}")

    
    def is_available(self) -> bool:
        """Check if Google Drive integration is available"""
        return GOOGLE_DRIVE_AVAILABLE and os.path.exists(self.credentials_file)
    
    def authenticate_account(self, account_alias: str = None) -> Optional[str]:
        """Authenticate new Google account and return account_id"""
        try:
            # Generate unique account ID
            account_id = str(uuid.uuid4())[:8]
            account_alias = account_alias or f"Account {len(self.account_manager.get_user_accounts()) + 1}"
            
            logger.info(f"🔐 Starting authentication for new account: {account_alias}")
            
            # Start OAuth flow
            flow = InstalledAppFlow.from_client_secrets_file(self.credentials_file, SCOPES)
            creds = flow.run_local_server(port=8080)
            
            # Get user email from credentials
            try:
                import jwt
                token_info = jwt.decode(creds.id_token, options={"verify_signature": False})
                account_email = token_info.get('email', 'unknown@gmail.com')
            except:
                account_email = 'unknown@gmail.com'
            
            # Create session
            session_mgr = SessionTokenManager(self.user_id, account_id)
            session_id = session_mgr.create_session(creds, account_email, account_alias)
            
            if session_id:
                # Add to account index
                self.account_manager.add_account(account_id, account_email, account_alias)
                logger.info(f"✅ Successfully authenticated account: {account_email}")
                return account_id
            
            return None
            
        except Exception as e:
            logger.error(f"💥 Authentication failed: {e}")
            return None
    
    def get_account_service(self, account_id: str) -> Optional[Any]:
        """Get Google Drive service for specific account"""
        try:
            session_mgr = SessionTokenManager(self.user_id, account_id)
            creds = session_mgr.get_google_credentials()
            
            if creds:
                return build('drive', 'v3', credentials=creds)
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get service for account {account_id}: {e}")
            return None
    
    def list_files_for_account(self, account_id: str, file_type: str = None) -> List[Dict[str, Any]]:
        """List files from specific Google Drive account"""
        try:
            service = self.get_account_service(account_id)
            if not service:
                return []
            
            query = "trashed=false"
            
            if file_type == 'documents':
                mime_types = [
                    'application/pdf',
                    'application/msword',
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    'text/plain',
                    'text/csv',
                    'application/vnd.ms-excel',
                    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                ]
                mime_query = " or ".join([f"mimeType='{mime}'" for mime in mime_types])
                query += f" and ({mime_query})"
            
            results = service.files().list(
                q=query,
                pageSize=50,
                fields="files(id, name, mimeType, size, modifiedTime)"
            ).execute()
            
            files = results.get('files', [])
            # Tag files with account info
            for file in files:
                file['account_id'] = account_id
            
            logger.info(f"📂 Found {len(files)} files for account {account_id}")
            return files
            
        except Exception as e:
            logger.error(f"❌ Failed to list files for account {account_id}: {e}")
            return []
    
    def download_files_with_conflict_resolution(self, selected_files: List[Dict[str, Any]]) -> List[str]:
        """Download files from multiple accounts with conflict resolution"""
        try:
            # SECURITY: Sanitize user_id for path creation
            safe_user_id = sanitize_path_component(self.user_id)
            
            # Create user-specific temp directory with secure path
            user_temp_dir = create_safe_user_path(self.base_temp_path, safe_user_id)
            os.makedirs(user_temp_dir, exist_ok=True)
            
            downloaded_files = []
            filename_counts = {}  # Track filename conflicts
            
            for file_info in selected_files:
                try:
                    account_id = file_info['account_id']
                    file_id = file_info['id']
                    original_filename = file_info['name']
                    
                    service = self.get_account_service(account_id)
                    if not service:
                        continue
                    
                    # Handle filename conflicts
                    safe_filename = self._resolve_filename_conflict(
                        original_filename, account_id, filename_counts
                    )
                    filename_counts[original_filename] = filename_counts.get(original_filename, 0) + 1
                    
                    # SECURITY: Validate the final file path
                    file_path = validate_safe_path(user_temp_dir, safe_filename)
                    
                    # Download file
                    request = service.files().get_media(fileId=file_id)
                    
                    with open(file_path, 'wb') as f:
                        downloader = MediaIoBaseDownload(f, request)
                        done = False
                        while done is False:
                            status, done = downloader.next_chunk()
                    
                    downloaded_files.append(file_path)
                    logger.info(f"📥 Downloaded {safe_filename} from account {account_id}")
                    
                except ValueError as ve:
                    logger.error(f"❌ Security validation failed for {file_info.get('name', 'unknown')}: {ve}")
                    continue
                except Exception as e:
                    logger.error(f"❌ Failed to download file {file_info.get('name', 'unknown')}: {e}")
                    continue
            
            return downloaded_files
            
        except ValueError as ve:
            logger.error(f"❌ Security validation failed for user {self.user_id}: {ve}")
            return []
        except Exception as e:
            logger.error(f"❌ Download failed for user {self.user_id}: {e}")
            return []
    
    def _resolve_filename_conflict(self, filename: str, account_id: str, filename_counts: Dict[str, int]) -> str:
        """Resolve filename conflicts by prefixing with account info"""
        # SECURITY: Sanitize filename first
        safe_filename = sanitize_filename(filename)
        
        if filename not in filename_counts:
            return safe_filename
        
        # Get file extension
        name, ext = os.path.splitext(safe_filename)
        
        # SECURITY: Sanitize account_id
        safe_account_id = sanitize_path_component(account_id)
        
        # Add account prefix to avoid conflicts
        prefixed_filename = f"acc{safe_account_id}_{name}{ext}"
        return prefixed_filename
    
    def cleanup_temp_files(self, full_cleanup: bool = False):
        """Clean up temp files and optionally sessions"""
        try:
            # SECURITY: Sanitize user_id
            safe_user_id = sanitize_path_component(self.user_id)
            
            # Clean user's temp directory with secure path
            user_temp_dir = create_safe_user_path(self.base_temp_path, safe_user_id)
            if os.path.exists(user_temp_dir):
                shutil.rmtree(user_temp_dir)
                logger.info(f"🗑️ Deleted temp directory for user {safe_user_id}")
            
            if full_cleanup:
                # REVOKE ALL TOKENS BEFORE CLEANUP (ENHANCED)
                revoked_count = self.account_manager.revoke_all_accounts()
                logger.info(f"🔒 Revoked {revoked_count} accounts during cleanup")
                
                # Clean up accounts index with secure path
                user_sessions_dir = validate_safe_path("sessions", safe_user_id)
                if os.path.exists(user_sessions_dir):
                    shutil.rmtree(user_sessions_dir)
                
                logger.info(f"🗑️ Full cleanup completed for user {safe_user_id}")
                
        except ValueError as ve:
            logger.error(f"❌ Security validation failed during cleanup: {ve}")
        except Exception as e:
            logger.error(f"❌ Cleanup failed for user {self.user_id}: {e}")
    
    def clear_streamlit_sessions(self):
        """Clear all Streamlit session data for this user"""
        keys_to_remove = []
        for key in st.session_state.keys():
            if key.startswith(f'drive_') and self.user_id in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del st.session_state[key]
        
        logger.info(f"🧹 Cleared Streamlit sessions for user {self.user_id}")
        
    def revoke_account_access(self, account_id: str) -> bool:
        """Immediately revoke Google access for specific account"""
        try:
            session_mgr = SessionTokenManager(self.user_id, account_id)
            success = session_mgr.revoke_google_tokens()
            
            if success:
                # Remove from account index
                self.account_manager.remove_account(account_id)
                
                # Clear Streamlit session data
                keys_to_remove = [key for key in st.session_state.keys() if account_id in key]
                for key in keys_to_remove:
                    del st.session_state[key]
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to revoke account {account_id}: {e}")
            return False
    
    def security_disconnect_all(self):
        """SECURITY FEATURE: Immediately revoke ALL accounts and clean up everything"""
        try:
            revoked_count = self.account_manager.revoke_all_accounts()
            
            # Clean up temp files
            self.cleanup_temp_files(full_cleanup=True)
            
            # Clear all Streamlit sessions
            self.clear_streamlit_sessions()
            
            logger.info(f"🔒 Security disconnect completed: {revoked_count} accounts revoked")
            return revoked_count
            
        except Exception as e:
            logger.error(f"❌ Security disconnect failed: {e}")
            return 0


def render_professional_ui():
    """Add professional CSS styling"""
    st.markdown("""
    <style>
    /* Account Cards */
    .account-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .account-card.inactive {
        background: linear-gradient(135deg, #bdc3c7 0%, #95a5a6 100%);
    }
    
    .account-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    
    .account-email {
        font-size: 16px;
        font-weight: 600;
    }
    
    .account-status {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
    }
    
    .status-active {
        background: #27ae60;
    }
    
    .status-expired {
        background: #e74c3c;
    }
    
    /* File Browser */
    .file-browser {
        background: white;
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid #e1e8ed;
    }
    
    .file-item {
        display: flex;
        align-items: center;
        padding: 12px;
        border-bottom: 1px solid #f0f2f6;
        transition: background-color 0.2s;
    }
    
    .file-item:hover {
        background-color: #f8f9fa;
    }
    
    .file-icon {
        margin-right: 12px;
        font-size: 20px;
    }
    
    /* Buttons */
    .custom-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 8px;
        color: white;
        padding: 12px 24px;
        font-weight: 600;
        cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .custom-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    /* Progress Bars */
    .progress-container {
        background: #f0f2f6;
        border-radius: 10px;
        overflow: hidden;
        height: 8px;
        margin: 10px 0;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        transition: width 0.3s ease;
    }
    
    /* Stats Cards */
    .stats-card {
        background: white;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        border: 1px solid #e1e8ed;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .stats-number {
        font-size: 24px;
        font-weight: 700;
        color: #667eea;
    }
    
    .stats-label {
        font-size: 12px;
        color: #657786;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    </style>
    """, unsafe_allow_html=True)


def render_multi_account_drive_picker(user_id: str, download_files: bool = True) -> Optional[List[str]]:
    """Professional multi-account Google Drive picker"""
    
    # Apply professional styling
    render_professional_ui()
    
    drive_manager = MultiAccountGoogleDriveManager(user_id)
    
    if not drive_manager.is_available():
        st.error("❌ Google Drive not configured")
        return None
    
    st.markdown("# 📁 Google Drive File Manager")
    st.markdown("Connect multiple Google Drive accounts and manage files from a unified interface.")
    
    # Get user accounts
    accounts = drive_manager.account_manager.get_user_accounts()
    
    # Account Management Section
    st.markdown("## 🔗 Connected Accounts")
    
    if accounts:
        # Display account cards
        for i, account in enumerate(accounts):
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    status_class = "active" if account['is_active'] else "inactive"
                    status_text = "🟢 Active" if account['is_active'] else "🔴 Expired"
                    
                    st.markdown(f"""
                    <div class="account-card {status_class}">
                        <div class="account-header">
                            <div class="account-email">{account.get('account_email', 'Unknown Email')}</div>
                            <div class="account-status status-{'active' if account['is_active'] else 'expired'}">{status_text}</div>
                        </div>
                        <div>Alias: {account.get('account_alias', f'Account {i+1}')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if not account['is_active']:
                        if st.button(f"🔄 Reconnect", key=f"reconnect_{account['account_id']}"):
                            # Remove old session and reconnect
                            drive_manager.account_manager.remove_account(account['account_id'])
                            new_account_id = drive_manager.authenticate_account(account['account_alias'])
                            if new_account_id:
                                st.success(f"✅ Reconnected {account['account_alias']}")
                                st.rerun()
                            else:
                                st.error("❌ Reconnection failed")
                
                with col3:
                    if st.button(f"🗑️ Remove", key=f"remove_{account['account_id']}"):
                        # REVOKE TOKENS BEFORE REMOVAL (NEW)
                        drive_manager.revoke_account_access(account['account_id'])
                        
                        # Clear related Streamlit session data
                        keys_to_remove = [key for key in st.session_state.keys() if account['account_id'] in key]
                        for key in keys_to_remove:
                            del st.session_state[key]
                        st.success(f"Removed and revoked {account['account_alias']}")
                        st.rerun()
    # Add security disconnect button
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔒 Security Disconnect All", type="secondary", help="Immediately revoke ALL Google access"):
            # Use session state for confirmation
            if 'confirm_disconnect_all' not in st.session_state:
                st.session_state.confirm_disconnect_all = False

            if not st.session_state.confirm_disconnect_all:
                st.session_state.confirm_disconnect_all = True
                st.error("⚠️ This will immediately revoke ALL Google Drive access!")
                st.info("Click the button again to confirm.")
                st.rerun()
            else:
                # User confirmed, proceed
                st.session_state.confirm_disconnect_all = False
            drive_manager = MultiAccountGoogleDriveManager(user_id)
            revoked_count = drive_manager.security_disconnect_all()
            
            if revoked_count > 0:
                st.success(f"✅ Successfully revoked {revoked_count} Google accounts")
                st.info("🛡️ All tokens have been immediately invalidated at Google")
                st.rerun()
            else:
                st.warning("No active accounts to revoke")
    
    with col2:
        # Individual account revoke in the remove button
        # (Already implemented in your existing remove logic)
        pass
    
    # Add new account button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("➕ Add Google Account", type="primary"):
            with st.spinner("🔐 Authenticating new account..."):
                # Auto-generate account alias without user input
                account_count = len(accounts) + 1
                account_alias = f"Account {account_count}"
                
                account_id = drive_manager.authenticate_account(account_alias)
                if account_id:
                    st.success("✅ Account added successfully!")
                    st.rerun()
                else:
                    st.error("❌ Authentication failed")
    
    # File Browser Section
    if accounts and any(acc['is_active'] for acc in accounts):
        st.markdown("---")
        st.markdown("## 📂 File Browser")
        
        # Load files from all active accounts
        all_files = []
        active_accounts = [acc for acc in accounts if acc['is_active']]
        
        for account in active_accounts:
            account_id = account['account_id']
            cache_key = f'drive_files_{user_id}_{account_id}'
            
            if cache_key not in st.session_state:
                with st.spinner(f"Loading files from {account['account_alias']}..."):
                    files = drive_manager.list_files_for_account(account_id, file_type='documents')
                    st.session_state[cache_key] = files
            
            account_files = st.session_state[cache_key]
            for file in account_files:
                file['account_alias'] = account['account_alias']
                file['account_email'] = account['account_email']
            all_files.extend(account_files)
        
        if all_files:
            # Stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="stats-card">
                    <div class="stats-number">{len(all_files)}</div>
                    <div class="stats-label">Total Files</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stats-card">
                    <div class="stats-number">{len(active_accounts)}</div>
                    <div class="stats-label">Active Accounts</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                total_size = sum(int(f.get('size', 0)) for f in all_files) / (1024 * 1024)
                st.markdown(f"""
                <div class="stats-card">
                    <div class="stats-number">{total_size:.1f}MB</div>
                    <div class="stats-label">Total Size</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### Select Files")
            
            # File selection with improved UI
            selected_files = []
            
            # Group files by account
            for account in active_accounts:
                account_files = [f for f in all_files if f['account_id'] == account['account_id']]
                if account_files:
                    with st.expander(f"📁 {account['account_alias']} ({len(account_files)} files)", expanded=True):
                        for file in account_files:
                            col1, col2, col3 = st.columns([4, 2, 1])
                            
                            with col1:
                                if st.checkbox(
                                    f"📄 {file['name']}", 
                                    key=f"file_{user_id}_{file['account_id']}_{file['id']}"
                                ):
                                    selected_files.append(file)
                            
                            with col2:
                                size_mb = int(file.get('size', 0)) / (1024 * 1024) if file.get('size') else 0
                                st.caption(f"{size_mb:.1f} MB")
                            
                            with col3:
                                st.caption(f"🏷️ {account['account_alias'][:8]}...")
            
            # Action buttons
            if selected_files:
                st.markdown("---")
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.info(f"📋 Selected {len(selected_files)} files from {len(set(f['account_id'] for f in selected_files))} accounts")
                
                with col2:
                    if download_files:
                        if st.button("✅ Use Selected Files", type="primary", key="use_selected"):
                            with st.spinner("Downloading files..."):
                                downloaded_paths = drive_manager.download_files_with_conflict_resolution(selected_files)
                                
                                if downloaded_paths:
                                    st.success(f"✅ Downloaded {len(downloaded_paths)} files")
                                    st.session_state[f'drive_downloaded_files_{user_id}'] = downloaded_paths
                                    st.rerun()
                                else:
                                    st.error("❌ Download failed")
                    else:
                        # Just return selected files metadata (no download)
                        st.session_state[f'drive_selected_files_{user_id}'] = selected_files
                        st.success(f"✅ Selected {len(selected_files)} files (ready for download)")
                
                with col3:
                    if st.button("🔄 Refresh All", key="refresh_all"):
                        # Clear all file caches
                        keys_to_remove = [key for key in st.session_state.keys() if key.startswith(f'drive_files_{user_id}_')]
                        for key in keys_to_remove:
                            del st.session_state[key]
                        st.rerun()
        
        else:
            st.info("No supported files found in connected accounts")
    
    elif accounts:
        st.warning("⚠️ All accounts have expired sessions. Please reconnect them.")
    
    else:
        st.info("🔗 Add your first Google Drive account to get started")
    
    # Return downloaded files OR selected files
    if download_files:
        return st.session_state.get(f'drive_downloaded_files_{user_id}')
    else:
        return st.session_state.get(f'drive_selected_files_{user_id}')


def cleanup_multi_account_session(user_id: str, keep_connection: bool = False):
    """Clean up multi-account session data"""
    logger.info(f"🧹 Multi-account cleanup for user: {user_id}, keep_connection: {keep_connection}")
    
    drive_manager = MultiAccountGoogleDriveManager(user_id)
    
    if keep_connection:
        # Only clear downloaded files
        if f'drive_downloaded_files_{user_id}' in st.session_state:
            del st.session_state[f'drive_downloaded_files_{user_id}']
        drive_manager.cleanup_temp_files(full_cleanup=False)
    else:
        # Full cleanup
        drive_manager.clear_streamlit_sessions()
        drive_manager.cleanup_temp_files(full_cleanup=True)
    
    logger.info(f"✅ Multi-account cleanup completed for user: {user_id}")
