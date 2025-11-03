"""
Organization Management - MongoDB-based implementation
Handles org creation, joining, and permission checks
"""

import random
import string
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import asyncio
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from os import getenv
import logging
logger = logging.getLogger(__name__)


def serialize_datetimes(obj):
    """Recursively convert datetimes in dicts/lists to ISO strings."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: serialize_datetimes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_datetimes(i) for i in obj]
    else:
        return obj


PERMISSIONS = {
    "owner": {
        # Collections
        "upload_documents": True,
        "edit_collection": True,
        "delete_collection": True,
        
        # Teams
        "create_team": True,
        "manage_teams": True,
        "delete_team": True,
        "view_teams": True,
        
        # Members
        "assign_members": True,
        "remove_members": True,
        "view_members": True,
        
        # Organization
        "view_invite_code": True,
        "regenerate_invite": True,
        "leave_org": True,
        "delete_org": True,
        
        # Query
        "select_team_for_query": True,
    },
    
    "team_admin": {
        # Collections (THEIR TEAM ONLY)
        "upload_documents": True,
        "edit_collection": True,
        "delete_collection": True,
        
        # Teams
        "create_team": False,
        "manage_teams": False,
        "delete_team": False,
        "view_teams": True,
        
        # Members (THEIR TEAM ONLY)
        "assign_members": True,
        "remove_members": True,
        "view_members": True,
        
        # Organization
        "view_invite_code": False,
        "regenerate_invite": False,
        "leave_org": True,
        "delete_org": False,
        
        # Query
        "select_team_for_query": False,
    },
    
    "member": {
        # Collections
        "upload_documents": False,
        "edit_collection": False,
        "delete_collection": False,
        
        # Teams
        "create_team": False,
        "manage_teams": False,
        "delete_team": False,
        "view_teams": True,
        
        # Members
        "assign_members": False,
        "remove_members": False,
        "view_members": True,
        
        # Organization
        "view_invite_code": False,
        "regenerate_invite": False,
        "leave_org": True,
        "delete_org": False,
        
        # Query
        "select_team_for_query": False,
    },
    
    "viewer": {
        # Collections
        "upload_documents": False,
        "edit_collection": False,
        "delete_collection": False,
        
        # Teams
        "create_team": False,
        "manage_teams": False,
        "delete_team": False,
        "view_teams": False,
        
        # Members
        "assign_members": False,
        "remove_members": False,
        "view_members": False,
        
        # Organization
        "view_invite_code": False,
        "regenerate_invite": False,
        "leave_org": True,
        "delete_org": False,
        
        # Query
        "select_team_for_query": False,
    }
}


class OrganizationManager:
    """Manage organizations using MongoDB"""
    
    def __init__(self, mongo_uri:str, database_name: str = "rag_system"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[database_name]
        self.organizations = self.db.organizations
        self._setup_indexes()
    
    def _setup_indexes(self):
        """Create indexes for efficient queries"""
        # Unique index on org_id
        self.organizations.create_index("org_id", unique=True)
        # Unique index on invite_code
        self.organizations.create_index("invite_code", unique=True)
        # Index on owner_id for quick lookups
        self.organizations.create_index("owner_id")
        # Index on member user_ids
        self.organizations.create_index("members.user_id")
    
    async def _generate_invite_code(self, org_name: str) -> str:
        """Generate unique 8-character invite code"""
        # Safe characters (no confusing ones: O, I, L, 0, 1)
        chars = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
        
        # Prefix with org initials (3 chars max)
        words = org_name.upper().split()
        prefix = ''.join([w[0] for w in words[:3]])[:3]
        
        # Generate unique code
        max_attempts = 10
        for _ in range(max_attempts):
            suffix = ''.join(random.choices(chars, k=5))
            code = f"{prefix}{suffix}"
            
            # Check if code exists
            existing = await asyncio.to_thread(
                self.organizations.find_one,
                {"invite_code": code}
            )
            if not existing:
                return code
        
        # Fallback to fully random if prefix collision
        for _ in range(max_attempts):
            code = ''.join(random.choices(chars, k=8))
            existing = await asyncio.to_thread(
                self.organizations.find_one,
                {"invite_code": code}
            )
            if not existing:
                return code
        
        raise Exception("Failed to generate unique invite code")
    
    def _generate_org_id(self, org_name: str) -> str:
        """Generate org_id from org name"""
        # Convert to lowercase, replace spaces with underscore
        org_id = org_name.lower().replace(" ", "_")
        # Remove special characters
        org_id = ''.join(c for c in org_id if c.isalnum() or c == '_')
        return f"org_{org_id}"
    
    async def create_organization(
        self, 
        org_name: str, 
        creator_name: str, 
        creator_id: str
    ) -> Dict[str, Any]:
        """
        Create a new organization or return existing if user is the owner
        
        Args:
            org_name: Name of the organization
            creator_name: Display name of creator
            creator_id: Unique user ID (from session)
        
        Returns:
            {
                "success": bool,
                "org_id": str,
                "invite_code": str,
                "is_existing": bool,
                "error": str (if failed)
            }
        """
        try:
            # Check if organization with this name already exists
            existing_org = await asyncio.to_thread(
                self.organizations.find_one,
                {"org_name": org_name}
            )
            
            if existing_org:
                # Check if the user is the owner
                if existing_org.get("owner_id") == creator_id:
                    # User is the owner, return existing org details
                    return {
                        "success": True,
                        "org_id": existing_org["org_id"],
                        "org_name": existing_org["org_name"],
                        "invite_code": existing_org["invite_code"],
                        "role": "owner",
                        "is_existing": True,
                        "message": f"Organization '{org_name}' already exists and you are the owner"
                    }
                else:
                    # User is not the owner
                    return {
                        "success": False,
                        "error": f"Organization '{org_name}' already exists and you are not the owner"
                    }
            
            # Organization doesn't exist, create new one
            org_id = self._generate_org_id(org_name)
            invite_code = await self._generate_invite_code(org_name)
            
            # Create organization structure
            org_data = {
                "org_id": org_id,
                "org_name": org_name,
                "invite_code": invite_code,
                "owner_id": creator_id,
                "created_at": datetime.now(timezone.utc),
                "teams": {},
                "members": {
                    creator_id: {
                        "name": creator_name,
                        "role": "owner",
                        "team_id": None,
                        "joined_at": datetime.now(timezone.utc)
                    }
                }
            }
            
            # Insert into MongoDB
            await asyncio.to_thread(
                self.organizations.insert_one,
                org_data
            )
            
            return {
                "success": True,
                "org_id": org_id,
                "org_name": org_name,
                "invite_code": invite_code,
                "role": "owner",
                "is_existing": False,
                "message": f"Organization '{org_name}' created successfully"
            }
            
        except DuplicateKeyError:
            
            existing_org = await asyncio.to_thread(
                self.organizations.find_one,
                {"org_name": org_name}
            )
            
            if existing_org and existing_org.get("owner_id") == creator_id:
                return {
                    "success": True,
                    "org_id": existing_org["org_id"],
                    "org_name": existing_org["org_name"],
                    "invite_code": existing_org["invite_code"],
                    "role": "owner",
                    "is_existing": True,
                    "message": f"Organization '{org_name}' already exists and you are the owner"
                }
            else:
                return {
                    "success": False,
                    "error": f"Organization '{org_name}' already exists and you are not the owner"
                }
        except Exception as e:
            logger.error(f"Failed to create organization: {e}")
            return {
                "success": False,
                "error": f"Failed to create organization: {str(e)}"
            }
    
    async def join_organization(
        self, 
        invite_code: str, 
        user_name: str, 
        user_id: str
    ) -> Dict[str, Any]:
        """
        Join an organization using invite code
        
        Args:
            invite_code: 8-character invite code
            user_name: Display name of user
            user_id: Unique user ID (from session)
        
        Returns:
            {
                "success": bool,
                "org_id": str,
                "org_name": str,
                "role": "viewer",
                "error": str (if failed)
            }
        """
        try:
            # Find org by invite code
            org = await asyncio.to_thread(
                self.organizations.find_one,
                {"invite_code": invite_code.upper()}
            )
            
            if not org:
                return {
                    "success": False,
                    "error": f"Invalid invite code: {invite_code}"
                }
            
            # Check if user already a member
            if user_id in org["members"]:
                return {
                    "success": True,
                    "org_id": org["org_id"],
                    "org_name": org["org_name"],
                    "role": org["members"][user_id]["role"],
                    "message": "You are already a member of this organization"
                }
            
            # Add user as viewer (unassigned)
            member_data = {
                "name": user_name,
                "role": "viewer",
                "team_id": None,
                "joined_at": datetime.now(timezone.utc)
            }
            
            await asyncio.to_thread(
                self.organizations.update_one,
                {"org_id": org["org_id"]},
                {"$set": {f"members.{user_id}": member_data}}
            )
            
            return {
                "success": True,
                "org_id": org["org_id"],
                "org_name": org["org_name"],
                "role": "viewer"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to join organization: {str(e)}"
            }
    
    async def get_user_role(self, org_id: str, user_id: str) -> Optional[str]:
        """Get user's role in organization"""
        try:
            org = await asyncio.to_thread(
                self.organizations.find_one,
                {"org_id": org_id},
                {"members": 1}
            )
            
            if org and user_id in org.get("members", {}):
                return org["members"][user_id].get("role")
            return None
        except Exception as e:
            print(f"Error getting user role: {e}")
            return None
    
    async def check_permission(self, org_id: str, user_id: str, action: str) -> bool:
        """
        Check if user has permission for action
        
        Args:
            org_id: Organization ID
            user_id: User ID
            action: Action from PERMISSIONS dict (e.g., 'upload_documents')
        
        Returns:
            bool: True if user has permission
        """
        role = await self.get_user_role(org_id, user_id)
        
        if not role:
            return False
        
        # Get role permissions from PERMISSIONS dict
        role_permissions = PERMISSIONS.get(role, {})
        logger.info(f"User {user_id} with role {role} permissions: {role_permissions}")
        return role_permissions.get(action, False)
    
    async def get_organization(self, org_id: str) -> Optional[Dict[str, Any]]:
        """Get organization data"""
        try:
            org = await asyncio.to_thread(
                self.organizations.find_one,
                {"org_id": org_id},
                {"_id": 0}
            )
            
            if org:
                org = serialize_datetimes(org)
            return org
        except Exception as e:
            print(f"Error getting organization: {e}")
            return None
    
    async def get_organization_by_code(self, invite_code: str) -> Optional[Dict[str, Any]]:
        """Find organization by invite code"""
        try:
            org = await asyncio.to_thread(
                self.organizations.find_one,
                {"invite_code": invite_code.upper()}
            )
            return org
        except Exception as e:
            print(f"Error finding organization: {e}")
            return None
    
    async def get_member_count(self, org_id: str) -> int:
        """Get number of members in organization"""
        org = await self.get_organization(org_id)
        if org:
            return len(org.get("members", {}))
        return 0
    
    async def get_members(self, org_id: str) -> List[Dict[str, Any]]:
        """Get list of members with details"""
        org = await self.get_organization(org_id)
        if org:
            members = []
            for user_id, member_data in org.get("members", {}).items():
                members.append({
                    "user_id": user_id,
                    "name": member_data.get("name"),
                    "role": member_data.get("role"),
                    "joined_at": member_data.get("joined_at")
                })
            return members
        return []
    
    async def transfer_admin(self, org_id: str, old_admin_id: str, new_admin_id: str):
        """Transfer admin role to another member"""
        await asyncio.to_thread(
            self.organizations.update_one,
            {"org_id": org_id},
            {
                "$unset": {f"members.{old_admin_id}": ""},
                "$set": {f"members.{new_admin_id}.role": "owner"}
            }
        )
    
    async def remove_member(self, org_id: str, user_id: str):
        """Remove member from organization"""
        await asyncio.to_thread(
            self.organizations.update_one,
            {"org_id": org_id},
            {"$unset": {f"members.{user_id}": ""}}
        )
    
    async def delete_organization(self, org_id: str):
        """Delete organization completely"""
        result = await asyncio.to_thread(
            self.organizations.delete_one,
            {"org_id": org_id}
        )

        if result.deleted_count > 0:
            return {"success": True, "deleted_count": result.deleted_count}
        else:
            return {"success": False, "error": f"No organization found with ID {org_id}"}

    
    async def create_team(self, org_id: str, team_name: str, owner_id: str) -> Dict[str, Any]:
        """Create a new team within organization"""
        try:
            org = await self.get_organization(org_id)
            
            if not org:
                return {"success": False, "error": "Organization not found"}
            
            # Check if owner
            if org["owner_id"] != owner_id:
                return {"success": False, "error": "Only owner can create teams"}
            
            # Generate team ID
            team_id = f"team_{team_name.lower().replace(' ', '_')}"
            
            # Check if team exists
            if team_id in org.get("teams", {}):
                return {"success": False, "error": "Team already exists"}
            
            # Create team
            team_data = {
                "team_id": team_id,
                "team_name": team_name,
                "team_admin_id": None,
                "created_at": datetime.now(timezone.utc)
            }
            
            await asyncio.to_thread(
                self.organizations.update_one,
                {"org_id": org_id},
                {"$set": {f"teams.{team_id}": team_data}}
            )
            
            return {
                "success": True,
                "team_id": team_id,
                "team_name": team_name
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_teams(self, org_id: str) -> List[Dict[str, Any]]:
        """Get all teams in organization"""
        try:
            org = await self.get_organization(org_id)
            if org:
                teams = org.get("teams", {})
                return [team_data for team_data in teams.values()]
            return []
        except Exception as e:
            print(f"Error getting teams: {e}")
            return []
    
    async def assign_team_admin(self, org_id: str, team_id: str, user_id: str, owner_id: str) -> Dict[str, Any]:
        """Assign team admin"""
        try:
            org = await self.get_organization(org_id)
            
            if not org:
                return {"success": False, "error": "Organization not found"}
            
            # Check if owner
            if org["owner_id"] != owner_id:
                return {"success": False, "error": "Only owner can assign team admin"}
            
            # Check if team exists
            if team_id not in org.get("teams", {}):
                return {"success": False, "error": "Team not found"}
            
            # Check if user in org
            if user_id not in org["members"]:
                return {"success": False, "error": "User not in organization"}
            
            # Assign team admin
            await asyncio.to_thread(
                self.organizations.update_one,
                {"org_id": org_id},
                {
                    "$set": {
                        f"teams.{team_id}.team_admin_id": user_id,
                        f"members.{user_id}.role": "team_admin",
                        f"members.{user_id}.team_id": team_id
                    }
                }
            )
            
            return {"success": True}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def add_member_to_team(self, org_id: str, team_id: str, user_id: str, by_user_id: str) -> Dict[str, Any]:
        """Add member to team (by owner or team admin)"""
        try:
            org = await self.get_organization(org_id)
            
            if not org:
                return {"success": False, "error": "Organization not found"}
            
            # Check permission (owner or team admin)
            is_owner = org["owner_id"] == by_user_id
            is_team_admin = (team_id in org.get("teams", {}) and 
                           org["teams"][team_id].get("team_admin_id") == by_user_id)
            
            if not (is_owner or is_team_admin):
                return {"success": False, "error": "Permission denied"}
            
            # Check if user exists
            if user_id not in org["members"]:
                return {"success": False, "error": "User not in organization"}
            
            # Check if team exists
            if team_id not in org.get("teams", {}):
                return {"success": False, "error": "Team not found"}
            
            # Prepare update
            update_dict = {f"members.{user_id}.team_id": team_id}
            
            # Auto-promote viewer to member when assigned
            current_role = org["members"][user_id]["role"]
            if current_role == "viewer":
                update_dict[f"members.{user_id}.role"] = "member"
            elif current_role not in ["owner", "team_admin"]:
                update_dict[f"members.{user_id}.role"] = "member"
            
            await asyncio.to_thread(
                self.organizations.update_one,
                {"org_id": org_id},
                {"$set": update_dict}
            )
            
            return {"success": True}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def remove_member_from_team(self, org_id: str, team_id: str, user_id: str, by_user_id: str) -> Dict[str, Any]:
        """Remove member from team (by owner or team admin)"""
        try:
            org = await self.get_organization(org_id)
            
            if not org:
                return {"success": False, "error": "Organization not found"}
            
            # Check permission
            is_owner = org["owner_id"] == by_user_id
            is_team_admin = (team_id in org.get("teams", {}) and 
                           org["teams"][team_id].get("team_admin_id") == by_user_id)
            
            if not (is_owner or is_team_admin):
                return {"success": False, "error": "Permission denied"}
            
            # Remove from team
            if user_id in org["members"]:
                current_role = org["members"][user_id]["role"]
                new_role = "viewer" if current_role in ["team_admin", "member"] else current_role
                
                await asyncio.to_thread(
                    self.organizations.update_one,
                    {"org_id": org_id},
                    {
                        "$set": {
                            f"members.{user_id}.team_id": None,
                            f"members.{user_id}.role": new_role
                        }
                    }
                )
            
            return {"success": True}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_unassigned_members(self, org_id: str) -> List[Dict[str, Any]]:
        """Get members not assigned to any team"""
        try:
            org = await self.get_organization(org_id)
            if org:
                members = []
                for user_id, member_data in org.get("members", {}).items():
                    if member_data.get("team_id") is None and member_data.get("role") != "owner":
                        members.append({
                            "user_id": user_id,
                            "name": member_data.get("name"),
                            "role": member_data.get("role")
                        })
                return members
            return []
        except Exception as e:
            print(f"Error getting unassigned members: {e}")
            return []
    
    async def get_team_members(self, org_id: str, team_id: str) -> List[Dict[str, Any]]:
        """Get all members of a team"""
        try:
            org = await self.get_organization(org_id)
            if org:
                members = []
                for user_id, member_data in org.get("members", {}).items():
                    if member_data.get("team_id") == team_id:
                        members.append({
                            "user_id": user_id,
                            "name": member_data.get("name"),
                            "role": member_data.get("role")
                        })
                return members
            return []
        except Exception as e:
            print(f"Error getting team members: {e}")
            return []
    
    async def delete_team(self, org_id: str, team_id: str, owner_id: str) -> Dict[str, Any]:
        """Delete team (only if empty)"""
        try:
            org = await self.get_organization(org_id)
            
            if not org:
                return {"success": False, "error": "Organization not found"}
            
            # Check if owner
            if org["owner_id"] != owner_id:
                return {"success": False, "error": "Only owner can delete teams"}
            
            # Check if team has members
            team_members = await self.get_team_members(org_id, team_id)
            if team_members:
                return {"success": False, "error": f"Team has {len(team_members)} members. Remove them first."}
            
            # Delete team
            await asyncio.to_thread(
                self.organizations.update_one,
                {"org_id": org_id},
                {"$unset": {f"teams.{team_id}": ""}}
            )
            
            return {"success": True}
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# Global instance
org_manager = None


def initialize_org_manager(connection_string: str = "mongodb://localhost:27017/", database_name: str = "knowledge_base") -> OrganizationManager:
    """Initialize the global organization manager"""
    global org_manager
    org_manager = OrganizationManager(getenv('MONGODB_URI'), database_name)
    return org_manager


# Convenience functions
async def create_organization(org_name: str, creator_name: str, creator_id: str) -> Dict[str, Any]:
    """Create new organization"""
    if org_manager is None:
        raise Exception("OrganizationManager not initialized. Call initialize_org_manager() first.")
    return await org_manager.create_organization(org_name, creator_name, creator_id)


async def join_organization(invite_code: str, user_name: str, user_id: str) -> Dict[str, Any]:
    """Join organization with invite code"""
    if org_manager is None:
        raise Exception("OrganizationManager not initialized. Call initialize_org_manager() first.")
    return await org_manager.join_organization(invite_code, user_name, user_id)


async def check_permission(org_id: str, user_id: str, action: str) -> bool:
    """Check if user can perform action"""
    if org_manager is None:
        raise Exception("OrganizationManager not initialized. Call initialize_org_manager() first.")
    return await org_manager.check_permission(org_id, user_id, action)


async def get_organization(org_id: str) -> Optional[Dict[str, Any]]:
    """Get organization data"""
    if org_manager is None:
        raise Exception("OrganizationManager not initialized. Call initialize_org_manager() first.")
    return await org_manager.get_organization(org_id)

async def delete_organization(org_id: str):
    """Delete organization"""
    if org_manager is None:
        raise Exception("OrganizationManager not initialized. Call initialize_org_manager() first.")
    return await org_manager.delete_organization(org_id)