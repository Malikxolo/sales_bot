"""
MongoDB MCP Integration
=======================

MCP client for MongoDB Atlas - connects to MongoDB's official MCP server.
Same pattern as Zapier MCP integration.

MongoDB MCP Server: @mongodb-js/mongodb-mcp-server
Docs: https://github.com/mongodb-js/mongodb-mcp-server

The MCP server provides tools like:
- find, aggregate, count (read operations)
- insertOne, updateOne, deleteOne (write operations)  
- listCollections, listDatabases (schema discovery)

Usage:
    from core.mcp.mongodb import MongoDBMCPClient, MongoDBToolManager
    
    # Using MongoDBToolManager (recommended - like ZapierToolManager)
    manager = MongoDBToolManager(
        security_manager=security_manager,
        connection_string="mongodb+srv://..."
    )
    await manager.initialize()
    tools = manager.get_tool_schemas()
    result = await manager.execute("mongodb_find", {"database": "test", ...})
    
    # Or using MongoDBMCPClient directly
    client = MongoDBMCPClient(connection_string="mongodb+srv://...")
    await client.connect()
    tools = await client.list_tools()
    result = await client.execute_tool("find", {...})
"""

import asyncio
import logging
import os
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .security import MCPSecurityManager
from .transport import StdioTransport, MCPRequest, MCPResponse, MCPMethod
from .exceptions import (
    MCPError,
    MCPConnectionError,
    MCPToolExecutionError,
)

logger = logging.getLogger(__name__)


@dataclass
class MongoDBTool:
    """Represents a tool from MongoDB MCP server"""
    name: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def required_params(self) -> List[str]:
        """Get required parameters from schema"""
        return self.input_schema.get("required", [])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
            "required": self.required_params
        }


@dataclass
class MongoDBToolResult:
    """Result of MongoDB tool execution"""
    success: bool
    result: Any = None
    error: str = None
    execution_time_ms: float = 0
    tool_name: str = ""


class MongoDBMCPClient:
    """
    MongoDB MCP Client - connects to MongoDB's official MCP server.
    
    The MongoDB MCP server runs as a subprocess and communicates via stdio.
    This is the same pattern as other MCP servers (Claude Desktop, Cursor, etc.)
    
    MCP Server: npx -y @mongodb-js/mongodb-mcp-server
    
    The server exposes tools like:
    - find: Query documents
    - aggregate: Run aggregation pipeline
    - insertOne/insertMany: Insert documents
    - updateOne/updateMany: Update documents
    - deleteOne/deleteMany: Delete documents
    - listDatabases: List all databases
    - listCollections: List collections in a database
    - createIndex: Create database indexes
    - dropCollection: Drop a collection
    """
    
    # NPX command to run MongoDB MCP server
    NPX_COMMAND = "npx"
    MCP_SERVER_PACKAGE = "@mongodb-js/mongodb-mcp-server"
    
    def __init__(
        self,
        connection_string: str = None,
        timeout: int = 30,
        startup_timeout: int = 120,  # MongoDB MCP server may take time to start
        security_manager: Optional[MCPSecurityManager] = None,
    ):
        """
        Initialize MongoDB MCP client.
        
        Args:
            connection_string: MongoDB connection string (mongodb+srv://...)
            timeout: Request timeout in seconds
            startup_timeout: Timeout for server startup in seconds
            security_manager: Optional security manager for credential handling
        """
        self.timeout = timeout
        self.startup_timeout = startup_timeout
        self.security_manager = security_manager
        
        # Get connection string from param or environment (.env)
        self._connection_string = connection_string or os.getenv("MONGODB_MCP_CONNECTION_STRING", "")
        
        self._transport: Optional[StdioTransport] = None
        self._connected = False
        self._tools: Dict[str, MongoDBTool] = {}
        self._server_info: Dict[str, Any] = {}
        
        # Stats
        self._call_count = 0
        self._success_count = 0
        self._error_count = 0
        self._connect_time: Optional[datetime] = None
        
        logger.info("✅ MongoDBMCPClient initialized")
        if self._connection_string:
            masked = self._mask_connection_string(self._connection_string)
            logger.info(f"   Connection: {masked}")
    
    @staticmethod
    def _mask_connection_string(conn_str: str) -> str:
        """Mask password in connection string for logging"""
        return re.sub(r':([^@/:]+)@', r':****@', conn_str)
    
    @property
    def is_configured(self) -> bool:
        """Check if MongoDB MCP is configured"""
        return bool(self._connection_string)
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to MCP server"""
        return self._connected and self._transport is not None and self._transport.is_connected
    
    async def connect(self) -> bool:
        """
        Connect to MongoDB MCP server.
        
        Starts the MCP server as a subprocess and connects via stdio.
        
        Returns:
            True if connection successful
        """
        if self.is_connected:
            logger.debug("Already connected to MongoDB MCP")
            return True
        
        if not self.is_configured:
            logger.error("❌ MongoDB not configured. Set connection_string or MONGODB_CONNECTION_STRING env var")
            raise MCPConnectionError("MongoDB not configured - no connection string provided")
        
        try:
            logger.info("🔗 Starting MongoDB MCP server...")
            
            # Create stdio transport
            # The MongoDB MCP server expects connection string as env variable
            self._transport = StdioTransport(
                command=self.NPX_COMMAND,
                args=["-y", self.MCP_SERVER_PACKAGE],
                env={
                    "MDB_MCP_CONNECTION_STRING": self._connection_string
                },
                timeout=self.timeout,
                startup_timeout=self.startup_timeout
            )
            
            # Connect (this starts the process and sends initialize)
            self._connected = await self._transport.connect()
            
            if self._connected:
                self._connect_time = datetime.now(timezone.utc)
                
                # Load available tools
                await self._discover_tools()
                
                logger.info(f"✅ Connected to MongoDB MCP server")
                logger.info(f"   Available tools: {len(self._tools)}")
                
                if self._tools:
                    logger.info(f"   Tools: {', '.join(list(self._tools.keys())[:5])}...")
            else:
                logger.error("❌ Failed to connect to MongoDB MCP server")
            
            return self._connected
            
        except FileNotFoundError:
            logger.error(f"❌ npx not found. Make sure Node.js is installed and npx is in PATH")
            raise MCPConnectionError("npx not found - install Node.js")
        except Exception as e:
            logger.error(f"❌ MongoDB MCP connection failed: {e}")
            await self.disconnect()
            raise MCPConnectionError(f"Connection failed: {e}")
    
    async def _discover_tools(self):
        """Discover available tools from MCP server"""
        if not self._transport:
            return
        
        try:
            # Send tools/list request
            request = MCPRequest(
                method=MCPMethod.TOOLS_LIST,
                params={}
            )
            
            response = await self._transport.send_request(request)
            
            if response.is_success and response.result:
                tools_data = response.result.get("tools", [])
                
                for tool_data in tools_data:
                    tool = MongoDBTool(
                        name=tool_data.get("name", ""),
                        description=tool_data.get("description", ""),
                        input_schema=tool_data.get("inputSchema", {})
                    )
                    self._tools[tool.name] = tool
                
                logger.info(f"📋 Discovered {len(self._tools)} MongoDB tools")
                
            elif response.error:
                logger.warning(f"⚠️ Could not discover tools: {response.error_message}")
                
        except Exception as e:
            logger.warning(f"⚠️ Tool discovery failed: {e}")
    
    async def disconnect(self):
        """Disconnect from MongoDB MCP server"""
        if self._transport:
            await self._transport.disconnect()
            self._transport = None
        
        self._connected = False
        self._tools.clear()
        
        logger.info("✅ MongoDB MCP disconnected")
    
    async def list_tools(self) -> List[MongoDBTool]:
        """
        Get list of available tools.
        
        Returns:
            List of MongoDBTool objects
        """
        if not self.is_connected:
            raise MCPConnectionError("Not connected to MongoDB MCP server")
        
        return list(self._tools.values())
    
    def get_tool(self, name: str) -> Optional[MongoDBTool]:
        """Get specific tool by name"""
        return self._tools.get(name)
    
    async def execute_tool(
        self,
        tool_name: str,
        params: Dict[str, Any]
    ) -> MongoDBToolResult:
        """
        Execute a MongoDB MCP tool.
        
        Args:
            tool_name: Name of tool (find, insertOne, aggregate, etc.)
            params: Tool parameters (database, collection, filter, etc.)
            
        Returns:
            MongoDBToolResult with execution result
        """
        if not self.is_connected:
            raise MCPConnectionError("Not connected to MongoDB MCP server")
        
        if not self._transport:
            raise MCPConnectionError("Transport not available")
        
        self._call_count += 1
        start_time = asyncio.get_event_loop().time()
        
        logger.info(f"🚀 Executing MongoDB tool: {tool_name}")
        logger.debug(f"   Params: {params}")
        
        try:
            # Send tools/call request
            request = MCPRequest(
                method=MCPMethod.TOOLS_CALL,
                params={
                    "name": tool_name,
                    "arguments": params
                }
            )
            
            response = await self._transport.send_request(request)
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            if response.is_success:
                self._success_count += 1
                
                # Extract result from MCP response
                result_data = response.result
                
                # MCP tools/call returns content array
                if isinstance(result_data, dict) and "content" in result_data:
                    content = result_data.get("content", [])
                    if content and isinstance(content, list):
                        # Get text content
                        text_content = next(
                            (c.get("text") for c in content if c.get("type") == "text"),
                            str(content)
                        )
                        result_data = text_content
                
                logger.info(f"✅ {tool_name} completed in {execution_time:.0f}ms")
                
                return MongoDBToolResult(
                    success=True,
                    result=result_data,
                    execution_time_ms=execution_time,
                    tool_name=tool_name
                )
            else:
                self._error_count += 1
                error_msg = response.error_message or "Unknown error"
                
                logger.error(f"❌ {tool_name} failed: {error_msg}")
                
                return MongoDBToolResult(
                    success=False,
                    error=error_msg,
                    execution_time_ms=execution_time,
                    tool_name=tool_name
                )
                
        except asyncio.TimeoutError:
            self._error_count += 1
            error_msg = f"Tool execution timeout ({self.timeout}s)"
            logger.error(f"❌ {tool_name} timeout")
            
            return MongoDBToolResult(
                success=False,
                error=error_msg,
                tool_name=tool_name
            )
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"❌ Error executing {tool_name}: {e}")
            
            return MongoDBToolResult(
                success=False,
                error=str(e),
                tool_name=tool_name
            )
    
    def get_tools_prompt(self) -> str:
        """
        Generate tools description for LLM prompts.
        
        Returns:
            Formatted string describing available MongoDB tools
        """
        if not self._tools:
            return ""
        
        lines = [
            "## MongoDB Database Tools (via MCP)",
            "",
            "The following MongoDB operations are available:",
            ""
        ]
        
        # Group tools by type
        read_tools = []
        write_tools = []
        schema_tools = []
        
        for name, tool in self._tools.items():
            if name in ["find", "aggregate", "count", "countDocuments"]:
                read_tools.append(tool)
            elif name in ["insertOne", "insertMany", "updateOne", "updateMany", "deleteOne", "deleteMany"]:
                write_tools.append(tool)
            else:
                schema_tools.append(tool)
        
        if read_tools:
            lines.append("### Read Operations")
            for tool in read_tools:
                desc = tool.description[:100] if tool.description else tool.name
                lines.append(f"- **{tool.name}**: {desc}")
            lines.append("")
        
        if write_tools:
            lines.append("### Write Operations")
            for tool in write_tools:
                desc = tool.description[:100] if tool.description else tool.name
                lines.append(f"- **{tool.name}**: {desc}")
            lines.append("")
        
        if schema_tools:
            lines.append("### Schema/Admin Operations")
            for tool in schema_tools:
                desc = tool.description[:100] if tool.description else tool.name
                lines.append(f"- **{tool.name}**: {desc}")
            lines.append("")
        
        return "\n".join(lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            "connected": self.is_connected,
            "connection_time": self._connect_time.isoformat() if self._connect_time else None,
            "tools_available": len(self._tools),
            "tool_names": list(self._tools.keys()),
            "calls": self._call_count,
            "successes": self._success_count,
            "errors": self._error_count,
            "success_rate": (self._success_count / max(self._call_count, 1)) * 100,
            "transport_stats": self._transport.get_stats() if self._transport else None
        }


class MongoDBToolManager:
    """
    Bridge between MongoDBMCPClient and OptimizedAgent.
    
    Same pattern as ZapierToolManager - provides tool schemas
    for LLM function calling and routes execution to MCP.
    
    Usage:
        manager = MongoDBToolManager(
            security_manager=security_manager,
            connection_string="mongodb+srv://..."
        )
        await manager.initialize()
        
        # Get tools for LLM
        schemas = manager.get_tool_schemas()
        
        # Execute tool (called by agent)
        result = await manager.execute("mongodb_find", {
            "database": "mydb",
            "collection": "users",
            "filter": {"active": true}
        })
    """
    
    def __init__(
        self,
        security_manager: Optional[MCPSecurityManager] = None,
        connection_string: str = None,
        prefix: str = "mongodb_",
        timeout: int = 30
    ):
        """
        Initialize MongoDB Tool Manager.
        
        Args:
            security_manager: Optional security manager
            connection_string: MongoDB connection string
            prefix: Prefix for tool names (default: "mongodb_")
            timeout: Request timeout in seconds
        """
        self.security_manager = security_manager
        self.connection_string = connection_string
        self.prefix = prefix
        self.timeout = timeout
        
        self._client: Optional[MongoDBMCPClient] = None
        self._initialized = False
        self._tool_schemas: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self) -> bool:
        """
        Initialize MongoDB MCP connection.
        
        Returns:
            True if initialization successful
        """
        try:
            self._client = MongoDBMCPClient(
                connection_string=self.connection_string,
                timeout=self.timeout,
                security_manager=self.security_manager
            )
            
            if not self._client.is_configured:
                logger.warning("⚠️ MongoDB MCP not configured - no connection string")
                return False
            
            connected = await self._client.connect()
            
            if connected:
                # Build tool schemas with prefix
                tools = await self._client.list_tools()
                
                for tool in tools:
                    prefixed_name = f"{self.prefix}{tool.name}"
                    self._tool_schemas[prefixed_name] = {
                        "name": prefixed_name,
                        "description": f"[MongoDB] {tool.description}",
                        "parameters": tool.input_schema,
                        "required": tool.required_params
                    }
                
                self._initialized = True
                logger.info(f"✅ MongoDBToolManager initialized ({len(self._tool_schemas)} tools)")
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ MongoDB initialization failed: {e}")
            return False
    
    @property
    def is_initialized(self) -> bool:
        """Check if manager is initialized"""
        return self._initialized
    
    @property
    def is_connected(self) -> bool:
        """Check if MongoDB is connected"""
        return self._client is not None and self._client.is_connected
    
    def get_tool_names(self) -> List[str]:
        """Get list of available tool names (with prefix)"""
        return list(self._tool_schemas.keys())
    
    def get_tool_schemas(self) -> Dict[str, Dict[str, Any]]:
        """
        Get tool schemas for LLM function calling.
        
        Returns:
            Dictionary of tool schemas
        """
        return self._tool_schemas.copy()
    
    def get_tools_prompt(self) -> str:
        """
        Generate tools description for LLM system prompt.
        
        Returns:
            Formatted tools description
        """
        if self._client:
            return self._client.get_tools_prompt()
        return ""
    
    async def execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a MongoDB tool.
        
        Args:
            tool_name: Tool name (with or without prefix)
            params: Tool parameters
            
        Returns:
            Result dictionary with success, result, error fields
        """
        if not self._initialized or not self._client:
            return {
                "success": False,
                "error": "MongoDB not initialized",
                "provider": "mongodb_mcp"
            }
        
        # Remove prefix if present
        actual_name = tool_name
        if tool_name.startswith(self.prefix):
            actual_name = tool_name[len(self.prefix):]
        
        try:
            result = await self._client.execute_tool(actual_name, params)
            
            return {
                "success": result.success,
                "tool": tool_name,
                "result": result.result,
                "error": result.error,
                "execution_time_ms": result.execution_time_ms,
                "provider": "mongodb_mcp"
            }
            
        except MCPConnectionError as e:
            return {
                "success": False,
                "tool": tool_name,
                "error": f"Connection error: {e}",
                "provider": "mongodb_mcp"
            }
        except Exception as e:
            return {
                "success": False,
                "tool": tool_name,
                "error": str(e),
                "provider": "mongodb_mcp"
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics"""
        stats = {
            "initialized": self._initialized,
            "tools_available": len(self._tool_schemas),
            "tool_names": list(self._tool_schemas.keys()),
            "prefix": self.prefix
        }
        
        if self._client:
            stats["client_stats"] = self._client.get_stats()
        
        return stats
    
    async def close(self):
        """Close MongoDB connection"""
        if self._client:
            await self._client.disconnect()
            self._client = None
        
        self._initialized = False
        self._tool_schemas.clear()
        
        logger.info("✅ MongoDBToolManager closed")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
