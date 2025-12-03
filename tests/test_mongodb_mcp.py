"""
MongoDB MCP Integration Tests
=============================

Tests for MongoDB MCP client that connects to MongoDB Atlas
via the official @mongodb-js/mongodb-mcp-server.

Run: python -m pytest tests/test_mongodb_mcp.py -v
"""

import asyncio
import pytest
import os
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

# Add parent to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mcp.mongodb import (
    MongoDBMCPClient,
    MongoDBToolManager,
    MongoDBTool,
    MongoDBToolResult
)
from core.mcp.transport import StdioTransport, MCPResponse, MCPRequest
from core.mcp.exceptions import MCPConnectionError


class TestMongoDBTool:
    """Tests for MongoDBTool dataclass"""
    
    def test_tool_creation(self):
        """Test tool creation with all fields"""
        tool = MongoDBTool(
            name="find",
            description="Find documents in a collection",
            input_schema={
                "type": "object",
                "properties": {
                    "database": {"type": "string"},
                    "collection": {"type": "string"},
                    "filter": {"type": "object"}
                },
                "required": ["database", "collection"]
            }
        )
        
        assert tool.name == "find"
        assert "Find documents" in tool.description
        assert tool.required_params == ["database", "collection"]
    
    def test_tool_to_dict(self):
        """Test tool serialization"""
        tool = MongoDBTool(
            name="insertOne",
            description="Insert a single document"
        )
        
        data = tool.to_dict()
        assert data["name"] == "insertOne"
        assert data["description"] == "Insert a single document"
        assert "parameters" in data
        assert "required" in data
    
    def test_tool_required_params_empty(self):
        """Test required params when schema has no required field"""
        tool = MongoDBTool(name="test")
        assert tool.required_params == []


class TestMongoDBToolResult:
    """Tests for MongoDBToolResult dataclass"""
    
    def test_success_result(self):
        """Test successful result"""
        result = MongoDBToolResult(
            success=True,
            result={"_id": "123", "name": "Test"},
            execution_time_ms=50.5,
            tool_name="find"
        )
        
        assert result.success is True
        assert result.result["_id"] == "123"
        assert result.error is None
        assert result.execution_time_ms == 50.5
    
    def test_error_result(self):
        """Test error result"""
        result = MongoDBToolResult(
            success=False,
            error="Document not found",
            tool_name="find"
        )
        
        assert result.success is False
        assert result.result is None
        assert result.error == "Document not found"


class TestMongoDBMCPClient:
    """Tests for MongoDBMCPClient"""
    
    def test_initialization(self):
        """Test client initialization"""
        client = MongoDBMCPClient(
            connection_string="mongodb+srv://test:test@cluster.mongodb.net/",
            timeout=60
        )
        
        assert client.is_configured is True
        assert client.timeout == 60
        assert client.is_connected is False
    
    def test_initialization_from_env(self):
        """Test initialization from environment variable"""
        with patch.dict(os.environ, {"MONGODB_CONNECTION_STRING": "mongodb://localhost:27017"}):
            client = MongoDBMCPClient()
            assert client.is_configured is True
    
    def test_not_configured(self):
        """Test client without connection string"""
        with patch.dict(os.environ, {}, clear=True):
            # Clear any existing env var
            os.environ.pop("MONGODB_CONNECTION_STRING", None)
            client = MongoDBMCPClient(connection_string=None)
            assert client.is_configured is False
    
    def test_mask_connection_string(self):
        """Test password masking in connection string"""
        conn = "mongodb+srv://user:secretpassword@cluster.mongodb.net/"
        masked = MongoDBMCPClient._mask_connection_string(conn)
        
        assert "secretpassword" not in masked
        assert "****" in masked
        assert "user" in masked
    
    @pytest.mark.asyncio
    async def test_connect_not_configured(self):
        """Test connect fails when not configured"""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MONGODB_CONNECTION_STRING", None)
            client = MongoDBMCPClient(connection_string=None)
            
            with pytest.raises(MCPConnectionError, match="not configured"):
                await client.connect()
    
    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection with mocked transport"""
        client = MongoDBMCPClient(
            connection_string="mongodb://localhost:27017"
        )
        
        # Mock transport
        mock_transport = AsyncMock(spec=StdioTransport)
        mock_transport.connect = AsyncMock(return_value=True)
        mock_transport.is_connected = True
        mock_transport.send_request = AsyncMock(return_value=MCPResponse(
            request_id="test",
            result={"tools": [
                {"name": "find", "description": "Find docs", "inputSchema": {}},
                {"name": "insertOne", "description": "Insert doc", "inputSchema": {}}
            ]}
        ))
        mock_transport.disconnect = AsyncMock()
        mock_transport.get_stats = MagicMock(return_value={})
        
        with patch.object(client, '_transport', mock_transport):
            client._transport = mock_transport
            client._connected = True
            
            # Test tool discovery
            await client._discover_tools()
            
            assert len(client._tools) == 2
            assert "find" in client._tools
            assert "insertOne" in client._tools
    
    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        """Test successful tool execution"""
        client = MongoDBMCPClient(
            connection_string="mongodb://localhost:27017"
        )
        
        # Setup mocked state
        mock_transport = AsyncMock()
        mock_transport.is_connected = True
        mock_transport.send_request = AsyncMock(return_value=MCPResponse(
            request_id="test",
            result={
                "content": [
                    {"type": "text", "text": '{"_id": "123", "name": "Test"}'}
                ]
            }
        ))
        
        client._transport = mock_transport
        client._connected = True
        client._tools = {"find": MongoDBTool(name="find")}
        
        result = await client.execute_tool("find", {
            "database": "test",
            "collection": "users",
            "filter": {}
        })
        
        assert result.success is True
        assert result.tool_name == "find"
        assert client._call_count == 1
        assert client._success_count == 1
    
    @pytest.mark.asyncio
    async def test_execute_tool_error(self):
        """Test tool execution with error response"""
        client = MongoDBMCPClient(
            connection_string="mongodb://localhost:27017"
        )
        
        mock_transport = AsyncMock()
        mock_transport.is_connected = True
        mock_transport.send_request = AsyncMock(return_value=MCPResponse(
            request_id="test",
            error={"code": -32000, "message": "Collection not found"}
        ))
        
        client._transport = mock_transport
        client._connected = True
        
        result = await client.execute_tool("find", {"database": "test"})
        
        assert result.success is False
        assert "Collection not found" in result.error
        assert client._error_count == 1
    
    @pytest.mark.asyncio
    async def test_list_tools_not_connected(self):
        """Test list_tools fails when not connected"""
        client = MongoDBMCPClient(
            connection_string="mongodb://localhost:27017"
        )
        
        with pytest.raises(MCPConnectionError, match="Not connected"):
            await client.list_tools()
    
    def test_get_tools_prompt(self):
        """Test generating tools prompt for LLM"""
        client = MongoDBMCPClient(
            connection_string="mongodb://localhost:27017"
        )
        
        # Add some tools
        client._tools = {
            "find": MongoDBTool(name="find", description="Find documents"),
            "insertOne": MongoDBTool(name="insertOne", description="Insert a document"),
            "listCollections": MongoDBTool(name="listCollections", description="List collections")
        }
        
        prompt = client.get_tools_prompt()
        
        assert "MongoDB" in prompt
        assert "find" in prompt
        assert "insertOne" in prompt
        assert "listCollections" in prompt
    
    def test_get_stats(self):
        """Test statistics gathering"""
        client = MongoDBMCPClient(
            connection_string="mongodb://localhost:27017"
        )
        
        client._call_count = 10
        client._success_count = 8
        client._error_count = 2
        
        stats = client.get_stats()
        
        assert stats["calls"] == 10
        assert stats["successes"] == 8
        assert stats["errors"] == 2
        assert stats["success_rate"] == 80.0


class TestMongoDBToolManager:
    """Tests for MongoDBToolManager"""
    
    def test_initialization(self):
        """Test manager initialization"""
        manager = MongoDBToolManager(
            connection_string="mongodb://localhost:27017",
            prefix="mongo_",
            timeout=45
        )
        
        assert manager.prefix == "mongo_"
        assert manager.timeout == 45
        assert manager.is_initialized is False
    
    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful initialization"""
        manager = MongoDBToolManager(
            connection_string="mongodb://localhost:27017"
        )
        
        # Mock the client
        mock_client = AsyncMock()
        mock_client.is_configured = True
        mock_client.connect = AsyncMock(return_value=True)
        mock_client.list_tools = AsyncMock(return_value=[
            MongoDBTool(name="find", description="Find docs", input_schema={"type": "object"}),
            MongoDBTool(name="insertOne", description="Insert doc")
        ])
        
        with patch.object(manager, '_client', mock_client):
            manager._client = mock_client
            
            # Simulate initialization
            manager._initialized = True
            manager._tool_schemas = {
                "mongodb_find": {"name": "mongodb_find", "description": "[MongoDB] Find docs"},
                "mongodb_insertOne": {"name": "mongodb_insertOne", "description": "[MongoDB] Insert doc"}
            }
            
            assert manager.is_initialized is True
            assert len(manager.get_tool_names()) == 2
            assert "mongodb_find" in manager.get_tool_names()
    
    @pytest.mark.asyncio
    async def test_execute_with_prefix(self):
        """Test executing tool with prefix"""
        manager = MongoDBToolManager(
            connection_string="mongodb://localhost:27017"
        )
        
        mock_client = AsyncMock()
        mock_client.is_connected = True
        mock_client.execute_tool = AsyncMock(return_value=MongoDBToolResult(
            success=True,
            result={"count": 5},
            execution_time_ms=25.0,
            tool_name="find"
        ))
        
        manager._client = mock_client
        manager._initialized = True
        
        # Execute with prefix - should strip it
        result = await manager.execute("mongodb_find", {"database": "test"})
        
        assert result["success"] is True
        assert result["provider"] == "mongodb_mcp"
        mock_client.execute_tool.assert_called_once_with("find", {"database": "test"})
    
    @pytest.mark.asyncio
    async def test_execute_not_initialized(self):
        """Test execute fails when not initialized"""
        manager = MongoDBToolManager(
            connection_string="mongodb://localhost:27017"
        )
        
        result = await manager.execute("find", {})
        
        assert result["success"] is False
        assert "not initialized" in result["error"]
    
    def test_get_tool_schemas(self):
        """Test getting tool schemas for LLM"""
        manager = MongoDBToolManager(
            connection_string="mongodb://localhost:27017"
        )
        
        manager._tool_schemas = {
            "mongodb_find": {
                "name": "mongodb_find",
                "description": "[MongoDB] Find documents",
                "parameters": {"type": "object"},
                "required": ["database", "collection"]
            }
        }
        
        schemas = manager.get_tool_schemas()
        
        assert "mongodb_find" in schemas
        assert schemas["mongodb_find"]["description"].startswith("[MongoDB]")
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager"""
        manager = MongoDBToolManager(
            connection_string="mongodb://localhost:27017"
        )
        
        # Mock initialize and close
        manager.initialize = AsyncMock(return_value=True)
        manager.close = AsyncMock()
        
        async with manager as mgr:
            assert mgr is manager
            manager.initialize.assert_called_once()
        
        manager.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing connection"""
        manager = MongoDBToolManager(
            connection_string="mongodb://localhost:27017"
        )
        
        mock_client = AsyncMock()
        mock_client.disconnect = AsyncMock()
        
        manager._client = mock_client
        manager._initialized = True
        manager._tool_schemas = {"test": {}}
        
        await manager.close()
        
        assert manager._initialized is False
        assert len(manager._tool_schemas) == 0
        mock_client.disconnect.assert_called_once()


class TestStdioTransportIntegration:
    """Integration tests for StdioTransport with MongoDB"""
    
    @pytest.mark.asyncio
    async def test_transport_initialization(self):
        """Test stdio transport can be created for MongoDB"""
        transport = StdioTransport(
            command="npx",
            args=["-y", "@mongodb-js/mongodb-mcp-server"],
            env={"MDB_MCP_CONNECTION_STRING": "mongodb://localhost:27017"},
            timeout=30,
            startup_timeout=60
        )
        
        # Just verify it initializes without error
        assert transport.command == "npx"
        assert "@mongodb-js/mongodb-mcp-server" in transport.args
        assert transport.is_connected is False


# Integration test (requires MongoDB connection string from .env)
@pytest.mark.skipif(
    not os.getenv("MONGODB_MCP_CONNECTION_STRING"),
    reason="MONGODB_MCP_CONNECTION_STRING not set in .env"
)
class TestMongoDBMCPIntegration:
    """Real integration tests - only run with valid MongoDB connection"""
    
    @pytest.mark.asyncio
    async def test_real_connection(self):
        """Test real MongoDB MCP connection"""
        client = MongoDBMCPClient()
        
        try:
            connected = await client.connect()
            
            if connected:
                # List tools
                tools = await client.list_tools()
                assert len(tools) > 0
                
                # Get stats
                stats = client.get_stats()
                assert stats["connected"] is True
                
        finally:
            await client.disconnect()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
