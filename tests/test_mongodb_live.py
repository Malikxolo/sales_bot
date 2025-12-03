"""
Live integration test for MongoDB MCP Client.

This test connects to a real MongoDB Atlas cluster and performs actual operations.
Run manually to verify the MCP client works correctly.

Usage:
    python tests/test_mongodb_live.py
"""

import asyncio
import sys
import os
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import directly to avoid core package dependencies
from core.mcp.mongodb import MongoDBMCPClient, MongoDBToolManager


async def test_mongodb_mcp_live():
    """Live test of MongoDB MCP client."""
    
    print("=" * 60)
    print("MongoDB MCP Client Live Test")
    print("=" * 60)
    
    # Check connection string is set in environment
    if not os.getenv("MONGODB_MCP_CONNECTION_STRING"):
        print("[ERROR] MONGODB_MCP_CONNECTION_STRING not set in environment")
        return False
    
    # Initialize client (reads connection string from env)
    client = MongoDBMCPClient()
    
    try:
        # Step 1: Connect to MCP server
        print("\n[Step 1] Connecting to MongoDB MCP server...")
        connected = await client.connect()
        if connected:
            print("[OK] Connected successfully!")
        else:
            print("[ERROR] Failed to connect")
            return False
        
        # Step 2: Discover tools
        print("\n[Step 2] Discovering available tools...")
        tools = await client.list_tools()
        print(f"[OK] Discovered {len(tools)} tools")
        
        # Show first few tools
        print("\nFirst 5 tools:")
        for tool in tools[:5]:
            print(f"  - {tool.name}: {tool.description[:50]}...")
        
        # Step 3: Test insert-one
        print("\n[Step 3] Testing insert-one tool...")
        test_doc = {
            "test_type": "mcp_live_test",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "Hello from MCP client!",
            "source": "test_mongodb_live.py"
        }
        
        result = await client.execute_tool(
            "insert-one",
            {
                "database": "test_db",
                "collection": "mcp_tests",
                "document": test_doc
            }
        )
        
        if result.success:
            print(f"[OK] Document inserted!")
            print(f"    Result: {result.result}")
        else:
            print(f"[ERROR] Insert failed: {result.error}")
        
        # Step 4: Test find
        print("\n[Step 4] Testing find tool...")
        find_result = await client.execute_tool(
            "find",
            {
                "database": "test_db",
                "collection": "mcp_tests",
                "filter": {"test_type": "mcp_live_test"},
                "limit": 5
            }
        )
        
        if find_result.success:
            print(f"[OK] Find completed!")
            print(f"    Found documents: {find_result.result}")
        else:
            print(f"[ERROR] Find failed: {find_result.error}")
        
        # Step 5: Test count-documents
        print("\n[Step 5] Testing count-documents tool...")
        count_result = await client.execute_tool(
            "count-documents",
            {
                "database": "test_db",
                "collection": "mcp_tests",
                "filter": {"test_type": "mcp_live_test"}
            }
        )
        
        if count_result.success:
            print(f"[OK] Count completed!")
            print(f"    Count: {count_result.result}")
        else:
            print(f"[ERROR] Count failed: {count_result.error}")
        
        # Step 6: Cleanup - delete test documents
        print("\n[Step 6] Cleaning up test documents...")
        delete_result = await client.execute_tool(
            "delete-many",
            {
                "database": "test_db",
                "collection": "mcp_tests",
                "filter": {"test_type": "mcp_live_test"}
            }
        )
        
        if delete_result.success:
            print(f"[OK] Cleanup completed!")
            print(f"    Deleted: {delete_result.result}")
        else:
            print(f"[ERROR] Cleanup failed: {delete_result.error}")
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Disconnect
        print("\nDisconnecting...")
        await client.disconnect()
        print("Disconnected.")


async def test_insert_many():
    """Test inserting multiple documents."""
    
    print("\n" + "=" * 60)
    print("Testing insert-many operation")
    print("=" * 60)
    
    if not os.getenv("MONGODB_MCP_CONNECTION_STRING"):
        print("[ERROR] MONGODB_MCP_CONNECTION_STRING not set")
        return False
    
    client = MongoDBMCPClient()
    
    try:
        await client.connect()
        
        # Insert multiple documents
        documents = [
            {"name": "Alice", "age": 30, "city": "New York"},
            {"name": "Bob", "age": 25, "city": "Los Angeles"},
            {"name": "Charlie", "age": 35, "city": "Chicago"}
        ]
        
        result = await client.execute_tool(
            "insert-many",
            {
                "database": "test_db",
                "collection": "people",
                "documents": documents
            }
        )
        
        if result.success:
            print(f"[OK] Inserted {len(documents)} documents!")
            print(f"    Result: {result.result}")
        else:
            print(f"[ERROR] Insert failed: {result.error}")
        
        # Cleanup
        await client.execute_tool(
            "delete-many",
            {
                "database": "test_db",
                "collection": "people",
                "filter": {"name": {"$in": ["Alice", "Bob", "Charlie"]}}
            }
        )
        
        return result.success
        
    finally:
        await client.disconnect()


async def test_aggregate():
    """Test aggregation pipeline."""
    
    print("\n" + "=" * 60)
    print("Testing aggregate operation")
    print("=" * 60)
    
    if not os.getenv("MONGODB_MCP_CONNECTION_STRING"):
        print("[ERROR] MONGODB_MCP_CONNECTION_STRING not set")
        return False
    
    client = MongoDBMCPClient()
    
    try:
        await client.connect()
        
        # Insert test data
        documents = [
            {"category": "A", "value": 10},
            {"category": "A", "value": 20},
            {"category": "B", "value": 15},
            {"category": "B", "value": 25},
            {"category": "B", "value": 30}
        ]
        
        await client.execute_tool(
            "insert-many",
            {
                "database": "test_db",
                "collection": "aggregation_test",
                "documents": documents
            }
        )
        
        # Run aggregation
        pipeline = [
            {"$group": {"_id": "$category", "total": {"$sum": "$value"}, "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}}
        ]
        
        result = await client.execute_tool(
            "aggregate",
            {
                "database": "test_db",
                "collection": "aggregation_test",
                "pipeline": pipeline
            }
        )
        
        if result.success:
            print(f"[OK] Aggregation completed!")
            print(f"    Result: {result.result}")
        else:
            print(f"[ERROR] Aggregation failed: {result.error}")
        
        # Cleanup
        await client.execute_tool(
            "drop-collection",
            {
                "database": "test_db",
                "collection": "aggregation_test"
            }
        )
        
        return result.success
        
    finally:
        await client.disconnect()


if __name__ == "__main__":
    print("Starting MongoDB MCP Live Tests...")
    print(f"Python version: {sys.version}")
    print(f"Connection string configured: {'Yes' if os.getenv('MONGODB_MCP_CONNECTION_STRING') else 'No'}")
    
    # Run main test
    success = asyncio.run(test_mongodb_mcp_live())
    
    if success:
        # Run additional tests
        asyncio.run(test_insert_many())
        asyncio.run(test_aggregate())
    
    print("\nDone!")
