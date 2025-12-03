"""
Simple MongoDB Test - Add Sample Data via MCP
==============================================

This test adds sample data to MongoDB using the MCP client.
Credentials are read from .env file.

Run:
    python tests/test_mongodb_sample_data.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from core.mcp.mongodb import MongoDBMCPClient


async def test_add_sample_data():
    """Add sample data to MongoDB using MCP"""
    
    print("=" * 60)
    print("MongoDB Sample Data Test (via MCP)")
    print("=" * 60)
    
    # Check env
    if not os.getenv("MONGODB_MCP_CONNECTION_STRING"):
        print("[ERROR] MONGODB_MCP_CONNECTION_STRING not set in .env")
        return False
    
    client = MongoDBMCPClient()
    
    try:
        # Connect
        print("\n[Step 1] Connecting to MongoDB MCP server...")
        connected = await client.connect()
        if not connected:
            print("[ERROR] Failed to connect")
            return False
        print("[OK] Connected!")
        
        # ========== ADD SAMPLE USERS ==========
        print("\n[Step 2] Adding sample users...")
        
        users = [
            {"name": "Zara Iqbal", "email": "zara@example.com", "age": 26, "city": "Multan"},
            {"name": "Bilal Shah", "email": "bilal@example.com", "age": 29, "city": "Faisalabad"},
            {"name": "Ayesha Tariq", "email": "ayesha@example.com", "age": 31, "city": "Rawalpindi"},
            {"name": "Omar Farooq", "email": "omar@example.com", "age": 27, "city": "Quetta"},
            {"name": "Hina Aslam", "email": "hina@example.com", "age": 33, "city": "Sialkot"},
        ]
        
        users_result = await client.execute_tool(
            "insert-many",
            {
                "database": "sample_database",
                "collection": "users",
                "documents": users
            }
        )
        
        if users_result.success:
            print(f"[OK] Inserted users: {users_result.result}")
        else:
            print(f"[ERROR] Insert users failed: {users_result.error}")
        
        # ========== ADD SAMPLE PRODUCTS ==========
        print("\n[Step 3] Adding sample products...")
        
        products = [
            {"name": "Smart Watch", "price": 35000, "category": "Electronics", "stock": 40},
            {"name": "Tablet", "price": 55000, "category": "Electronics", "stock": 30},
            {"name": "Bluetooth Speaker", "price": 8000, "category": "Electronics", "stock": 75},
            {"name": "Office Table", "price": 28000, "category": "Furniture", "stock": 20},
            {"name": "Bookshelf", "price": 15000, "category": "Furniture", "stock": 35},
        ]
        
        products_result = await client.execute_tool(
            "insert-many",
            {
                "database": "sample_database",
                "collection": "products",
                "documents": products
            }
        )
        
        if products_result.success:
            print(f"[OK] Inserted products: {products_result.result}")
        else:
            print(f"[ERROR] Insert products failed: {products_result.error}")
        
        # ========== ADD SAMPLE ORDERS ==========
        print("\n[Step 4] Adding sample orders...")
        
        orders = [
            {"user": "Zara Iqbal", "product": "Smart Watch", "quantity": 2, "total": 70000, "status": "completed"},
            {"user": "Bilal Shah", "product": "Tablet", "quantity": 1, "total": 55000, "status": "shipped"},
            {"user": "Ayesha Tariq", "product": "Bluetooth Speaker", "quantity": 3, "total": 24000, "status": "pending"},
            {"user": "Omar Farooq", "product": "Office Table", "quantity": 1, "total": 28000, "status": "completed"},
            {"user": "Hina Aslam", "product": "Bookshelf", "quantity": 2, "total": 30000, "status": "pending"},
        ]
        
        orders_result = await client.execute_tool(
            "insert-many",
            {
                "database": "sample_database",
                "collection": "orders",
                "documents": orders
            }
        )
        
        if orders_result.success:
            print(f"[OK] Inserted orders: {orders_result.result}")
        else:
            print(f"[ERROR] Insert orders failed: {orders_result.error}")
        
        # ========== VERIFY DATA - Find Users ==========
        print("\n" + "=" * 60)
        print("VERIFYING DATA")
        print("=" * 60)
        
        print("\n[Step 5] Finding all users...")
        find_users = await client.execute_tool(
            "find",
            {
                "database": "sample_database",
                "collection": "users",
                "filter": {}
            }
        )
        print(f"Users found: {find_users.result}")
        
        # Find users from Karachi
        print("\n[Step 6] Finding users from Karachi...")
        karachi_users = await client.execute_tool(
            "find",
            {
                "database": "sample_database",
                "collection": "users",
                "filter": {"city": "Karachi"}
            }
        )
        print(f"Karachi users: {karachi_users.result}")
        
        # Find cheap products
        print("\n[Step 7] Finding products under 20,000...")
        cheap_products = await client.execute_tool(
            "find",
            {
                "database": "sample_database",
                "collection": "products",
                "filter": {"price": {"$lt": 20000}}
            }
        )
        print(f"Cheap products: {cheap_products.result}")
        
        # Find pending orders
        print("\n[Step 8] Finding pending orders...")
        pending_orders = await client.execute_tool(
            "find",
            {
                "database": "sample_database",
                "collection": "orders",
                "filter": {"status": "pending"}
            }
        )
        print(f"Pending orders: {pending_orders.result}")
        
        print("\n" + "=" * 60)
        print("[OK] TEST COMPLETED!")
        print("=" * 60)
        print("\nData should be in MongoDB Atlas database: 'sample_database'")
        print("Collections: users, products, orders")
        print("\nNote: If you see 'You need to connect to a MongoDB instance',")
        print("the MCP server itself needs to connect. Check the tool discovery logs.")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        print("\nDisconnecting...")
        await client.disconnect()
        print("Done.")


if __name__ == "__main__":
    asyncio.run(test_add_sample_data())
