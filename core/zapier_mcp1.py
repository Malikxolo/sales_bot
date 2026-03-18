"""
Zapier MCP Client
Connects to Zapier's MCP server using Bearer token auth.
Endpoint: https://mcp.zapier.com/api/v1/connect
Auth:      Authorization: Bearer <token>
Transport: HTTP SSE — server streams text/event-stream, we read and parse JSON-RPC from it
"""

import aiohttp
import json
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class ZapierMCPManager:
    """
    Manages connection to Zapier's MCP server.
    - Authenticates via Bearer token
    - Fetches available tools on initialization
    - Executes tools by name with given arguments
    - Tracks usage stats
    """

    MCP_URL = "https://mcp.zapier.com/api/v1/connect"

    def __init__(self, token: str):
        self.token = token
        self.session: Optional[aiohttp.ClientSession] = None
        self.available_tools: Dict[str, Dict] = {}
        self._stats = {
            "tools_fetched": 0,
            "calls_success": 0,
            "calls_failed": 0,
        }

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        if not self.session or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    async def _jsonrpc_sse(self, method: str, params: Dict = None, request_id: int = 1) -> Dict:
        """
        Send a JSON-RPC 2.0 request and read the SSE stream response.
        Zapier responds with text/event-stream — each line is either:
            data: <json>
            event: <name>
            (blank line = end of event)
        We collect all data: lines and return the first one that contains
        a valid JSON-RPC result or error.
        """
        session = await self._get_session()
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {}
        }

        async with session.post(
            self.MCP_URL,
            headers=self._headers(),
            json=payload
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Zapier MCP HTTP {resp.status} on '{method}': {text[:300]}")

            # Read SSE stream line by line
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8").rstrip("\r\n")

                if not line.startswith("data:"):
                    continue

                data_str = line[len("data:"):].strip()
                if not data_str:
                    continue

                try:
                    parsed = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Accept any JSON-RPC response (result or error)
                if "result" in parsed or "error" in parsed:
                    return parsed

        raise Exception(f"Zapier MCP: no JSON-RPC response received for '{method}'")

    async def initialize(self):
        """
        Fetch the tool list from Zapier and cache it locally.
        Call this once from ToolManager during app startup (lifespan).
        """
        try:
            data = await self._jsonrpc_sse("tools/list")

            if "error" in data:
                raise Exception(f"tools/list error: {data['error']}")

            tools = data.get("result", {}).get("tools", [])
            self.available_tools = {t["name"]: t for t in tools}
            self._stats["tools_fetched"] = len(self.available_tools)

            logger.info(f"✅ Zapier MCP initialized — {len(self.available_tools)} tools available")
            if self.available_tools:
                logger.info(f"   Tools: {', '.join(self.available_tools.keys())}")
            else:
                logger.warning(
                    "⚠️ Zapier MCP connected but returned 0 tools — "
                    "check your enabled Zaps in Zapier"
                )

        except Exception as e:
            logger.error(f"❌ Zapier MCP initialization failed: {e}")

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a Zapier MCP tool by name.

        Returns:
            {
                "success": bool,
                "result":  str,
                "error":   str,
                "tool_name": str
            }
        """
        if tool_name not in self.available_tools:
            return {
                "success": False,
                "error": f"Zapier tool '{tool_name}' not found. Available: {list(self.available_tools.keys())}",
                "tool_name": tool_name
            }

        try:
            data = await self._jsonrpc_sse(
                method="tools/call",
                params={"name": tool_name, "arguments": arguments},
                request_id=2
            )

            if "error" in data:
                self._stats["calls_failed"] += 1
                return {
                    "success": False,
                    "error": f"Zapier MCP error: {data['error']}",
                    "tool_name": tool_name
                }

            content_blocks = data.get("result", {}).get("content", [])
            result_text = "\n".join(
                block.get("text", "")
                for block in content_blocks
                if block.get("type") == "text"
            )

            self._stats["calls_success"] += 1
            logger.info(f"✅ Zapier tool '{tool_name}' executed successfully")

            return {
                "success": True,
                "result": result_text,
                "tool_name": tool_name
            }

        except Exception as e:
            self._stats["calls_failed"] += 1
            logger.error(f"❌ Zapier tool '{tool_name}' failed: {e}")
            return {
                "success": False,
                "error": f"Zapier tool execution failed: {str(e)}",
                "tool_name": tool_name
            }

    def get_tool_descriptions(self) -> Dict[str, str]:
        return {
            name: schema.get("description", "No description")
            for name, schema in self.available_tools.items()
        }

    def get_tool_names(self) -> List[str]:
        return list(self.available_tools.keys())

    def get_stats(self) -> Dict[str, Any]:
        return {**self._stats, "available_tools": list(self.available_tools.keys())}

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("🔒 Zapier MCP session closed")