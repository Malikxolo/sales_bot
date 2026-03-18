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

# Zapier meta-tools that configure the MCP server itself — never expose to users
_META_TOOLS_TO_EXCLUDE = {"add_tools", "edit_tools"}


class ZapierMCPManager:
    """
    Manages connection to Zapier's MCP server.
    - Authenticates via Bearer token
    - Fetches available tools on initialization (excludes meta-tools)
    - Executes tools by name with given arguments
    - Detects hidden clarification questions / errors in Zapier responses
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
            event: <n>
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

    def _extract_clarification_or_error(self, result: Any) -> Optional[str]:
        """
        Detect when Zapier returns success=True but embeds a clarification
        question or hidden error inside the result content.

        Zapier sometimes responds with things like:
          - "Which calendar do you want to use?"
          - {"isError": true, "error": "authentication failed"}
          - {"question": "Which account should I use?"}

        Returns the question/error string if found, None if result is clean.
        """
        if not result:
            return None

        try:
            if isinstance(result, dict):

                # Top-level isError flag
                if result.get("isError"):
                    content = result.get("content", [])
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text = item.get("text", "")
                                try:
                                    parsed = json.loads(text)
                                    if isinstance(parsed, dict) and parsed.get("error"):
                                        return parsed["error"]
                                except json.JSONDecodeError:
                                    pass
                    return "Zapier returned an error"

                # Walk content blocks looking for embedded questions/errors
                content = result.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text = item.get("text", "")
                            try:
                                parsed = json.loads(text)
                                if isinstance(parsed, dict):
                                    if parsed.get("isError"):
                                        return parsed.get("error", "Zapier returned an error")
                                    error = parsed.get("error", "")
                                    if error and ("Question:" in str(error) or "?" in str(error)):
                                        return error
                                    question = parsed.get("question", "")
                                    if question:
                                        return question
                            except json.JSONDecodeError:
                                # Raw text — check for question patterns
                                if "Question:" in text or ("?" in text and "which" in text.lower()):
                                    return text

            elif isinstance(result, str):
                try:
                    parsed = json.loads(result)
                    if isinstance(parsed, dict):
                        if parsed.get("isError"):
                            return parsed.get("error", "Zapier returned an error")
                        error = parsed.get("error", "")
                        if error and ("Question:" in str(error) or "?" in str(error)):
                            return error
                except json.JSONDecodeError:
                    if "Question:" in result or ("?" in result and "which" in result.lower()):
                        return result

        except Exception as e:
            logger.debug(f"Error in _extract_clarification_or_error: {e}")

        return None

    async def initialize(self):
        """
        Fetch the tool list from Zapier and cache it locally.
        Excludes meta-tools (add_tools, edit_tools) that configure MCP itself.
        Call this once from ToolManager during app startup (lifespan).
        """
        try:
            data = await self._jsonrpc_sse("tools/list")

            if "error" in data:
                raise Exception(f"tools/list error: {data['error']}")

            tools = data.get("result", {}).get("tools", [])

            # Filter out meta-tools that should never be exposed to users
            filtered_tools = [t for t in tools if t["name"] not in _META_TOOLS_TO_EXCLUDE]
            skipped = len(tools) - len(filtered_tools)
            if skipped:
                skipped_names = _META_TOOLS_TO_EXCLUDE & {t["name"] for t in tools}
                logger.info(f"   Skipped {skipped} Zapier meta-tool(s): {skipped_names}")

            self.available_tools = {t["name"]: t for t in filtered_tools}
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
                "success":                bool,
                "result":                 str,
                "error":                  str,
                "tool_name":              str,
                "needs_clarification":    bool  (only present when Zapier asks a question)
                "clarification_question": str   (only present when needs_clarification=True)
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

            raw_result = data.get("result", {})
            content_blocks = raw_result.get("content", [])
            result_text = "\n".join(
                block.get("text", "")
                for block in content_blocks
                if block.get("type") == "text"
            )

            # Detect hidden clarification questions / errors before claiming success
            clarification = self._extract_clarification_or_error(raw_result)
            if clarification:
                logger.warning(f"⚠️ Zapier needs clarification for '{tool_name}': {clarification}")
                self._stats["calls_failed"] += 1
                return {
                    "success": False,
                    "error": f"Zapier needs more information: {clarification}",
                    "needs_clarification": True,
                    "clarification_question": clarification,
                    "tool_name": tool_name
                }

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

    def get_tool_required_params(self) -> Dict[str, List[str]]:
        """
        Return required parameter names per tool, extracted from the
        inputSchema that Zapier returns during tools/list.
        Used by SalesAgent to build better LLM prompts.
        """
        result = {}
        for name, schema in self.available_tools.items():
            input_schema = schema.get("inputSchema", {})
            required = input_schema.get("required", [])
            result[name] = required
        return result

    def get_tool_names(self) -> List[str]:
        return list(self.available_tools.keys())

    def get_stats(self) -> Dict[str, Any]:
        return {**self._stats, "available_tools": list(self.available_tools.keys())}

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("🔒 Zapier MCP session closed")