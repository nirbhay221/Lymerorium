"""
MCP client wrapper for SwarmCore tool phase.

Provides get_mcp_tools(agent_name) - returns the filtered tool list for an agent
from all connected MCP servers (local swarm-tools server + any external servers).

Usage (async context):
    from mcp_client import call_mcp_tool
    result = await call_mcp_tool("sympy_solve", {"expression": "x**2 - 5*x + 6 = 0"})

The tool_phase_node in simulation.py uses the sync dispatch_tool() path by default.
Switch to MCP by calling init_mcp_session() at startup and replacing dispatch_tool
calls with call_mcp_tool_sync().
"""

import asyncio
import os
import sys
from functools import lru_cache

sys.path.insert(0, os.path.dirname(__file__))

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False

from mcp_server import EXTERNAL_MCP_SERVERS


# Local swarm-tools server config (stdio - same process spawns it)
_LOCAL_SERVER = {
    "swarm_tools": {
        "command": sys.executable,
        "args": [os.path.join(os.path.dirname(__file__), "mcp_server.py")],
        "transport": "stdio",
    }
}

_ALL_SERVERS = {**_LOCAL_SERVER, **EXTERNAL_MCP_SERVERS}


async def get_mcp_tools(agent_name: str | None = None) -> list:
    """
    Connect to all MCP servers and return LangChain-compatible tool objects.
    Optionally filter by agent_name using _WORKER_TOOLS.
    """
    if not _MCP_AVAILABLE:
        raise RuntimeError(
            "langchain-mcp-adapters not installed - run: pip install langchain-mcp-adapters"
        )
    from simulation import _WORKER_TOOLS
    async with MultiServerMCPClient(_ALL_SERVERS) as client:
        all_tools = await client.get_tools()
        if agent_name is None:
            return all_tools
        allowed = set(_WORKER_TOOLS.get(agent_name, []))
        return [t for t in all_tools if t.name in allowed]


async def call_mcp_tool(tool_name: str, args: dict) -> str:
    """Call a single tool via MCP and return its string result."""
    if not _MCP_AVAILABLE:
        # Graceful fallback to direct dispatch
        import tools
        fn = tools.TOOL_MAP.get(tool_name)
        return fn(**args) if fn else f"[mcp_client: tool '{tool_name}' not found]"
    async with MultiServerMCPClient(_ALL_SERVERS) as client:
        all_tools = await client.get_tools()
        tool = next((t for t in all_tools if t.name == tool_name), None)
        if not tool:
            return f"[mcp_client: tool '{tool_name}' not found in any server]"
        return await tool.ainvoke(args)


def call_mcp_tool_sync(tool_name: str, args: dict) -> str:
    """Sync wrapper around call_mcp_tool - safe to call from ThreadPoolExecutor."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already inside an event loop (e.g. async Flask) - use a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(asyncio.run, call_mcp_tool(tool_name, args)).result()
        return loop.run_until_complete(call_mcp_tool(tool_name, args))
    except Exception as e:
        # Always fall back to direct dispatch - MCP is transport, not required
        import tools
        fn = tools.TOOL_MAP.get(tool_name)
        return fn(**args) if fn else f"[mcp_client error: {e}]"
