"""
SwarmCore MCP Server - exposes all swarm tools as MCP-compliant endpoints.

FastMCP auto-generates JSON schemas from each function's type hints + docstring,
so any MCP client (LangGraph MultiServerMCPClient, or other agents)
can discover and call these tools without hardcoded mappings.

Run standalone (stdio transport):
    python swarm_core/mcp_server.py

Run from LangGraph / eval:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    client = MultiServerMCPClient(MCP_SERVERS)
    # see mcp_client.py for the full async wrapper
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

try:
    from fastmcp import FastMCP
except ImportError:
    raise SystemExit("FastMCP not installed - run: pip install fastmcp")

from tools import (
    # Web search
    web_search_tool,
    web_search_and_fetch,
    fetch_url,
    tavily_search,
    # Structured knowledge
    wiki_summary,
    wikidata_entity,
    openalex_search,
    semantic_scholar_search,
    # Math / computation
    python_eval,
    sympy_solve,
    pint_convert,
    wolfram_query,
    frankfurter_fx,
    # Knowledge graph
    search_knowledge_graph,
    get_agent_opinions,
    get_entity_relationships,
)

mcp = FastMCP(
    "swarm-tools",
    description=(
        "Tool server for the SwarmCore multi-agent debate system. "
        "Provides web search, structured knowledge lookup, math solving, "
        "unit conversion, academic paper search, and entity fact retrieval."
    ),
)

# ── Web search ───────────────────────────────────────────────────────────────
mcp.tool()(web_search_tool)
mcp.tool()(web_search_and_fetch)
mcp.tool()(fetch_url)
mcp.tool()(tavily_search)

# ── Structured knowledge ─────────────────────────────────────────────────────
mcp.tool()(wiki_summary)
mcp.tool()(wikidata_entity)
mcp.tool()(openalex_search)
mcp.tool()(semantic_scholar_search)

# ── Math / computation ───────────────────────────────────────────────────────
mcp.tool()(python_eval)
mcp.tool()(sympy_solve)
mcp.tool()(pint_convert)
mcp.tool()(wolfram_query)
mcp.tool()(frankfurter_fx)

# ── Knowledge graph ──────────────────────────────────────────────────────────
mcp.tool()(search_knowledge_graph)
mcp.tool()(get_agent_opinions)
mcp.tool()(get_entity_relationships)


# ── External MCP server registry ─────────────────────────────────────────────
# These are community / official MCP servers that plug in alongside this one.
# Each entry is passed directly to MultiServerMCPClient (see mcp_client.py).
# Set to {} to disable; add entries to plug in new servers without any code changes.
EXTERNAL_MCP_SERVERS: dict = {
    # Wolfram official MCP - richer output than our REST wrapper, same App ID
    # Uncomment when wolfram-mcp is published to PyPI / npm:
    # "wolfram_official": {
    #     "command": "npx",
    #     "args": ["-y", "@wolfram/mcp-server"],
    #     "env": {"WOLFRAM_APP_ID": os.environ.get("WOLFRAM_APP_ID", "")},
    #     "transport": "stdio",
    # },
    #
    # arXiv paper search MCP server (optional, community plugin):
    # "arxiv": {
    #     "command": "python",
    #     "args": ["path/to/arxiv_mcp_server.py"],
    #     "transport": "stdio",
    # },
    #
    # Exa semantic search MCP (if you get an Exa API key):
    # "exa": {
    #     "command": "npx",
    #     "args": ["-y", "exa-mcp-server"],
    #     "env": {"EXA_API_KEY": os.environ.get("EXA_API_KEY", "")},
    #     "transport": "stdio",
    # },
}

if __name__ == "__main__":
    mcp.run()   # stdio transport - works with MultiServerMCPClient out of the box
