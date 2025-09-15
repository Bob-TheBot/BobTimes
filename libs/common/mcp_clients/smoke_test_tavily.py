#!/usr/bin/env python3
"""Tavily MCP smoke test.

This test validates:
1) MCP connectivity via TavilyMCPClient (lists tools, runs a sample search)
2) ResearcherToolRegistry integration (executes 'search' tool and checks provider)

Safe to run: read-only network calls; no data mutations.
"""
from __future__ import annotations

import asyncio

from core.config_service import ConfigService
from core.logging_service import get_logger

logger = get_logger(__name__)


async def test_mcp_client(config: ConfigService) -> bool:
    print("\n[1/2] Testing direct MCP client...")
    try:
        from mcp_clients.tavily_mcp_client import TavilyMCPClient  # type: ignore
    except Exception as e:
        print(f"  ❌ TavilyMCPClient import failed: {e}")
        return False

    client = TavilyMCPClient(config)
    try:
        await client.connect()
        tools = await client.list_tools()
        print(f"  ✅ Connected. Tools available: {len(tools)}")
        print("   → ", ", ".join(tools[:6]))

        tool_name = await client.resolve_tool(["search", "tavily_search", "web_search"])
        if not tool_name:
            print("  ❌ No Tavily search tool available")
            return False

        print(f"  🔧 Using tool: {tool_name}")
        raw = await client.call_tool(tool_name, {"query": "latest AI news"})
        # Print a compact preview
        if isinstance(raw, dict):
            size = len(raw.get("results", [])) if "results" in raw else len(str(raw))
            print(f"  ✅ Search call returned dict with size: {size}")
        elif isinstance(raw, list):
            print(f"  ✅ Search call returned list with {len(raw)} items")
        else:
            print(f"  ✅ Search call returned type {type(raw).__name__}")
        return True
    except Exception as e:
        print(f"  ❌ MCP client test failed: {e}")
        return False
    finally:
        try:
            await client.close()
        except Exception:
            pass


async def test_researcher_tool_registry(config: ConfigService) -> bool:
    print("\n[2/2] Testing ResearcherToolRegistry integration...")
    try:
        from agents.researcher_agent.researcher_tools import (
            ResearcherToolRegistry,
            SearchParams,
            SearchType,
        )
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

    registry = ResearcherToolRegistry(config)
    search_tool = registry.get_tool_by_name("search")
    if not search_tool:
        print("  ❌ 'search' tool not found in registry")
        return False

    try:
        params = SearchParams(query="OpenAI GPT news", search_type=SearchType.NEWS, time_limit="w", max_results=2)
        result = await search_tool.execute(params)
        provider = (result.metadata or {}).get("provider") if hasattr(result, "metadata") else None
        print(f"  ✅ Tool executed: success={result.success}, sources={len(result.sources)} provider={provider}")
        if not result.success:
            print(f"     Error: {result.error}")
        return bool(result.success)
    except Exception as e:
        print(f"  ❌ Researcher tool execution failed: {e}")
        return False


async def main() -> int:
    config = ConfigService()
    api_key_present = bool(config.get("tavily.api_key") or config.get("tavili.api_key"))
    print("Tavily API key present:", api_key_present)
    if not api_key_present:
        print("❌ Missing tavily.api_key (or 'tavili.api_key') in libs/secrets.yaml")
        return 1

    ok_client = await test_mcp_client(config)
    ok_registry = await test_researcher_tool_registry(config)

    all_ok = ok_client and ok_registry
    print("\nRESULT:", "✅ All checks passed" if all_ok else "❌ Some checks failed")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

