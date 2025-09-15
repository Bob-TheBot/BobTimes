"""Lightweight Tavily MCP client wrapper using FastMCP.

- Builds an MCP Remote command to connect to Tavily's hosted MCP server
- Reads API key from ConfigService (secrets.yaml: tavily.api_key)
- Provides minimal helpers to list tools and call a tool safely

This module is optional; it degrades gracefully when fastmcp is not installed.
"""
from __future__ import annotations

from typing import Any

from core.config_service import ConfigService
from core.logging_service import get_logger

logger = get_logger(__name__)

try:  # Optional dependency
    from fastmcp import FastMCPClient
except Exception:  # pragma: no cover - missing optional dep
    FastMCPClient = None  


class TavilyMCPClient:
    """Minimal FastMCP client for Tavily's remote MCP server.

    Usage pattern:
        client = TavilyMCPClient(config)
        await client.connect()
        tools = await client.list_tools()
        result = await client.call_tool("search", {"query": "..."})
        await client.close()
    """

    def __init__(self, config_service: ConfigService, timeout: int = 60) -> None:
        self.config_service = config_service
        self.timeout = timeout
        self._client: Optional[FastMCPClient] = None  # type: ignore

    def _build_command(self) -> tuple[str, list[str], dict[str, str]]:
        api_key = self._get_api_key()
        if not api_key:
            raise RuntimeError(
                "Tavily API key is not configured. Add tavily.api_key to libs/common/secrets.yaml"
            )
        # Use MCP Remote to connect to the hosted Tavily MCP HTTP endpoint
        base_url = "https://mcp.tavily.com/mcp/"
        full_url = f"{base_url}?tavilyApiKey={api_key}"
        command = "npx"
        args = ["-y", "mcp-remote", full_url]
        env: dict[str, str] = {}
        return command, args, env

    def _get_api_key(self) -> str:
        value = self.config_service.get("tavily.api_key")
        return str(value) if value else ""

    async def connect(self) -> None:
        if FastMCPClient is None:
            raise RuntimeError(
                "fastmcp is not installed. Please run: uv add fastmcp (in the workspace root)"
            )
        if self._client is not None:
            return
        command, args, env = self._build_command()
        client = FastMCPClient(command=command, args=args, env=env, timeout=self.timeout)  # type: ignore
        await client.connect()
        self._client = client
        logger.info("Connected to Tavily MCP via mcp-remote", command=command)

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.close()
            finally:
                self._client = None

    async def list_tools(self) -> list[str]:
        if self._client is None:
            raise RuntimeError("Client not connected. Call connect() first.")
        tools = await self._client.list_tools()  # type: ignore
        # tools may be list of objects; normalize to names
        names: list[str] = []
        for t in tools:
            name = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else None)
            if name:
                names.append(str(name))
        return names

    async def resolve_tool(self, candidates: list[str]) -> str | None:
        try:
            available = await self.list_tools()
            for cand in candidates:
                if cand in available:
                    return cand
            # also try fuzzy contains
            for cand in candidates:
                for name in available:
                    if cand in name:
                        return name
            return None
        except Exception as e:
            logger.error("Failed to resolve Tavily MCP tool", error=str(e))
            return None

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if self._client is None:
            raise RuntimeError("Client not connected. Call connect() first.")
        return await self._client.call_tool(name, arguments=arguments)  # type: ignore

