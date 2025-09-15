# Lab 05: Tavily MCP Integration (Hosted)

This lab integrates Tavily via the Model Context Protocol (MCP) using the hosted Tavily MCP server. We connect through FastMCP and npx mcp-remote. No local MCP server installation is required.

Outcome: The Researcher agent uses Tavily-backed search and scrape tools when a Tavily API key is configured.

---

## What you’ll build
- A working connection to Tavily’s hosted MCP server
- Researcher tools that call Tavily for search and content extraction

- A smoke test and a tiny end-to-end research run to verify everything

---

## Prerequisites
- Node.js and npx available (`node -v`, `npx -v`)
- uv-managed Python environment (already set up in this repo)
- Tavily API key (format: `tvly-...`)
- fastmcp dependency is already included in `libs/common/pyproject.toml`

---
## If starting from a clean branch (build it before seeing the code)
If your branch does not yet contain the Tavily MCP integration files, follow these steps. If the files already exist, you can skip to Step 1 below.

1) Install the dependency (if missing)
- Commands:
  - cd libs/common
  - uv add fastmcp
  - cd -
  - uv sync
- Quick check:
  - uv run --package libs/common python -c "import fastmcp, sys; sys.stdout.write(fastmcp.__version__)"

2) Create the Tavily MCP client wrapper (if missing)
- Path: libs/common/mcp_clients/tavily_mcp_client.py
- Minimal implementation:

```python
from __future__ import annotations
from typing import Any

from core.config_service import ConfigService
from fastmcp.client import Client
from fastmcp.client.stdio import NpxStdioTransport


class TavilyMCPClient:
    def __init__(self, config: ConfigService | None = None) -> None:
        self.config = config or ConfigService()
        self._client: Client | None = None

    def _remote_url(self) -> str:
        key = self.config.get("tavily.api_key") or self.config.get("tavili.api_key")
        if not key:
            raise RuntimeError("Missing tavily.api_key in secrets.yaml")
        return f"https://mcp.tavily.com/mcp/?tavilyApiKey={key}"

    async def connect(self) -> None:
        url = self._remote_url()
        self._client = Client(NpxStdioTransport(package="mcp-remote", args=[url]))
        await self._client.__aenter__()

    async def list_tools(self) -> list[str]:
        assert self._client is not None
        tools = await self._client.list_tools()
        return [t.name for t in tools.tools]

    async def resolve_tool(self, candidates: list[str]) -> str | None:
        names = set(await self.list_tools())
        for c in candidates:
            if c in names:
                return c
        return None

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        assert self._client is not None
        return await self._client.call_tool(name, arguments)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.__aexit__(None, None, None)
            self._client = None
```

3) Ensure the Researcher tool registry prefers Tavily (if needed)
- File: libs/common/agents/researcher_agent/researcher_tools.py
- Inside ResearcherToolRegistry.__init__, ensure this logic exists:

```python
if self.config_service.get("tavily.api_key") or self.config_service.get("tavili.api_key"):
    search_tool = TavilyMCPSearchTool(self.config_service)
    scrape_tool = TavilyMCPScrapeTool(self.config_service)
else:
    search_tool = ResearcherSearchTool()
    scrape_tool = ResearcherScraperTool()
self.tools.update({"search": search_tool, "scrape": scrape_tool})
```

4) Create a simple smoke test (if missing)
- Path: libs/common/mcp_clients/smoke_test_tavily.py
- Minimal script:

```python
import asyncio
from core.config_service import ConfigService
from mcp_clients.tavily_mcp_client import TavilyMCPClient

async def main():
    client = TavilyMCPClient(ConfigService())
    await client.connect()
    tools = await client.list_tools()
    print("TOOLS:", tools)
    name = await client.resolve_tool(["search", "tavily_search", "web_search"])
    assert name, "No Tavily search tool found"
    res = await client.call_tool(name, {"query": "OpenAI news today"})
    print("SEARCH RESULT OK", type(res).__name__)
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

Verification
- Node + npx available: node -v, npx -v
- Secrets file exists and contains tavily.api_key
- Smoke test runs without errors
- You can proceed to Step 1 below

---


## Step 1 — Configure your Tavily API key
Store your key in the shared secrets (not committed to git):

Path: `libs/common/secrets.yaml`

Example:

```yaml
# Tavily (Hosted MCP) configuration
# Only api_key is required

tavily:
  api_key: "tvly-..."

# Compatibility: the code also checks `tavili.api_key` if present
# tavili:
#   api_key: "tvly-..."
```

Notes:
- Do not commit `libs/common/secrets.yaml`
- The key is read via ConfigService at `tavily.api_key`

---

## Step 2 — Verify connectivity (smoke test)
Run the built-in Tavily MCP smoke test:

```bash
uv run python libs/common/mcp_clients/smoke_test_tavily.py
```
Copy/paste verification
- Run:
  - node -v && npx -v
  - uv run python libs/common/mcp_clients/smoke_test_tavily.py
- Success signal:
  - Prints a TOOLS list, e.g. `TOOLS: ['search', 'read', ...]`
  - Prints a search result type, e.g. `SEARCH RESULT TYPE: CallToolResult` (or `dict` / `list`)
  - No errors or tracebacks


Expected behavior:
- Connects via `npx mcp-remote` to https://mcp.tavily.com/mcp/
- Lists available Tavily tools (e.g., search/extract)
- Executes a sample search and prints a short result summary
- Ends with a success message like: “RESULT: ✅ All checks passed”

If this passes, your Tavily MCP integration is working.
Tip: If the file doesn’t exist yet, create it with this minimal content:

```python
import asyncio
from core.config_service import ConfigService
from mcp_clients.tavily_mcp_client import TavilyMCPClient

async def main():
    client = TavilyMCPClient(ConfigService())
    await client.connect()

    tools = await client.list_tools()
    print("TOOLS:", tools)

    # Resolve a suitable search tool name from the remote Tavily MCP
    name = await client.resolve_tool(["search", "tavily_search", "web_search"])
    assert name, "No Tavily search tool found"
    res = await client.call_tool(name, {"query": "OpenAI news today"})
    print("SEARCH RESULT TYPE:", type(res).__name__)

    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```


---

Common pitfalls (Step 2)
- npx not found: Install Node.js so `npx` is available; verify with `node -v` and `npx -v`.
- Missing/invalid API key: Ensure libs/common/secrets.yaml contains `tavily.api_key: tvly-...`.
- No tools returned: Network/firewall may block https://mcp.tavily.com/mcp/; try from a different network.
- Timeout/hanging: Re-run the command; if persistent, confirm your Tavily key is valid and internet is stable.

## Step 3 — Tiny end-to-end research run
Run a minimal end-to-end task through the Researcher agent:


```bash
uv run python libs/common/agents/researcher_agent/research_e2e.py
```
Copy/paste verification
- Run:
  - uv run python libs/common/agents/researcher_agent/research_e2e.py
- Success signal:
  - Prints TOPIC and TOP SOURCE lines
  - Prints SUMMARY PREVIEW followed by DONE
  - No errors or tracebacks


What you should see:
- The agent runs a small topic query (e.g., recent GPT-4o updates)
- It uses the Tavily-backed tools under the hood
- A compact printed summary with a few sources and extracted facts

If the file doesn’t exist yet, create it with this minimal script:

```python
import asyncio
from core.config_service import ConfigService
from agents.researcher_agent.researcher_tools import (
    ResearcherToolRegistry,
    SearchParams,
    ScrapeParams,
)

async def main():
    cfg = ConfigService()
    registry = ResearcherToolRegistry(cfg)

    topic = "Latest AI safety news (last 7 days)"
    print("TOPIC:", topic)

    # 1) Search
    search_tool = registry.tools["search"]
    search_res = await search_tool.execute(SearchParams(query=topic))
    if not search_res.success or not search_res.sources:
        print("Search failed or no sources:", search_res.error)
        return

    top = search_res.sources[0]
    print("TOP SOURCE:", getattr(top, "title", None) or getattr(top, "url", "(no url)"))

    # 2) Scrape the top result
    scrape_tool = registry.tools["scrape"]
    scrape_res = await scrape_tool.execute(ScrapeParams(url=top.url))
    if not scrape_res.success:
        print("Scrape failed:", scrape_res.error)
        return

    summary = scrape_res.summary or ""
    text_preview = summary[:500] + ("..." if len(summary) > 500 else "")
    print("SUMMARY PREVIEW:\n", text_preview)
    print("DONE")

if __name__ == "__main__":
    asyncio.run(main())
```

Common pitfalls (Step 3)
- Search returns no sources: Try a broader query or a different recent topic.
- Scrape fails for a source: Some sites block automated readers or are paywalled. Pick another source from the list.
- Very slow or timeouts: Check network stability; you can re-run with a simpler topic.
- Key not picked up: Ensure `tavily.api_key` is present in libs/common/secrets.yaml and restart your shell if environment variables changed.

### Optional extension — Add Tavily tools to the Reporter agent (not required for Lab 05)
If you want the Reporter to perform web research directly (beyond this lab’s scope), add `search` and `scrape` tools to its registry. This mirrors the Researcher’s configuration and uses Tavily MCP when `tavily.api_key` is present; otherwise it falls back.

Location: `libs/common/agents/reporter_agent/reporter_tools.py` (inside `ReporterToolRegistry.__init__`)

```python
# Local imports avoid circular dependencies
from agents.researcher_agent.researcher_tools import (
    TavilyMCPSearchTool, TavilyMCPScrapeTool,
    ResearcherSearchTool, ResearcherScraperTool,
)

# Existing tools
self.tools = {
    "fetch_from_memory": FetchFromMemoryTool(),
    "use_llm": UseLLMTool(),
}

# Add web research tools under standard names
if self.config_service.get("tavily.api_key") or self.config_service.get("tavili.api_key"):
    self.tools["search"] = TavilyMCPSearchTool(self.config_service)
    self.tools["scrape"] = TavilyMCPScrapeTool(self.config_service)
else:
    self.tools["search"] = ResearcherSearchTool()
    self.tools["scrape"] = ResearcherScraperTool()
```

Notes:
- This change is optional. Lab 05 only requires the Researcher to use Tavily.
- When added, the Reporter’s system prompt will include `search` and `scrape` automatically, and tool calls will route through the same logic used by the Researcher.

---

---

## Code examples (key integration points)

### 1) Tavily MCP client wrapper
Location: `libs/common/mcp_clients/tavily_mcp_client.py`

```python
class TavilyMCPClient:
    def _build_command(self) -> tuple[str, dict[str, str]]:
        api_key = self._get_api_key()
        base_url = "https://mcp.tavily.com/mcp/"
        full_url = f"{base_url}?tavilyApiKey={api_key}"
        return full_url, {}

    async def connect(self) -> None:
        transport = NpxStdioTransport(package="mcp-remote", args=[remote_url], env_vars=env)
        client = Client(transport)
        await client.__aenter__()
        self._client = client

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        assert self._client is not None
        result = await self._client.call_tool(name, arguments)
        return result
```

### 2) Researcher: conditional tool registration
Location: `libs/common/agents/researcher_agent/researcher_tools.py`

```python
class ResearcherToolRegistry:
    def __init__(self, config_service: ConfigService | None = None) -> None:
        self.config_service = config_service or ConfigService()
        if self.config_service.get("tavily.api_key") or self.config_service.get("tavili.api_key"):
            search_tool = TavilyMCPSearchTool(self.config_service)
            scrape_tool = TavilyMCPScrapeTool(self.config_service)
        else:
            search_tool = ResearcherSearchTool()
            scrape_tool = ResearcherScraperTool()
        self.tools = {"search": search_tool, "scrape": scrape_tool}
```


---

## Troubleshooting
- Node/npx missing
  - Ensure Node and npx are available: `node -v` and `npx -v`

- API key issues
  - Confirm `libs/common/secrets.yaml` has `tavily.api_key: tvly-...`
  - Verify the key is valid and not expired

- “No results” or empty tool output
  - Re-run the smoke test; confirm tools list appears and a basic search works
  - Ensure your environment can reach https://mcp.tavily.com/mcp/

- fastmcp import errors
  - The dependency is already declared in `libs/common/pyproject.toml`; if your local env is stale, run `uv sync`

---

## Key files
- Connection wrapper: `libs/common/mcp_clients/tavily_mcp_client.py`
- Smoke test: `libs/common/mcp_clients/smoke_test_tavily.py`
- Research tools (provider switching): `libs/common/agents/researcher_agent/researcher_tools.py`

- Tiny end-to-end runner: `libs/common/agents/researcher_agent/research_e2e.py`

---

## Security
- Never commit secrets: keep `libs/common/secrets.yaml` out of version control
- API keys are loaded via the shared ConfigService

---

That’s it. With your API key set and the smoke test passing, your Researcher agent uses Tavily’s hosted MCP today.
yes