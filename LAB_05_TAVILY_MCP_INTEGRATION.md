# Lab 5: Tavily MCP Integration - Advanced Research with Model Context Protocol

Welcome to Lab 5! In this lab, you'll integrate Tavily MCP (Model Context Protocol) as a powerful research tool using FastMCP client. This adds advanced web search and research capabilities to your existing three-agent system while maintaining seamless integration with current research workflows.

## 🎯 Lab Objectives

By the end of this lab, you will:
- ✅ Set up FastMCP client for MCP server communication
- ✅ Integrate Tavily MCP server for advanced web research
- ✅ Create MCP research tool compatible with existing system
- ✅ Format MCP outputs to match current research tool standards
- ✅ Test MCP integration with the three-agent workflow
- ✅ Understand MCP architecture and best practices

## 📋 Prerequisites

- ✅ Completed Labs 1-4 (Basic setup through Researcher Agent)
- ✅ Working DevContainer or local development environment
- ✅ Understanding of the three-agent workflow (Researcher → Editor → Reporter)
- ✅ Familiarity with research tools and data models
- ✅ Tavily API key (sign up at https://tavily.com)

## 🏗️ Step 1: Understanding MCP Architecture

### 1.1 Model Context Protocol Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   BobTimes      │    │   FastMCP       │    │   Tavily MCP    │
│   Research      │────│   Client        │────│   Server        │
│   Tool          │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Research Models │    │ MCP Protocol    │    │ Tavily API      │
│ Integration     │    │ Communication   │    │ Web Search      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 1.2 MCP Integration Benefits

**Advanced Research Capabilities:**
- **Real-time Web Search**: Access to current web information
- **Content Summarization**: AI-powered content analysis
- **Source Verification**: Credible source identification
- **Multi-query Research**: Complex research workflows

**Seamless Integration:**
- **Compatible Output Format**: Matches existing research tool outputs
- **Unified Research Interface**: Works with current three-agent system
- **Scalable Architecture**: Easy to add more MCP servers
- **Error Handling**: Robust fallback mechanisms

### 1.3 Tavily MCP Capabilities

**Search Functions:**
- **Web Search**: Comprehensive web search with AI filtering
- **News Search**: Recent news and current events
- **Academic Search**: Scholarly articles and research papers
- **Content Extraction**: Full content retrieval from URLs

**AI Features:**
- **Content Summarization**: AI-generated summaries
- **Relevance Scoring**: Automatic relevance assessment
- **Source Credibility**: Source reliability evaluation
- **Topic Clustering**: Related content grouping

## 🔧 Step 2: Setup FastMCP Client and Dependencies

### 2.1 Install Required Dependencies

Add to your `pyproject.toml`:

```toml
[project]
dependencies = [
    # ... existing dependencies ...
    "fastmcp>=0.1.0",
    "httpx>=0.25.0",
    "websockets>=11.0",
    "pydantic>=2.0.0",
]
```

Install dependencies:

```bash
# In DevContainer terminal
uv sync
```

### 2.2 Configure Tavily API Key

Add Tavily configuration to your secrets file `libs/common/secrets.yaml`:

```yaml
# Tavily MCP Configuration
tavily:
  api_key: "your_tavily_api_key_here"
  base_url: "https://api.tavily.com"
  max_results: 10
  search_depth: "advanced"  # basic, advanced, or comprehensive

# MCP Server Configuration
mcp:
  tavily_server:
    command: "npx"
    args: ["@tavily/mcp-server"]
    env:
      TAVILY_API_KEY: "${tavily.api_key}"
    timeout: 30
    max_retries: 3
```

Add environment variables to `.env.development`:

```bash
# Tavily MCP Settings
TAVILY_ENABLED=true
TAVILY_MAX_RESULTS=10
TAVILY_SEARCH_DEPTH=advanced
TAVILY_TIMEOUT=30

# MCP Client Settings
MCP_CLIENT_TIMEOUT=60
MCP_MAX_CONNECTIONS=5
MCP_RETRY_ATTEMPTS=3
```

### 2.3 Install Tavily MCP Server

Install the Tavily MCP server globally:

```bash
# Install Tavily MCP server
npm install -g @tavily/mcp-server

# Verify installation
npx @tavily/mcp-server --version
```

## 🛠️ Step 3: Create MCP Client Infrastructure

### 3.1 Create MCP Client Service

Create the file `libs/common/utils/mcp_client_service.py`:

```python
"""MCP Client Service for connecting to MCP servers."""

import asyncio
import json
from typing import Any, Dict, List, Optional
from datetime import datetime

from core.config_service import ConfigService
from core.logging_service import get_logger
from pydantic import BaseModel, Field

try:
    from fastmcp import FastMCPClient
    from fastmcp.exceptions import MCPError, ConnectionError as MCPConnectionError
except ImportError:
    FastMCPClient = None
    MCPError = Exception
    MCPConnectionError = Exception

logger = get_logger(__name__)


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server."""
    name: str = Field(description="Server name")
    command: str = Field(description="Command to start server")
    args: List[str] = Field(default_factory=list, description="Command arguments")
    env: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    timeout: int = Field(default=30, description="Connection timeout")
    max_retries: int = Field(default=3, description="Maximum retry attempts")


class MCPToolCall(BaseModel):
    """MCP tool call request."""
    tool_name: str = Field(description="Name of the MCP tool to call")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")


class MCPToolResult(BaseModel):
    """MCP tool call result."""
    success: bool = Field(description="Whether the call was successful")
    result: Any = Field(None, description="Tool result data")
    error: Optional[str] = Field(None, description="Error message if failed")
    execution_time: float = Field(default=0.0, description="Execution time in seconds")
    server_name: str = Field(description="Name of the MCP server used")


class MCPClientService:
    """Service for managing MCP client connections and tool calls."""

    def __init__(self, config_service: ConfigService):
        """Initialize MCP client service."""
        self.config_service = config_service
        self.clients: Dict[str, FastMCPClient] = {}
        self.server_configs: Dict[str, MCPServerConfig] = {}
        
        if FastMCPClient is None:
            logger.error("FastMCP not available. Install with: pip install fastmcp")
            return
        
        # Load server configurations
        self._load_server_configs()
        
        logger.info(f"MCP Client Service initialized with {len(self.server_configs)} server configs")

    def _load_server_configs(self):
        """Load MCP server configurations."""
        try:
            # Load from secrets.yaml
            mcp_config = self.config_service.get_secret("mcp", {})
            
            for server_name, server_data in mcp_config.items():
                try:
                    # Resolve environment variable references
                    resolved_env = {}
                    for key, value in server_data.get("env", {}).items():
                        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                            # Extract the path like "tavily.api_key"
                            path = value[2:-1]
                            resolved_value = self._resolve_config_path(path)
                            resolved_env[key] = resolved_value
                        else:
                            resolved_env[key] = value
                    
                    server_config = MCPServerConfig(
                        name=server_name,
                        command=server_data.get("command", ""),
                        args=server_data.get("args", []),
                        env=resolved_env,
                        timeout=server_data.get("timeout", 30),
                        max_retries=server_data.get("max_retries", 3)
                    )
                    
                    self.server_configs[server_name] = server_config
                    logger.info(f"Loaded MCP server config: {server_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to load server config {server_name}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to load MCP server configurations: {e}")

    def _resolve_config_path(self, path: str) -> str:
        """Resolve configuration path like 'tavily.api_key'."""
        try:
            parts = path.split(".")
            if len(parts) == 2:
                section, key = parts
                section_data = self.config_service.get_secret(section, {})
                return section_data.get(key, "")
            return ""
        except Exception as e:
            logger.error(f"Failed to resolve config path {path}: {e}")
            return ""

    async def connect_to_server(self, server_name: str) -> bool:
        """Connect to an MCP server."""
        if FastMCPClient is None:
            logger.error("FastMCP not available")
            return False
        
        if server_name not in self.server_configs:
            logger.error(f"Server config not found: {server_name}")
            return False
        
        if server_name in self.clients:
            logger.info(f"Already connected to server: {server_name}")
            return True
        
        config = self.server_configs[server_name]
        
        try:
            logger.info(f"Connecting to MCP server: {server_name}")
            
            client = FastMCPClient(
                command=config.command,
                args=config.args,
                env=config.env,
                timeout=config.timeout
            )
            
            await client.connect()
            self.clients[server_name] = client
            
            logger.info(f"Successfully connected to MCP server: {server_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {server_name}: {e}")
            return False

    async def disconnect_from_server(self, server_name: str):
        """Disconnect from an MCP server."""
        if server_name in self.clients:
            try:
                await self.clients[server_name].disconnect()
                del self.clients[server_name]
                logger.info(f"Disconnected from MCP server: {server_name}")
            except Exception as e:
                logger.error(f"Error disconnecting from {server_name}: {e}")

    async def call_tool(self, server_name: str, tool_call: MCPToolCall) -> MCPToolResult:
        """Call a tool on an MCP server."""
        start_time = datetime.now()
        
        try:
            # Ensure connection
            if server_name not in self.clients:
                connected = await self.connect_to_server(server_name)
                if not connected:
                    return MCPToolResult(
                        success=False,
                        error=f"Failed to connect to server: {server_name}",
                        server_name=server_name
                    )
            
            client = self.clients[server_name]
            
            # Make the tool call
            result = await client.call_tool(
                name=tool_call.tool_name,
                arguments=tool_call.arguments
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return MCPToolResult(
                success=True,
                result=result,
                execution_time=execution_time,
                server_name=server_name
            )
            
        except MCPConnectionError as e:
            logger.error(f"MCP connection error for {server_name}: {e}")
            # Try to reconnect
            await self.disconnect_from_server(server_name)
            return MCPToolResult(
                success=False,
                error=f"Connection error: {str(e)}",
                server_name=server_name
            )
            
        except MCPError as e:
            logger.error(f"MCP error for {server_name}: {e}")
            return MCPToolResult(
                success=False,
                error=f"MCP error: {str(e)}",
                server_name=server_name
            )
            
        except Exception as e:
            logger.error(f"Unexpected error calling tool on {server_name}: {e}")
            return MCPToolResult(
                success=False,
                error=f"Unexpected error: {str(e)}",
                server_name=server_name
            )

    async def list_tools(self, server_name: str) -> List[Dict[str, Any]]:
        """List available tools on an MCP server."""
        try:
            if server_name not in self.clients:
                connected = await self.connect_to_server(server_name)
                if not connected:
                    return []
            
            client = self.clients[server_name]
            tools = await client.list_tools()
            
            return tools
            
        except Exception as e:
            logger.error(f"Failed to list tools for {server_name}: {e}")
            return []

    async def cleanup(self):
        """Cleanup all MCP connections."""
        for server_name in list(self.clients.keys()):
            await self.disconnect_from_server(server_name)
        
        logger.info("MCP Client Service cleanup completed")
```

This establishes the foundation for MCP integration. In the next steps, we'll create the Tavily-specific research tool and integrate it with the existing system.

## 🔄 Next Steps Preview

In the remaining steps, we will:
- Create Tavily MCP research tool with compatible output format
- Integrate with the existing Researcher Agent
- Test MCP functionality and error handling
- Create comprehensive research workflows
- Optimize performance and reliability

Ready to build the MCP-powered research system!

## 🔍 Step 4: Create Tavily MCP Research Tool

### 4.1 Create Tavily MCP Research Tool

Create the file `libs/common/utils/tavily_mcp_tool.py`:

```python
"""Tavily MCP research tool for advanced web search and content analysis."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from agents.tools.base_tool import BaseTool
from agents.researcher_agent.research_models import ResearchSource, ResearchSourceType
from core.config_service import ConfigService
from core.llm_service import ModelSpeed
from core.logging_service import get_logger
from pydantic import BaseModel, Field

from .mcp_client_service import MCPClientService, MCPToolCall

logger = get_logger(__name__)


class TavilySearchParams(BaseModel):
    """Parameters for Tavily MCP search."""
    query: str = Field(description="Search query")
    search_type: str = Field(default="web", description="Search type: web, news, academic")
    max_results: int = Field(default=10, description="Maximum number of results")
    include_content: bool = Field(default=True, description="Include full content extraction")
    include_summary: bool = Field(default=True, description="Include AI-generated summaries")
    search_depth: str = Field(default="advanced", description="Search depth: basic, advanced, comprehensive")
    include_domains: List[str] = Field(default_factory=list, description="Domains to include")
    exclude_domains: List[str] = Field(default_factory=list, description="Domains to exclude")


class TavilySearchResult(BaseModel):
    """Result from Tavily MCP search."""
    success: bool = Field(description="Whether the search was successful")
    query: str = Field(description="Original search query")
    sources: List[ResearchSource] = Field(default_factory=list, description="Research sources found")
    total_results: int = Field(default=0, description="Total number of results")
    search_time: float = Field(default=0.0, description="Search execution time")
    summary: str = Field(description="Summary of search results")
    related_queries: List[str] = Field(default_factory=list, description="Related search queries")
    error: Optional[str] = Field(None, description="Error message if failed")


class TavilyMCPTool(BaseTool):
    """Tool for conducting research using Tavily MCP server."""

    def __init__(self, config_service: ConfigService | None = None):
        """Initialize Tavily MCP tool."""
        name = "tavily_mcp_search"
        description = f"""
Conduct advanced web research using Tavily MCP server with AI-powered search and content analysis.

PARAMETER SCHEMA:
{TavilySearchParams.model_json_schema()}

CORRECT USAGE EXAMPLES:
# General web search
{{"query": "artificial intelligence breakthrough 2024", "search_type": "web", "max_results": 10}}

# News search with content extraction
{{"query": "quantum computing news", "search_type": "news", "include_content": true, "max_results": 5}}

# Academic research with comprehensive depth
{{"query": "climate change ocean currents", "search_type": "academic", "search_depth": "comprehensive", "max_results": 8}}

# Targeted domain search
{{"query": "tech industry analysis", "include_domains": ["techcrunch.com", "wired.com"], "max_results": 6}}

SEARCH TYPES:
- web: General web search across all domains
- news: Recent news articles and current events
- academic: Scholarly articles and research papers

SEARCH DEPTHS:
- basic: Quick search with essential results
- advanced: Comprehensive search with AI analysis
- comprehensive: Deep search with extensive content extraction

FEATURES:
- AI-powered content summarization
- Relevance scoring and ranking
- Source credibility assessment
- Related query suggestions
- Full content extraction from URLs

RETURNS:
- sources: List of ResearchSource objects compatible with existing system
- total_results: Number of results found
- summary: AI-generated summary of findings
- related_queries: Suggested follow-up searches

This tool provides advanced research capabilities through Tavily's AI-powered search engine.
"""
        super().__init__(name=name, description=description)
        self.params_model = TavilySearchParams
        self.config_service = config_service or ConfigService()

        # Initialize MCP client service
        self.mcp_service = MCPClientService(self.config_service)
        self.server_name = "tavily_server"

        logger.info("Tavily MCP tool initialized")

    async def execute(self, params: BaseModel, model_speed: ModelSpeed = ModelSpeed.FAST) -> TavilySearchResult:
        """Execute Tavily MCP search."""
        if not isinstance(params, TavilySearchParams):
            return TavilySearchResult(
                success=False,
                query="",
                summary="Invalid parameters",
                error=f"Expected TavilySearchParams, got {type(params)}"
            )

        start_time = datetime.now()

        try:
            logger.info(f"Executing Tavily search: {params.query}")

            # Prepare MCP tool call
            tool_call = MCPToolCall(
                tool_name="search",
                arguments={
                    "query": params.query,
                    "search_type": params.search_type,
                    "max_results": params.max_results,
                    "include_content": params.include_content,
                    "include_summary": params.include_summary,
                    "search_depth": params.search_depth,
                    "include_domains": params.include_domains,
                    "exclude_domains": params.exclude_domains
                }
            )

            # Execute MCP call
            mcp_result = await self.mcp_service.call_tool(self.server_name, tool_call)

            if not mcp_result.success:
                return TavilySearchResult(
                    success=False,
                    query=params.query,
                    summary="MCP call failed",
                    error=mcp_result.error
                )

            # Process MCP result
            search_result = await self._process_mcp_result(params, mcp_result.result)

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            search_result.search_time = execution_time

            logger.info(f"Tavily search completed: {len(search_result.sources)} sources in {execution_time:.1f}s")

            return search_result

        except Exception as e:
            logger.error(f"Tavily MCP search failed: {e}")
            return TavilySearchResult(
                success=False,
                query=params.query,
                summary="Search execution failed",
                error=str(e)
            )

    async def _process_mcp_result(self, params: TavilySearchParams, mcp_data: Any) -> TavilySearchResult:
        """Process MCP result data into TavilySearchResult format."""
        try:
            # Handle different possible MCP result formats
            if isinstance(mcp_data, dict):
                results_data = mcp_data
            elif isinstance(mcp_data, list) and len(mcp_data) > 0:
                results_data = mcp_data[0] if isinstance(mcp_data[0], dict) else {"results": mcp_data}
            else:
                results_data = {"results": []}

            # Extract search results
            raw_results = results_data.get("results", [])
            if not isinstance(raw_results, list):
                raw_results = []

            # Convert to ResearchSource objects
            sources = []
            for i, result in enumerate(raw_results[:params.max_results]):
                try:
                    source = await self._convert_to_research_source(result, i)
                    if source:
                        sources.append(source)
                except Exception as e:
                    logger.error(f"Failed to convert result {i}: {e}")
                    continue

            # Generate summary
            summary = await self._generate_search_summary(params.query, sources, results_data)

            # Extract related queries
            related_queries = results_data.get("related_queries", [])
            if not isinstance(related_queries, list):
                related_queries = []

            return TavilySearchResult(
                success=True,
                query=params.query,
                sources=sources,
                total_results=len(raw_results),
                summary=summary,
                related_queries=related_queries[:5]  # Limit to 5 related queries
            )

        except Exception as e:
            logger.error(f"Failed to process MCP result: {e}")
            return TavilySearchResult(
                success=False,
                query=params.query,
                summary="Failed to process search results",
                error=str(e)
            )

    async def _convert_to_research_source(self, result: Dict[str, Any], index: int) -> Optional[ResearchSource]:
        """Convert Tavily result to ResearchSource format."""
        try:
            # Extract basic information
            url = result.get("url", "")
            title = result.get("title", f"Search Result {index + 1}")
            content = result.get("content", "")
            summary = result.get("summary", content[:300] + "..." if len(content) > 300 else content)

            # Determine source type based on URL or content
            source_type = self._determine_source_type(url, result)

            # Extract metadata
            metadata = {
                "score": result.get("score", 0.0),
                "published_date": result.get("published_date", ""),
                "author": result.get("author", ""),
                "domain": result.get("domain", ""),
                "search_rank": index + 1
            }

            # Add Tavily-specific metadata
            if "relevance_score" in result:
                metadata["relevance_score"] = result["relevance_score"]
            if "credibility_score" in result:
                metadata["credibility_score"] = result["credibility_score"]

            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(result)

            return ResearchSource(
                url=url,
                title=title,
                summary=summary,
                source_type=source_type,
                content=content if content else summary,
                metadata=metadata,
                relevance_score=relevance_score,
                accessed_at=datetime.now()
            )

        except Exception as e:
            logger.error(f"Failed to convert result to ResearchSource: {e}")
            return None

    def _determine_source_type(self, url: str, result: Dict[str, Any]) -> ResearchSourceType:
        """Determine the source type based on URL and result data."""
        url_lower = url.lower()

        # Check for news sources
        news_indicators = ["news", "reuters", "cnn", "bbc", "ap", "bloomberg", "wsj"]
        if any(indicator in url_lower for indicator in news_indicators):
            return ResearchSourceType.NEWS_ARTICLE

        # Check for academic sources
        academic_indicators = ["arxiv", "scholar", "pubmed", "ieee", "acm", ".edu"]
        if any(indicator in url_lower for indicator in academic_indicators):
            return ResearchSourceType.WEB_SEARCH  # We don't have academic type, use web_search

        # Check for social media (though Tavily typically filters these)
        social_indicators = ["twitter", "facebook", "linkedin", "reddit"]
        if any(indicator in url_lower for indicator in social_indicators):
            return ResearchSourceType.SOCIAL_MEDIA

        # Default to web search
        return ResearchSourceType.WEB_SEARCH

    def _calculate_relevance_score(self, result: Dict[str, Any]) -> float:
        """Calculate relevance score from Tavily result data."""
        try:
            # Use Tavily's score if available
            if "score" in result:
                return min(float(result["score"]), 1.0)

            # Use relevance score if available
            if "relevance_score" in result:
                return min(float(result["relevance_score"]), 1.0)

            # Calculate based on available metadata
            score = 0.5  # Base score

            # Boost for credibility
            if "credibility_score" in result:
                score += float(result["credibility_score"]) * 0.2

            # Boost for recent content
            if "published_date" in result and result["published_date"]:
                # Simple recency boost (would need proper date parsing in production)
                score += 0.1

            # Boost for content length (indicates comprehensive coverage)
            content_length = len(result.get("content", ""))
            if content_length > 1000:
                score += 0.1
            elif content_length > 500:
                score += 0.05

            return min(score, 1.0)

        except Exception as e:
            logger.error(f"Failed to calculate relevance score: {e}")
            return 0.5

    async def _generate_search_summary(self, query: str, sources: List[ResearchSource], results_data: Dict[str, Any]) -> str:
        """Generate summary of search results."""
        try:
            if not sources:
                return f"No results found for query: {query}"

            # Use Tavily's summary if available
            if "summary" in results_data and results_data["summary"]:
                return results_data["summary"]

            # Generate basic summary
            total_sources = len(sources)
            avg_relevance = sum(source.relevance_score for source in sources) / total_sources if total_sources > 0 else 0

            # Count source types
            source_types = {}
            for source in sources:
                source_type = source.source_type.value
                source_types[source_type] = source_types.get(source_type, 0) + 1

            summary_parts = [
                f"Found {total_sources} relevant sources for '{query}'",
                f"Average relevance score: {avg_relevance:.2f}",
                f"Source types: {', '.join(f'{k}: {v}' for k, v in source_types.items())}"
            ]

            # Add top source information
            if sources:
                top_source = max(sources, key=lambda s: s.relevance_score)
                summary_parts.append(f"Top result: {top_source.title} (score: {top_source.relevance_score:.2f})")

            return ". ".join(summary_parts) + "."

        except Exception as e:
            logger.error(f"Failed to generate search summary: {e}")
            return f"Search completed for query: {query}"

    async def search_news(self, query: str, max_results: int = 5) -> TavilySearchResult:
        """Convenience method for news search."""
        params = TavilySearchParams(
            query=query,
            search_type="news",
            max_results=max_results,
            search_depth="advanced"
        )
        return await self.execute(params)

    async def search_academic(self, query: str, max_results: int = 8) -> TavilySearchResult:
        """Convenience method for academic search."""
        params = TavilySearchParams(
            query=query,
            search_type="academic",
            max_results=max_results,
            search_depth="comprehensive"
        )
        return await self.execute(params)

    async def search_web(self, query: str, max_results: int = 10) -> TavilySearchResult:
        """Convenience method for general web search."""
        params = TavilySearchParams(
            query=query,
            search_type="web",
            max_results=max_results,
            search_depth="advanced"
        )
        return await self.execute(params)
```

### 4.2 Create MCP Tool Registry

Create the file `libs/common/utils/mcp_tool_registry.py`:

```python
"""Registry for MCP tools and their management."""

from typing import Dict, List, Optional

from core.config_service import ConfigService
from core.logging_service import get_logger

from .mcp_client_service import MCPClientService
from .tavily_mcp_tool import TavilyMCPTool

logger = get_logger(__name__)


class MCPToolRegistry:
    """Registry for managing MCP tools and their lifecycle."""

    def __init__(self, config_service: ConfigService):
        """Initialize MCP tool registry."""
        self.config_service = config_service
        self.mcp_service = MCPClientService(config_service)
        self.tools: Dict[str, any] = {}

        # Initialize available MCP tools
        self._initialize_tools()

        logger.info(f"MCP Tool Registry initialized with {len(self.tools)} tools")

    def _initialize_tools(self):
        """Initialize available MCP tools."""
        try:
            # Initialize Tavily MCP tool
            if self._is_tavily_enabled():
                self.tools["tavily_search"] = TavilyMCPTool(self.config_service)
                logger.info("Tavily MCP tool registered")

            # Add more MCP tools here as needed
            # self.tools["other_mcp_tool"] = OtherMCPTool(self.config_service)

        except Exception as e:
            logger.error(f"Failed to initialize MCP tools: {e}")

    def _is_tavily_enabled(self) -> bool:
        """Check if Tavily MCP is enabled and configured."""
        try:
            # Check environment variable
            tavily_enabled = self.config_service.get_env_var("TAVILY_ENABLED", "false").lower() == "true"

            # Check if API key is configured
            tavily_config = self.config_service.get_secret("tavily", {})
            has_api_key = bool(tavily_config.get("api_key"))

            return tavily_enabled and has_api_key

        except Exception as e:
            logger.error(f"Failed to check Tavily configuration: {e}")
            return False

    def get_tool(self, tool_name: str) -> Optional[any]:
        """Get an MCP tool by name."""
        return self.tools.get(tool_name)

    def list_tools(self) -> List[str]:
        """List available MCP tool names."""
        return list(self.tools.keys())

    def is_tool_available(self, tool_name: str) -> bool:
        """Check if an MCP tool is available."""
        return tool_name in self.tools

    async def test_tool_connection(self, tool_name: str) -> bool:
        """Test connection to an MCP tool."""
        try:
            tool = self.get_tool(tool_name)
            if not tool:
                return False

            # For Tavily tool, test with a simple search
            if tool_name == "tavily_search":
                from .tavily_mcp_tool import TavilySearchParams

                test_params = TavilySearchParams(
                    query="test connection",
                    max_results=1,
                    search_depth="basic"
                )

                result = await tool.execute(test_params)
                return result.success

            return True

        except Exception as e:
            logger.error(f"Failed to test tool connection {tool_name}: {e}")
            return False

    async def cleanup(self):
        """Cleanup MCP tool registry and connections."""
        try:
            await self.mcp_service.cleanup()
            logger.info("MCP Tool Registry cleanup completed")
        except Exception as e:
            logger.error(f"MCP Tool Registry cleanup failed: {e}")
```

## 🔗 Step 5: Integrate MCP Tools with Researcher Agent

### 5.1 Update Multi-Source Research Tool

Update the existing `MultiSourceResearchTool` to include MCP capabilities. Modify `libs/common/agents/researcher_agent/research_tools.py`:

```python
# Add these imports at the top
from utils.mcp_tool_registry import MCPToolRegistry
from utils.tavily_mcp_tool import TavilySearchParams

# Add to the MultiSourceResearchTool class __init__ method:
def __init__(self, config_service: ConfigService | None = None):
    # ... existing initialization code ...

    # Initialize MCP tool registry
    try:
        self.mcp_registry = MCPToolRegistry(self.config_service)
        logger.info("MCP tools initialized for research")
    except Exception as e:
        logger.error(f"Failed to initialize MCP tools: {e}")
        self.mcp_registry = None

# Add new method to the MultiSourceResearchTool class:
async def _research_topic_mcp(self, topic: str, field: ReporterField) -> list[ResearchSource]:
    """Research a topic using MCP tools (Tavily)."""
    if not self.mcp_registry or not self.mcp_registry.is_tool_available("tavily_search"):
        return []

    try:
        tavily_tool = self.mcp_registry.get_tool("tavily_search")

        # Create field-specific search query
        field_context = {
            ReporterField.TECHNOLOGY: "technology tech innovation",
            ReporterField.SCIENCE: "science research scientific",
            ReporterField.ECONOMICS: "economics finance business market"
        }

        context = field_context.get(field, "")
        enhanced_query = f"{topic} {context}".strip()

        # Perform MCP search
        search_params = TavilySearchParams(
            query=enhanced_query,
            search_type="web",
            max_results=5,
            include_content=True,
            search_depth="advanced"
        )

        result = await tavily_tool.execute(search_params)

        if result.success:
            logger.info(f"MCP search found {len(result.sources)} sources for '{topic}'")
            return result.sources
        else:
            logger.error(f"MCP search failed for '{topic}': {result.error}")
            return []

    except Exception as e:
        logger.error(f"MCP research for '{topic}' failed: {e}")
        return []

# Update the _research_topic_in_depth method to include MCP research:
async def _research_topic_in_depth(self, topic_title: str, field: ReporterField, params: ResearchRequest) -> TopicResearch | None:
    """Research a specific topic in depth across multiple sources including MCP."""
    try:
        sources = []

        # YouTube research
        if params.include_youtube:
            youtube_sources = await self._research_topic_youtube(topic_title, field)
            sources.extend(youtube_sources)

        # Web search research
        if params.include_web_search:
            search_sources = await self._research_topic_search(topic_title, field)
            sources.extend(search_sources)

        # MCP research (Tavily)
        mcp_sources = await self._research_topic_mcp(topic_title, field)
        sources.extend(mcp_sources)

        # Web scraping research
        if params.include_scraping and len(sources) > 0:
            scraping_sources = await self._research_topic_scraping(sources[:3])
            sources.extend(scraping_sources)

        # Limit sources based on research depth
        sources = sources[:params.research_depth]

        if not sources:
            return None

        # Generate topic research summary
        topic_research = await self._compile_topic_research(
            topic_title, field, sources, params
        )

        return topic_research

    except Exception as e:
        logger.error(f"In-depth research for '{topic_title}' failed: {e}")
        return None
```

### 5.2 Create MCP Research Integration Test

Create the file `test_mcp_integration.py`:

```python
"""Test MCP integration with the research system."""

import asyncio
from datetime import datetime

from agents.agent_factory import AgentFactory
from core.config_service import ConfigService
from utils.mcp_tool_registry import MCPToolRegistry
from utils.tavily_mcp_tool import TavilyMCPTool, TavilySearchParams


async def test_mcp_integration():
    """Test MCP integration functionality."""
    print("🔌 MCP Integration Test")
    print("=" * 40)

    config_service = ConfigService()

    # Test 1: MCP Tool Registry
    print("\n1️⃣ Testing MCP Tool Registry...")

    mcp_registry = MCPToolRegistry(config_service)
    available_tools = mcp_registry.list_tools()

    print(f"   📋 Available MCP tools: {available_tools}")

    if "tavily_search" in available_tools:
        print("   ✅ Tavily MCP tool is available")

        # Test connection
        connection_ok = await mcp_registry.test_tool_connection("tavily_search")
        if connection_ok:
            print("   ✅ Tavily MCP connection test passed")
        else:
            print("   ❌ Tavily MCP connection test failed")
            return
    else:
        print("   ❌ Tavily MCP tool not available")
        return

    # Test 2: Direct Tavily Tool Usage
    print("\n2️⃣ Testing direct Tavily MCP tool...")

    tavily_tool = TavilyMCPTool(config_service)

    # Test web search
    search_params = TavilySearchParams(
        query="artificial intelligence breakthrough 2024",
        search_type="web",
        max_results=5,
        include_content=True,
        search_depth="advanced"
    )

    result = await tavily_tool.execute(search_params)

    if result.success:
        print(f"   ✅ Tavily search successful")
        print(f"   📊 Found {len(result.sources)} sources")
        print(f"   ⏱️  Search time: {result.search_time:.1f}s")
        print(f"   📄 Summary: {result.summary[:100]}...")

        # Show source details
        for i, source in enumerate(result.sources[:2]):
            print(f"   \n   Source {i+1}:")
            print(f"      Title: {source.title}")
            print(f"      URL: {source.url}")
            print(f"      Type: {source.source_type.value}")
            print(f"      Relevance: {source.relevance_score:.2f}")
            print(f"      Summary: {source.summary[:100]}...")
    else:
        print(f"   ❌ Tavily search failed: {result.error}")
        return

    # Test 3: News search
    print("\n3️⃣ Testing Tavily news search...")

    news_result = await tavily_tool.search_news("quantum computing news", max_results=3)

    if news_result.success:
        print(f"   ✅ News search successful")
        print(f"   📰 Found {len(news_result.sources)} news sources")
        print(f"   📄 Summary: {news_result.summary[:100]}...")

        # Show news source
        if news_result.sources:
            news_source = news_result.sources[0]
            print(f"   📰 Top news: {news_source.title}")
            print(f"      Relevance: {news_source.relevance_score:.2f}")
    else:
        print(f"   ❌ News search failed: {news_result.error}")

    # Test 4: Integration with Researcher Agent
    print("\n4️⃣ Testing integration with Researcher Agent...")

    factory = AgentFactory(config_service)
    researcher = factory.create_researcher()

    # Test research with MCP integration
    research_result = await researcher.research_trending_topics(
        fields=["technology"],
        topics_per_field=2
    )

    if research_result["success"]:
        print(f"   ✅ Researcher Agent with MCP integration successful")
        print(f"   📊 Topics researched: {len(research_result['topics_researched'])}")
        print(f"   🔍 Total sources: {research_result['total_sources']}")

        # Check if MCP sources are included
        mcp_sources_found = 0
        for topic_data in research_result["topics_researched"]:
            for source in topic_data["sources"]:
                if "tavily" in source.get("metadata", {}).get("search_rank", ""):
                    mcp_sources_found += 1

        print(f"   🔌 MCP sources found: {mcp_sources_found}")

        # Show sample topic with sources
        if research_result["topics_researched"]:
            sample_topic = research_result["topics_researched"][0]
            print(f"   \n   Sample Topic: {sample_topic['topic_title']}")
            print(f"      Sources: {len(sample_topic['sources'])}")

            # Show source breakdown
            source_types = {}
            for source in sample_topic["sources"]:
                source_type = source["source_type"]
                source_types[source_type] = source_types.get(source_type, 0) + 1

            print(f"      Source types: {dict(source_types)}")
    else:
        print(f"   ❌ Researcher Agent integration failed: {research_result['error']}")

    # Cleanup
    await mcp_registry.cleanup()

    print("\n🎉 MCP Integration test completed!")


async def test_mcp_error_handling():
    """Test MCP error handling and fallback mechanisms."""
    print("\n🛡️ MCP Error Handling Test")
    print("=" * 35)

    config_service = ConfigService()
    tavily_tool = TavilyMCPTool(config_service)

    # Test 1: Invalid query
    print("\n1️⃣ Testing invalid query handling...")

    invalid_params = TavilySearchParams(
        query="",  # Empty query
        max_results=5
    )

    result = await tavily_tool.execute(invalid_params)
    print(f"   Empty query result: {'✅ Handled gracefully' if not result.success else '❌ Should have failed'}")

    # Test 2: Large result set
    print("\n2️⃣ Testing large result set handling...")

    large_params = TavilySearchParams(
        query="technology news",
        max_results=50,  # Large number
        search_depth="comprehensive"
    )

    result = await tavily_tool.execute(large_params)
    if result.success:
        print(f"   ✅ Large result set handled: {len(result.sources)} sources")
    else:
        print(f"   ⚠️  Large result set failed: {result.error}")

    # Test 3: Network timeout simulation
    print("\n3️⃣ Testing timeout handling...")

    timeout_params = TavilySearchParams(
        query="very specific complex query that might timeout",
        search_depth="comprehensive",
        max_results=20
    )

    start_time = datetime.now()
    result = await tavily_tool.execute(timeout_params)
    duration = (datetime.now() - start_time).total_seconds()

    print(f"   Query completed in {duration:.1f}s")
    if result.success:
        print(f"   ✅ Query successful: {len(result.sources)} sources")
    else:
        print(f"   ⚠️  Query failed: {result.error}")

    print("\n🛡️ Error handling test completed!")


if __name__ == "__main__":
    asyncio.run(test_mcp_integration())
    asyncio.run(test_mcp_error_handling())
```

### 5.3 Create MCP Configuration Validator

Create the file `validate_mcp_setup.py`:

```python
"""Validate MCP setup and configuration."""

import asyncio
import os
import subprocess
from pathlib import Path

from core.config_service import ConfigService
from utils.mcp_client_service import MCPClientService


async def validate_mcp_setup():
    """Validate complete MCP setup."""
    print("🔧 MCP Setup Validation")
    print("=" * 30)

    validation_results = {
        "dependencies": False,
        "tavily_server": False,
        "configuration": False,
        "api_key": False,
        "connection": False
    }

    # Test 1: Check Python dependencies
    print("\n1️⃣ Checking Python dependencies...")

    try:
        import fastmcp
        import httpx
        import websockets
        print("   ✅ FastMCP and dependencies installed")
        validation_results["dependencies"] = True
    except ImportError as e:
        print(f"   ❌ Missing dependencies: {e}")
        print("   💡 Run: uv sync")

    # Test 2: Check Tavily MCP server installation
    print("\n2️⃣ Checking Tavily MCP server...")

    try:
        result = subprocess.run(
            ["npx", "@tavily/mcp-server", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            print(f"   ✅ Tavily MCP server installed: {result.stdout.strip()}")
            validation_results["tavily_server"] = True
        else:
            print(f"   ❌ Tavily MCP server not working: {result.stderr}")
            print("   💡 Run: npm install -g @tavily/mcp-server")
    except subprocess.TimeoutExpired:
        print("   ❌ Tavily MCP server check timed out")
    except FileNotFoundError:
        print("   ❌ npm/npx not found")
        print("   💡 Install Node.js and npm")
    except Exception as e:
        print(f"   ❌ Error checking Tavily server: {e}")

    # Test 3: Check configuration files
    print("\n3️⃣ Checking configuration...")

    config_service = ConfigService()

    # Check secrets.yaml
    secrets_path = Path("libs/common/secrets.yaml")
    if secrets_path.exists():
        print("   ✅ secrets.yaml exists")

        # Check MCP configuration
        mcp_config = config_service.get_secret("mcp", {})
        if mcp_config:
            print("   ✅ MCP configuration found in secrets.yaml")
            validation_results["configuration"] = True
        else:
            print("   ❌ MCP configuration missing from secrets.yaml")
    else:
        print("   ❌ secrets.yaml not found")

    # Check environment variables
    env_file = Path(".env.development")
    if env_file.exists():
        print("   ✅ .env.development exists")
    else:
        print("   ❌ .env.development not found")

    # Test 4: Check Tavily API key
    print("\n4️⃣ Checking Tavily API key...")

    tavily_config = config_service.get_secret("tavily", {})
    api_key = tavily_config.get("api_key", "")

    if api_key and api_key != "your_tavily_api_key_here":
        print("   ✅ Tavily API key configured")
        validation_results["api_key"] = True

        # Validate API key format
        if api_key.startswith("tvly-"):
            print("   ✅ API key format looks correct")
        else:
            print("   ⚠️  API key format might be incorrect (should start with 'tvly-')")
    else:
        print("   ❌ Tavily API key not configured")
        print("   💡 Get API key from https://tavily.com and add to secrets.yaml")

    # Test 5: Test MCP connection
    print("\n5️⃣ Testing MCP connection...")

    if validation_results["dependencies"] and validation_results["tavily_server"] and validation_results["api_key"]:
        try:
            mcp_service = MCPClientService(config_service)
            connected = await mcp_service.connect_to_server("tavily_server")

            if connected:
                print("   ✅ MCP connection successful")
                validation_results["connection"] = True

                # List available tools
                tools = await mcp_service.list_tools("tavily_server")
                print(f"   📋 Available tools: {len(tools)}")

                await mcp_service.cleanup()
            else:
                print("   ❌ MCP connection failed")
        except Exception as e:
            print(f"   ❌ MCP connection error: {e}")
    else:
        print("   ⏭️  Skipping connection test (prerequisites not met)")

    # Summary
    print("\n📊 Validation Summary:")
    print("=" * 25)

    passed = sum(validation_results.values())
    total = len(validation_results)

    for check, result in validation_results.items():
        status = "✅" if result else "❌"
        print(f"   {status} {check.replace('_', ' ').title()}")

    print(f"\n🎯 Overall: {passed}/{total} checks passed")

    if passed == total:
        print("🎉 MCP setup is complete and ready!")
        return True
    else:
        print("⚠️  MCP setup needs attention. Check failed items above.")
        return False


if __name__ == "__main__":
    asyncio.run(validate_mcp_setup())
```

## 🧪 Step 6: Testing and Validation

### 6.1 Run Complete Test Suite

Execute the complete MCP testing suite:

```bash
# Step 1: Validate MCP setup
python validate_mcp_setup.py

# Step 2: Test MCP integration
python test_mcp_integration.py

# Step 3: Test with three-agent workflow
python test_three_agent_workflow.py
```

### 6.2 Expected Test Results

**MCP Setup Validation Results:**
```
🔧 MCP Setup Validation
==============================

1️⃣ Checking Python dependencies...
   ✅ FastMCP and dependencies installed

2️⃣ Checking Tavily MCP server...
   ✅ Tavily MCP server installed: v1.0.0

3️⃣ Checking configuration...
   ✅ secrets.yaml exists
   ✅ MCP configuration found in secrets.yaml
   ✅ .env.development exists

4️⃣ Checking Tavily API key...
   ✅ Tavily API key configured
   ✅ API key format looks correct

5️⃣ Testing MCP connection...
   ✅ MCP connection successful
   📋 Available tools: 3

📊 Validation Summary:
=========================
   ✅ Dependencies
   ✅ Tavily Server
   ✅ Configuration
   ✅ Api Key
   ✅ Connection

🎯 Overall: 5/5 checks passed
🎉 MCP setup is complete and ready!
```

**MCP Integration Test Results:**
```
🔌 MCP Integration Test
========================================

1️⃣ Testing MCP Tool Registry...
   📋 Available MCP tools: ['tavily_search']
   ✅ Tavily MCP tool is available
   ✅ Tavily MCP connection test passed

2️⃣ Testing direct Tavily MCP tool...
   ✅ Tavily search successful
   📊 Found 5 sources
   ⏱️  Search time: 2.3s
   📄 Summary: Found 5 relevant sources for 'artificial intelligence breakthrough 2024'. Average relevance...

   Source 1:
      Title: Major AI Breakthrough Announced by OpenAI
      URL: https://example.com/ai-breakthrough
      Type: news_article
      Relevance: 0.92
      Summary: OpenAI announces significant advancement in AI reasoning capabilities...

3️⃣ Testing Tavily news search...
   ✅ News search successful
   📰 Found 3 news sources
   📄 Summary: Recent quantum computing developments show promising advances...
   📰 Top news: Quantum Computing Milestone Achieved at IBM
      Relevance: 0.88

4️⃣ Testing integration with Researcher Agent...
   ✅ Researcher Agent with MCP integration successful
   📊 Topics researched: 2
   🔍 Total sources: 8
   🔌 MCP sources found: 3

   Sample Topic: AI-Powered Medical Diagnostics
      Sources: 4
      Source types: {'web_search': 2, 'news_article': 1, 'youtube_video': 1}

🎉 MCP Integration test completed!

🛡️ MCP Error Handling Test
===================================

1️⃣ Testing invalid query handling...
   Empty query result: ✅ Handled gracefully

2️⃣ Testing large result set handling...
   ✅ Large result set handled: 20 sources

3️⃣ Testing timeout handling...
   Query completed in 4.2s
   ✅ Query successful: 15 sources

🛡️ Error handling test completed!
```

### 6.3 Performance Benchmarking

Create the file `benchmark_mcp_performance.py`:

```python
"""Benchmark MCP performance and compare with other research methods."""

import asyncio
import time
from statistics import mean, median

from agents.agent_factory import AgentFactory
from core.config_service import ConfigService
from utils.tavily_mcp_tool import TavilyMCPTool, TavilySearchParams


async def benchmark_mcp_performance():
    """Benchmark MCP performance against other research methods."""
    print("⚡ MCP Performance Benchmark")
    print("=" * 40)

    config_service = ConfigService()

    # Test queries
    test_queries = [
        "artificial intelligence breakthrough 2024",
        "quantum computing advances",
        "climate change research",
        "economic market analysis",
        "space exploration news"
    ]

    # Benchmark 1: Tavily MCP Tool Performance
    print("\n1️⃣ Benchmarking Tavily MCP Tool...")

    tavily_tool = TavilyMCPTool(config_service)
    tavily_times = []
    tavily_results = []

    for query in test_queries:
        start_time = time.time()

        params = TavilySearchParams(
            query=query,
            search_type="web",
            max_results=5,
            search_depth="advanced"
        )

        result = await tavily_tool.execute(params)

        duration = time.time() - start_time
        tavily_times.append(duration)

        if result.success:
            tavily_results.append(len(result.sources))
            print(f"   Query: '{query[:30]}...' - {duration:.1f}s, {len(result.sources)} sources")
        else:
            tavily_results.append(0)
            print(f"   Query: '{query[:30]}...' - {duration:.1f}s, FAILED")

    # Benchmark 2: Traditional Research Tool Performance
    print("\n2️⃣ Benchmarking Traditional Research Tools...")

    factory = AgentFactory(config_service)
    researcher = factory.create_researcher()

    traditional_times = []
    traditional_results = []

    for query in test_queries[:3]:  # Limit to 3 for time
        start_time = time.time()

        # Use the researcher's traditional research method
        result = await researcher.research_trending_topics(
            fields=["technology"],
            topics_per_field=1
        )

        duration = time.time() - start_time
        traditional_times.append(duration)

        if result["success"]:
            total_sources = result["total_sources"]
            traditional_results.append(total_sources)
            print(f"   Query: '{query[:30]}...' - {duration:.1f}s, {total_sources} sources")
        else:
            traditional_results.append(0)
            print(f"   Query: '{query[:30]}...' - {duration:.1f}s, FAILED")

    # Performance Analysis
    print("\n📊 Performance Analysis:")
    print("=" * 30)

    print(f"🔌 Tavily MCP Performance:")
    print(f"   Average time: {mean(tavily_times):.1f}s")
    print(f"   Median time: {median(tavily_times):.1f}s")
    print(f"   Average sources: {mean(tavily_results):.1f}")
    print(f"   Success rate: {sum(1 for r in tavily_results if r > 0) / len(tavily_results) * 100:.0f}%")

    if traditional_times:
        print(f"\n🔍 Traditional Research Performance:")
        print(f"   Average time: {mean(traditional_times):.1f}s")
        print(f"   Median time: {median(traditional_times):.1f}s")
        print(f"   Average sources: {mean(traditional_results):.1f}")
        print(f"   Success rate: {sum(1 for r in traditional_results if r > 0) / len(traditional_results) * 100:.0f}%")

        # Comparison
        speed_improvement = mean(traditional_times) / mean(tavily_times) if mean(tavily_times) > 0 else 0
        print(f"\n⚡ Speed Comparison:")
        print(f"   MCP is {speed_improvement:.1f}x {'faster' if speed_improvement > 1 else 'slower'} than traditional research")

    print("\n🎯 Benchmark completed!")


async def benchmark_concurrent_requests():
    """Benchmark concurrent MCP requests."""
    print("\n🔄 Concurrent Request Benchmark")
    print("=" * 40)

    config_service = ConfigService()
    tavily_tool = TavilyMCPTool(config_service)

    # Test concurrent requests
    concurrent_queries = [
        "AI technology trends",
        "quantum computing news",
        "renewable energy research",
        "biotechnology advances",
        "space technology updates"
    ]

    # Sequential execution
    print("\n1️⃣ Sequential execution...")
    start_time = time.time()

    sequential_results = []
    for query in concurrent_queries:
        params = TavilySearchParams(query=query, max_results=3)
        result = await tavily_tool.execute(params)
        sequential_results.append(result.success)

    sequential_time = time.time() - start_time
    print(f"   Sequential time: {sequential_time:.1f}s")
    print(f"   Success rate: {sum(sequential_results) / len(sequential_results) * 100:.0f}%")

    # Concurrent execution
    print("\n2️⃣ Concurrent execution...")
    start_time = time.time()

    async def search_query(query):
        params = TavilySearchParams(query=query, max_results=3)
        return await tavily_tool.execute(params)

    concurrent_results = await asyncio.gather(*[search_query(q) for q in concurrent_queries])
    concurrent_time = time.time() - start_time

    success_count = sum(1 for r in concurrent_results if r.success)

    print(f"   Concurrent time: {concurrent_time:.1f}s")
    print(f"   Success rate: {success_count / len(concurrent_results) * 100:.0f}%")

    # Performance improvement
    improvement = sequential_time / concurrent_time if concurrent_time > 0 else 0
    print(f"\n⚡ Concurrency Improvement: {improvement:.1f}x faster")


if __name__ == "__main__":
    asyncio.run(benchmark_mcp_performance())
    asyncio.run(benchmark_concurrent_requests())
```

## 🔧 Step 7: Production Optimization

### 7.1 Create MCP Connection Pool

Create the file `libs/common/utils/mcp_connection_pool.py`:

```python
"""Connection pool for MCP clients to improve performance and reliability."""

import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from core.config_service import ConfigService
from core.logging_service import get_logger

from .mcp_client_service import MCPClientService, MCPToolCall, MCPToolResult

logger = get_logger(__name__)


class MCPConnectionPool:
    """Connection pool for managing multiple MCP client connections."""

    def __init__(self, config_service: ConfigService, pool_size: int = 3):
        """Initialize MCP connection pool."""
        self.config_service = config_service
        self.pool_size = pool_size
        self.pools: Dict[str, List[MCPClientService]] = {}
        self.pool_usage: Dict[str, List[datetime]] = {}
        self.pool_locks: Dict[str, asyncio.Lock] = {}

        logger.info(f"MCP Connection Pool initialized with pool size: {pool_size}")

    async def get_client(self, server_name: str) -> Optional[MCPClientService]:
        """Get an available MCP client from the pool."""
        if server_name not in self.pools:
            await self._initialize_pool(server_name)

        if server_name not in self.pool_locks:
            self.pool_locks[server_name] = asyncio.Lock()

        async with self.pool_locks[server_name]:
            pool = self.pools.get(server_name, [])
            usage = self.pool_usage.get(server_name, [])

            # Find least recently used client
            if pool and usage:
                oldest_index = usage.index(min(usage))
                client = pool[oldest_index]
                usage[oldest_index] = datetime.now()
                return client

            return None

    async def _initialize_pool(self, server_name: str):
        """Initialize connection pool for a server."""
        try:
            self.pools[server_name] = []
            self.pool_usage[server_name] = []

            for i in range(self.pool_size):
                client = MCPClientService(self.config_service)
                connected = await client.connect_to_server(server_name)

                if connected:
                    self.pools[server_name].append(client)
                    self.pool_usage[server_name].append(datetime.now())
                    logger.info(f"Created MCP client {i+1}/{self.pool_size} for {server_name}")
                else:
                    logger.error(f"Failed to create MCP client {i+1} for {server_name}")

            logger.info(f"Initialized pool for {server_name}: {len(self.pools[server_name])}/{self.pool_size} clients")

        except Exception as e:
            logger.error(f"Failed to initialize pool for {server_name}: {e}")

    async def execute_with_pool(self, server_name: str, tool_call: MCPToolCall) -> MCPToolResult:
        """Execute tool call using connection pool."""
        client = await self.get_client(server_name)

        if not client:
            return MCPToolResult(
                success=False,
                error=f"No available client for server: {server_name}",
                server_name=server_name
            )

        return await client.call_tool(server_name, tool_call)

    async def cleanup(self):
        """Cleanup all connection pools."""
        for server_name, pool in self.pools.items():
            for client in pool:
                try:
                    await client.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up client for {server_name}: {e}")

        self.pools.clear()
        self.pool_usage.clear()
        self.pool_locks.clear()

        logger.info("MCP Connection Pool cleanup completed")
```

### 7.2 Create MCP Caching Layer

Create the file `libs/common/utils/mcp_cache.py`:

```python
"""Caching layer for MCP results to improve performance."""

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from core.config_service import ConfigService
from core.logging_service import get_logger

logger = get_logger(__name__)


class MCPCache:
    """In-memory cache for MCP results with TTL support."""

    def __init__(self, config_service: ConfigService, default_ttl: int = 3600):
        """Initialize MCP cache."""
        self.config_service = config_service
        self.default_ttl = default_ttl  # 1 hour default
        self.cache: Dict[str, Dict[str, Any]] = {}

        logger.info(f"MCP Cache initialized with default TTL: {default_ttl}s")

    def _generate_cache_key(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Generate cache key from MCP call parameters."""
        # Create deterministic key from parameters
        key_data = {
            "server": server_name,
            "tool": tool_name,
            "args": arguments
        }

        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Optional[Any]:
        """Get cached result if available and not expired."""
        cache_key = self._generate_cache_key(server_name, tool_name, arguments)

        if cache_key not in self.cache:
            return None

        cache_entry = self.cache[cache_key]

        # Check if expired
        if datetime.now() > cache_entry["expires_at"]:
            del self.cache[cache_key]
            logger.debug(f"Cache entry expired and removed: {cache_key[:8]}...")
            return None

        logger.debug(f"Cache hit: {cache_key[:8]}...")
        return cache_entry["result"]

    def set(self, server_name: str, tool_name: str, arguments: Dict[str, Any], result: Any, ttl: Optional[int] = None) -> None:
        """Cache result with TTL."""
        cache_key = self._generate_cache_key(server_name, tool_name, arguments)
        ttl = ttl or self.default_ttl

        self.cache[cache_key] = {
            "result": result,
            "cached_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(seconds=ttl)
        }

        logger.debug(f"Cached result: {cache_key[:8]}... (TTL: {ttl}s)")

    def clear_expired(self) -> int:
        """Clear expired cache entries."""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self.cache.items()
            if now > entry["expires_at"]
        ]

        for key in expired_keys:
            del self.cache[key]

        if expired_keys:
            logger.info(f"Cleared {len(expired_keys)} expired cache entries")

        return len(expired_keys)

    def clear_all(self) -> int:
        """Clear all cache entries."""
        count = len(self.cache)
        self.cache.clear()

        if count > 0:
            logger.info(f"Cleared all {count} cache entries")

        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = datetime.now()
        active_entries = sum(1 for entry in self.cache.values() if now <= entry["expires_at"])
        expired_entries = len(self.cache) - active_entries

        return {
            "total_entries": len(self.cache),
            "active_entries": active_entries,
            "expired_entries": expired_entries,
            "default_ttl": self.default_ttl
        }
```

## 🚀 Step 8: Complete Integration Example

### 8.1 Create Enhanced Three-Agent Workflow with MCP

Create the file `generate_newspaper_with_mcp.py`:

```python
"""Complete newspaper generation using three-agent workflow enhanced with MCP."""

import asyncio
from datetime import datetime

from agents.agent_factory import AgentFactory
from agents.researcher_agent.research_models import ResearchRequest, EditorialReviewRequest
from agents.types import ReporterField
from core.config_service import ConfigService
from utils.forbidden_topics_tool import ForbiddenTopicsTool, ForbiddenTopicsParams
from utils.mcp_tool_registry import MCPToolRegistry


async def generate_newspaper_with_mcp():
    """Generate newspaper using MCP-enhanced research capabilities."""
    print("📰 BobTimes MCP-Enhanced Newspaper Generation")
    print("=" * 70)

    config_service = ConfigService()
    factory = AgentFactory(config_service)

    # Initialize MCP tools
    mcp_registry = MCPToolRegistry(config_service)
    available_mcp_tools = mcp_registry.list_tools()

    print(f"🔌 MCP Tools Available: {available_mcp_tools}")

    # Create agents
    researcher = factory.create_researcher()
    editor = factory.create_editor()
    reporters = {
        ReporterField.TECHNOLOGY: factory.create_reporter(ReporterField.TECHNOLOGY),
        ReporterField.SCIENCE: factory.create_reporter(ReporterField.SCIENCE),
        ReporterField.ECONOMICS: factory.create_reporter(ReporterField.ECONOMICS)
    }

    print("✅ Created all agents with MCP integration")

    # Phase 1: Enhanced Research with MCP
    print("\n🔬 PHASE 1: MCP-ENHANCED RESEARCH")
    print("-" * 40)

    start_time = datetime.now()

    # Research with MCP integration
    research_result = await researcher.research_trending_topics(
        fields=["technology", "science", "economics"],
        topics_per_field=4  # More topics due to enhanced research
    )

    research_duration = (datetime.now() - start_time).total_seconds()

    if not research_result["success"]:
        print(f"❌ Research failed: {research_result['error']}")
        return

    print(f"   ✅ MCP-enhanced research completed in {research_duration:.1f}s")
    print(f"   📊 Topics discovered: {len(research_result['topics_researched'])}")
    print(f"   🔍 Total sources gathered: {research_result['total_sources']}")

    # Analyze source distribution
    source_breakdown = {}
    mcp_sources = 0

    for topic_data in research_result["topics_researched"]:
        for source in topic_data["sources"]:
            source_type = source["source_type"]
            source_breakdown[source_type] = source_breakdown.get(source_type, 0) + 1

            # Count MCP sources (from Tavily)
            if "tavily" in source.get("metadata", {}).get("domain", "").lower():
                mcp_sources += 1

    print(f"   📋 Source breakdown: {dict(source_breakdown)}")
    print(f"   🔌 MCP sources: {mcp_sources}")

    # Show sample MCP-enhanced topics
    print(f"\n   🎯 Sample MCP-Enhanced Topics:")
    for i, topic_data in enumerate(research_result["topics_researched"][:3]):
        print(f"      {i+1}. {topic_data['topic_title']}")
        print(f"         Field: {topic_data['field']}")
        print(f"         Trending Score: {topic_data['trending_score']:.2f}")
        print(f"         Sources: {len(topic_data['sources'])} (depth: {topic_data['research_depth']})")

        # Show source quality
        avg_relevance = sum(s["relevance_score"] for s in topic_data["sources"]) / len(topic_data["sources"])
        print(f"         Avg Relevance: {avg_relevance:.2f}")

    # Phase 2: Editorial Review with Enhanced Data
    print("\n📝 PHASE 2: EDITORIAL REVIEW")
    print("-" * 30)

    # Get forbidden topics
    forbidden_tool = ForbiddenTopicsTool(config_service)
    forbidden_params = ForbiddenTopicsParams(operation="get_forbidden", days_back=30)
    forbidden_result = await forbidden_tool.execute(forbidden_params)
    forbidden_topics = forbidden_result.forbidden_topics if forbidden_result.success else []

    print(f"   🚫 Checking against {len(forbidden_topics)} forbidden topics")

    # Convert research for editorial review
    from agents.researcher_agent.research_models import ResearchResult, TopicResearch

    topics_researched = [TopicResearch(**topic_data) for topic_data in research_result["topics_researched"]]
    research_obj = ResearchResult(
        success=True,
        topics_researched=topics_researched,
        total_sources=research_result["total_sources"],
        research_summary=research_result["research_summary"],
        fields_covered=[ReporterField(f) for f in research_result["fields_covered"]],
        research_duration=research_result["research_duration"]
    )

    # Editorial review with enhanced criteria
    review_request = EditorialReviewRequest(
        research_result=research_obj,
        forbidden_topics=forbidden_topics,
        priority_fields=[ReporterField.TECHNOLOGY, ReporterField.SCIENCE],
        max_topics_to_select=8  # More topics due to better research quality
    )

    editorial_tool = next((tool for tool in editor.tools if tool.name == "editorial_research_review"), None)
    if not editorial_tool:
        print("❌ Editorial research review tool not found")
        return

    editorial_decision = await editorial_tool.execute(review_request)

    print(f"   ✅ Editorial review completed")
    print(f"   📝 Selected {len(editorial_decision.selected_topics)} high-quality topics")
    print(f"   ❌ Rejected {len(editorial_decision.rejected_topics)} topics")

    # Show selected topics with quality metrics
    print(f"\n   🎯 Selected Topics for Publication:")
    for topic in editorial_decision.selected_topics:
        priority = editorial_decision.assignment_priority.get(topic.topic_title, 0)
        avg_source_relevance = sum(s.relevance_score for s in topic.sources) / len(topic.sources)
        print(f"      ✅ {topic.topic_title}")
        print(f"         Field: {topic.field.value} | Priority: {priority} | Avg Relevance: {avg_source_relevance:.2f}")
        print(f"         Sources: {len(topic.sources)} | Trending: {topic.trending_score:.2f}")

    # Phase 3: Enhanced Story Writing
    print("\n✍️  PHASE 3: MCP-ENHANCED STORY WRITING")
    print("-" * 45)

    completed_stories = []
    writing_stats = {
        "total_time": 0,
        "total_words": 0,
        "total_sources_used": 0,
        "avg_research_coverage": 0
    }

    for topic in editorial_decision.selected_topics:
        try:
            print(f"\n   📝 Writing: {topic.topic_title}")

            # Get appropriate reporter
            reporter = reporters[topic.field]

            # Create enhanced story assignment
            from agents.researcher_agent.research_models import StoryAssignment

            assignment = StoryAssignment(
                topic_research=topic,
                reporter_field=topic.field,
                assignment_notes=editorial_decision.editorial_notes.get(topic.topic_title, ""),
                priority_level=editorial_decision.assignment_priority.get(topic.topic_title, 1),
                required_word_count=600,  # Longer stories due to richer research
                editorial_guidelines="Write comprehensive story leveraging MCP-enhanced research data with multiple high-quality sources."
            )

            # Find research story tool
            research_tool = next((tool for tool in reporter.tools if tool.name == "research_story_writer"), None)
            if not research_tool:
                print(f"   ⚠️  Research story tool not found for {topic.topic_title}")
                continue

            # Write story with enhanced parameters
            from agents.reporter_agent.research_story_tool import ResearchStoryParams

            story_params = ResearchStoryParams(
                story_assignment=assignment,
                writing_style="engaging",
                include_quotes=True,
                focus_angle="comprehensive"
            )

            story_result = await research_tool.execute(story_params)

            if story_result.success:
                completed_stories.append(story_result.story_draft)

                # Update stats
                writing_stats["total_time"] += story_result.writing_time
                writing_stats["total_words"] += story_result.word_count
                writing_stats["total_sources_used"] += story_result.sources_used
                writing_stats["avg_research_coverage"] += story_result.research_coverage

                print(f"      ✅ Story completed successfully")
                print(f"         Word count: {story_result.word_count}")
                print(f"         Sources used: {story_result.sources_used}/{len(topic.sources)}")
                print(f"         Research coverage: {story_result.research_coverage:.1%}")
                print(f"         Writing time: {story_result.writing_time:.1f}s")
            else:
                print(f"      ❌ Story failed: {story_result.error}")

        except Exception as e:
            print(f"   ❌ Story creation failed for {topic.topic_title}: {e}")
            continue

    # Calculate final stats
    if completed_stories:
        writing_stats["avg_research_coverage"] /= len(completed_stories)

    # Phase 4: Publication and Memory Update
    print("\n📰 PHASE 4: PUBLICATION")
    print("-" * 25)

    print(f"   📚 Stories completed: {len(completed_stories)}")

    # Update topic memory
    for story in completed_stories:
        from utils.topic_memory_models import CoveredTopic, TopicStatus

        covered_topic = CoveredTopic(
            title=story.title,
            description=story.summary,
            field=story.field,
            published_date=datetime.now().strftime("%Y-%m-%d"),
            story_id=f"story-{story.field.value}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            keywords=[],  # Could extract from content
            status=TopicStatus.PUBLISHED
        )

        add_params = ForbiddenTopicsParams(operation="add_topic", new_topic=covered_topic)
        await forbidden_tool.execute(add_params)

    print(f"   ✅ Updated topic memory with {len(completed_stories)} new topics")

    # Final Summary
    print(f"\n🎉 MCP-ENHANCED NEWSPAPER GENERATION COMPLETED!")
    print("=" * 60)

    # Field distribution
    field_counts = {}
    for story in completed_stories:
        field_counts[story.field.value] = field_counts.get(story.field.value, 0) + 1

    print(f"📊 FINAL STATISTICS:")
    print(f"   Stories by field: {dict(field_counts)}")
    print(f"   Total word count: {writing_stats['total_words']:,}")
    print(f"   Total sources used: {writing_stats['total_sources_used']}")
    print(f"   Average research coverage: {writing_stats['avg_research_coverage']:.1%}")
    print(f"   Total writing time: {writing_stats['total_time']:.1f}s")
    print(f"   Research time: {research_duration:.1f}s")
    print(f"   Total generation time: {(datetime.now() - start_time).total_seconds():.1f}s")

    # MCP Impact Analysis
    print(f"\n🔌 MCP IMPACT ANALYSIS:")
    print(f"   MCP sources integrated: {mcp_sources}")
    print(f"   Source quality improvement: Enhanced relevance scoring")
    print(f"   Research speed: Parallel MCP + traditional sources")
    print(f"   Content depth: Richer research data for better stories")

    # Show sample story
    if completed_stories:
        sample_story = completed_stories[0]
        print(f"\n📖 SAMPLE MCP-ENHANCED STORY:")
        print(f"   Title: {sample_story.title}")
        print(f"   Field: {sample_story.field.value}")
        print(f"   Word Count: {sample_story.word_count}")
        print(f"   Sources: {len(sample_story.sources)}")
        print(f"   Content Preview:")
        print(f"   {sample_story.content[:300]}...")

    # Cleanup
    await mcp_registry.cleanup()

    print(f"\n🏆 MCP-enhanced journalism pipeline completed successfully!")


if __name__ == "__main__":
    asyncio.run(generate_newspaper_with_mcp())
```

## 🎯 Step 9: Lab Summary and Best Practices

### 9.1 MCP Integration Summary

**✅ Key Achievements:**

1. **FastMCP Client Integration**: Seamless connection to MCP servers
2. **Tavily MCP Tool**: Advanced web search with AI-powered analysis
3. **Compatible Output Format**: MCP results formatted as ResearchSource objects
4. **Three-Agent Integration**: MCP tools work with existing workflow
5. **Performance Optimization**: Connection pooling and caching
6. **Error Handling**: Robust fallback mechanisms and retry logic

**🔧 Technical Components:**

- **MCPClientService**: Core MCP client management
- **TavilyMCPTool**: Tavily-specific research tool
- **MCPToolRegistry**: Tool registration and lifecycle management
- **MCPConnectionPool**: Performance optimization through connection pooling
- **MCPCache**: Caching layer for improved response times

### 9.2 MCP Benefits Realized

**🚀 Enhanced Research Capabilities:**
- **Real-time Web Search**: Access to current information via Tavily
- **AI-Powered Analysis**: Intelligent content summarization and relevance scoring
- **Source Credibility**: Automatic source reliability assessment
- **Comprehensive Coverage**: Multi-source research including MCP providers

**⚡ Performance Improvements:**
- **Parallel Processing**: MCP calls alongside traditional research
- **Connection Pooling**: Efficient resource utilization
- **Intelligent Caching**: Reduced redundant API calls
- **Concurrent Requests**: Multiple searches simultaneously

**🎯 Quality Enhancements:**
- **Higher Relevance Scores**: AI-powered relevance assessment
- **Better Source Diversity**: Mix of traditional and MCP sources
- **Enhanced Content Depth**: Richer research data for story writing
- **Improved Editorial Decisions**: Better data for topic selection

### 9.3 Best Practices for MCP Integration

**Configuration Management:**
1. **Secure API Keys**: Store in secrets.yaml, never in code
2. **Environment Variables**: Use .env files for non-sensitive settings
3. **Server Configuration**: Properly configure MCP server parameters
4. **Timeout Settings**: Set appropriate timeouts for reliability

**Performance Optimization:**
1. **Connection Pooling**: Use connection pools for high-throughput scenarios
2. **Caching Strategy**: Cache results with appropriate TTL values
3. **Concurrent Requests**: Leverage async/await for parallel processing
4. **Resource Limits**: Set reasonable limits on results and content length

**Error Handling:**
1. **Graceful Degradation**: Fall back to traditional methods if MCP fails
2. **Retry Logic**: Implement exponential backoff for transient failures
3. **Circuit Breaker**: Prevent cascading failures with circuit breaker pattern
4. **Monitoring**: Log MCP performance and error rates

**Integration Patterns:**
1. **Compatible Interfaces**: Ensure MCP tools match existing tool interfaces
2. **Unified Data Models**: Use consistent data models across all research sources
3. **Seamless Fallback**: Transparent fallback when MCP services are unavailable
4. **Quality Metrics**: Track and compare MCP vs traditional research quality

## 🏆 Lab 5 Complete!

Congratulations! You've successfully integrated Tavily MCP into your BobTimes newspaper system, creating a powerful hybrid research platform that combines traditional methods with cutting-edge MCP technology.

### **🎯 Major Achievements:**

1. **MCP Infrastructure**: Complete FastMCP client setup with Tavily integration
2. **Seamless Integration**: MCP tools work transparently with existing three-agent workflow
3. **Enhanced Research**: AI-powered web search with relevance scoring and content analysis
4. **Performance Optimization**: Connection pooling, caching, and concurrent processing
5. **Production Ready**: Robust error handling, monitoring, and fallback mechanisms

### **📈 System Capabilities Enhanced:**

- **Real-Time Research**: Access to current web information through Tavily
- **AI-Powered Analysis**: Intelligent content summarization and source evaluation
- **Hybrid Research**: Combination of YouTube, web search, scraping, and MCP sources
- **Quality Metrics**: Enhanced relevance scoring and source credibility assessment
- **Scalable Architecture**: Connection pooling and caching for high performance

### **🔄 Workflow Excellence:**

Your newspaper now operates with state-of-the-art research capabilities:
1. **Research** → Multi-source discovery including MCP-powered web search
2. **Editorial** → Enhanced decision making with richer research data
3. **Writing** → Stories based on comprehensive, AI-analyzed research
4. **Publication** → High-quality journalism with diverse, credible sources

Your BobTimes newspaper now leverages the power of Model Context Protocol to deliver cutting-edge, research-driven journalism with unprecedented depth and accuracy!

**🚀 Ready for Advanced Features?** Consider implementing:
- Additional MCP servers (Claude, GPT, specialized research tools)
- Real-time news monitoring and alerts via MCP
- Multi-language research and translation through MCP
- Advanced analytics and research quality metrics
- Custom MCP servers for specialized domains

🎉 **Lab 5 Complete!** Your newspaper now has MCP-powered research capabilities that rival professional newsrooms!
```
```
