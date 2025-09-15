"""Researcher-specific tools for news research and content creation."""

from datetime import datetime
from enum import StrEnum
from typing import Any
import json

from agents.models.search_models import CleanedSearchResult
from agents.models.story_models import StorySource
from agents.tools.base_tool import BaseTool, UnifiedToolResult
from core.config_service import ConfigService
from core.llm_service import ModelSpeed
from core.logging_service import get_logger
from pydantic import BaseModel, Field, PrivateAttr

logger = get_logger(__name__)


def _normalize_topic_name(topic: str) -> str:
    """Normalize topic name for consistent storage and retrieval."""
    return topic.strip().title()


# ============================================================================
# MEMORY FETCH TOOL
# ============================================================================

class FetchFromMemoryParams(BaseModel):
    """Parameters for fetching content from SharedMemoryStore."""
    topic_key: str = Field(description="Exact topic key from memory to fetch content for")
    field: str = Field(description="Field to search within (technology/economics/science)")


class FetchFromMemoryResult(BaseModel):
    """Result of fetching content from memory."""
    success: bool = Field(description="Whether the fetch was successful")
    topic_key: str = Field(description="The topic key that was fetched")
    sources_count: int = Field(default=0, description="Number of sources retrieved")
    content_summary: str = Field(default="", description="Summary of retrieved content")
    error: str | None = Field(default=None, description="Error message if fetch failed")


class FetchFromMemoryTool(BaseTool):
    """Fetch content from SharedMemoryStore for a specific topic key.

    This tool allows researchers to retrieve validated content that was previously
    stored in memory during topic discovery phase.
    """

    name: str = "fetch_from_memory"
    description: str = """
Fetch content from SharedMemoryStore using an exact topic key.
Use this when you have a topic assignment and need to retrieve the research content.

Available memory topics for your field will be shown in your context.
Pick the topic key that best matches your assignment and use it to fetch content.

Parameters:
- topic_key: Exact topic key from memory (must match exactly)
- field: Field to search within (technology/economics/science)

Usage: <tool>fetch_from_memory</tool><args>{"topic_key": "Best Ai Tools To Create Viral Content", "field": "technology"}</args>

Returns: Sources and content for the specified topic key
"""
    params_model: type[BaseModel] | None = FetchFromMemoryParams

    async def execute(self, params: BaseModel, model_speed: ModelSpeed = ModelSpeed.FAST) -> FetchFromMemoryResult:
        """Fetch content from SharedMemoryStore."""
        if not isinstance(params, FetchFromMemoryParams):
            return FetchFromMemoryResult(
                success=False,
                topic_key="",
                error="Invalid parameters provided"
            )

        try:
            from agents.shared_memory_store import get_shared_memory_store
            memory_store = get_shared_memory_store()

            # Get the specific memory entry
            memory_entry = memory_store.get_memory(params.topic_key)

            if not memory_entry:
                # List available topics for debugging
                available_topics = memory_store.list_topics(field=params.field)
                return FetchFromMemoryResult(
                    success=False,
                    topic_key=params.topic_key,
                    error=f"Topic key '{params.topic_key}' not found in memory. Available topics: {available_topics}"
                )

            # Verify field matches
            if memory_entry.field != params.field:
                return FetchFromMemoryResult(
                    success=False,
                    topic_key=params.topic_key,
                    error=f"Topic key '{params.topic_key}' belongs to field '{memory_entry.field}', not '{params.field}'"
                )

            # Create content summary
            content_parts = []
            for source in memory_entry.sources:
                if source.content:
                    content_parts.append(f"Source: {source.title}\nContent: {source.content[:200]}...")
                elif source.summary:
                    content_parts.append(f"Source: {source.title}\nSummary: {source.summary}")

            content_summary = "\n\n".join(content_parts)

            logger.info(
                f"📚 [FETCH-MEMORY] Retrieved content for topic: {params.topic_key}",
                sources_count=len(memory_entry.sources),
                field=params.field,
                content_length=len(content_summary)
            )

            return FetchFromMemoryResult(
                success=True,
                topic_key=params.topic_key,
                sources_count=len(memory_entry.sources),
                content_summary=content_summary,
                error=None
            )

        except Exception as e:
            logger.error(f"Failed to fetch from memory: {e}")
            return FetchFromMemoryResult(
                success=False,
                topic_key=params.topic_key,
                error=f"Error fetching from memory: {str(e)}"
            )


def _extract_topics_from_search_results(results: list[CleanedSearchResult], query: str) -> tuple[list[str], dict[str, Any]]:
    """Extract topics from search results and create topic-source mapping.

    Args:
        results: List of search results
        query: Original search query

    Returns:
        Tuple of (topics_list, topic_source_mapping)
    """
    topics: list[str] = []
    topic_source_mapping: dict[str, Any] = {}

    for result in results:
        if hasattr(result, 'title') and result.title:
            # Use the result title as the topic, normalized
            topic_name = _normalize_topic_name(result.title)

            # Get available content - search results usually only have snippets
            snippet = getattr(result, 'snippet', getattr(result, 'body', ''))
            full_content = getattr(result, 'content', None)
            url = getattr(result, 'url', getattr(result, 'href', ''))

            # Validate content quality - we need substantial content for topics
            content_length = len(full_content) if full_content else len(snippet)
            has_substantial_content = content_length > 100  # Minimum content threshold

            # Create source for this topic
            source = StorySource(
                url=url,
                title=result.title,
                summary=snippet,
                content=full_content,
                source_type='search',
                accessed_at=datetime.now()
            )

            # Map topic to source with content validation info
            topic_source_mapping[topic_name] = {
                'source': source,
                'query': query,
                'needs_content_fetch': not has_substantial_content,  # Flag for fallback
                'content_length': content_length
            }

            topics.append(topic_name)

    return topics, topic_source_mapping


# Search Tool Models
class SearchType(StrEnum):
    """Types of search operations."""
    TEXT = "text"
    NEWS = "news"
    IMAGES = "images"


class SearchParams(BaseModel):
    """Parameters for search operations."""
    query: str = Field(description="Search query string")
    search_type: SearchType = Field(default=SearchType.NEWS, description="Type of search to perform")
    max_results: int = Field(default=4, description="Maximum number of results to return")
    time_limit: str | None = Field(default=None, description="Time limit for search (d, w, m, y)")


class SearchToolResult(BaseModel):
    """Result from search tool execution."""
    search_type: str
    query: str
    success: bool
    cleaned_results: list[CleanedSearchResult] = Field(default_factory=lambda: [])
    topics_extracted: list[str] = Field(default_factory=list, description="Topics extracted from search results")
    topic_source_mapping: dict[str, Any] = Field(default_factory=dict, description="Mapping of topics to their sources")
    error: str | None = None


# Scraper Tool Models
class ScrapeParams(BaseModel):
    """Parameters for scraping operations."""
    url: str = Field(description="URL to scrape")


class ScraperToolResult(BaseModel):
    """Result from scraper tool execution."""
    success: bool
    url: str
    title: str | None = None
    content: str | None = None
    word_count: int = 0
    error: str | None = None


# Tool Implementations
class ResearcherSearchTool(BaseTool):
    """Enhanced search tool for researcher agents with detailed output format."""

    def __init__(self) -> None:
        name = "search"
        description = f"""
Search for current information and news using DuckDuckGo.

PARAMETER SCHEMA:
{SearchParams.model_json_schema()}

CORRECT USAGE EXAMPLES:
{{"query": "AI tools technology trends", "search_type": "news", "time_limit": "w", "max_results": 5}}
{{"query": "OpenAI GPT-4 release", "search_type": "news", "time_limit": "d"}}
{{"query": "machine learning background", "search_type": "text", "max_results": 3}}

USAGE GUIDELINES:
- ALWAYS use "news" search_type for breaking news and current events
- Use "text" search_type only for background information or historical context
- Include time_limit parameter: "d" (day), "w" (week), "m" (month), "y" (year)
- Optimize queries with specific terms for better results
- Use max_results to control the number of results (default: 4)

RETURNS:
- search_type: Type of search performed
- query: The search query used
- success: Whether the search was successful
- cleaned_results: List of cleaned search results with url, title, snippet
- error: Error message if search failed

The tool automatically extracts and cleans search results for easy consumption.
When searching for news, always include recent time filters for current events.
"""
        super().__init__(name=name, description=description)
        self.params_model = SearchParams

    async def execute(self, params: BaseModel, model_speed: ModelSpeed = ModelSpeed.FAST) -> UnifiedToolResult:
        """Execute search with DuckDuckGo API."""
        if not isinstance(params, SearchParams):
            raise ValueError(f"Expected SearchParams, got {type(params)}")

        try:
            # Import here to avoid circular imports
            from utils.duckduckgo_search_tool import DDGRegion, DDGSafeSearch, DDGTimeLimit, DuckDuckGoSearchTool, NewsSearchResult, TextSearchResult

            # Create the underlying search tool
            search_tool = DuckDuckGoSearchTool()

            # Convert time_limit string to DDGTimeLimit enum
            time_limit_enum = None
            if params.time_limit:
                if params.time_limit == "d":
                    time_limit_enum = DDGTimeLimit.day
                elif params.time_limit == "w":
                    time_limit_enum = DDGTimeLimit.week
                elif params.time_limit == "m":
                    time_limit_enum = DDGTimeLimit.month
                elif params.time_limit == "y":
                    time_limit_enum = DDGTimeLimit.year

            # Execute the search based on search type - methods return lists, not result objects
            if params.search_type == SearchType.NEWS:
                results = search_tool.search_news(
                    query=params.query,
                    region=DDGRegion.united_states,
                    safe_search=DDGSafeSearch.moderate,
                    time_limit=time_limit_enum,
                    max_results=params.max_results
                )
            else:
                results = search_tool.search_text(
                    query=params.query,
                    region=DDGRegion.united_states,
                    safe_search=DDGSafeSearch.moderate,
                    time_limit=time_limit_enum,
                    max_results=params.max_results
                )

            # Convert results to CleanedSearchResult format based on search type
            from typing import cast
            cleaned_results = []
            for result in results:
                if params.search_type == SearchType.NEWS:
                    # News search returns NewsSearchResult with url, body, source, date, image
                    news_result = cast("NewsSearchResult", result)
                    cleaned_result = CleanedSearchResult(
                        title=news_result.title,
                        url=news_result.url,
                        content=news_result.body,
                        image_url=news_result.image,
                        image_local_path=None,
                        image_size_kb=None,
                        source=news_result.source,
                        date=news_result.date
                    )
                else:
                    # Text search returns TextSearchResult with href, body (no source, date, image)
                    text_result = cast("TextSearchResult", result)
                    cleaned_result = CleanedSearchResult(
                        title=text_result.title,
                        url=text_result.href,
                        content=text_result.body,
                        image_url=None,
                        image_local_path=None,
                        image_size_kb=None,
                        source=None,
                        date=None
                    )
                cleaned_results.append(cleaned_result)

            # Extract topics from results for SharedMemoryStore
            topics_extracted, topic_source_mapping = _extract_topics_from_search_results(cleaned_results, params.query)

            # Convert to story sources for unified interface
            story_sources = ResearcherToolRegistry.convert_search_results_to_sources(cleaned_results)

            return UnifiedToolResult(
                success=True,
                operation="search",
                query=params.query,
                sources=story_sources,
                topics_extracted=topics_extracted,
                topic_source_mapping=topic_source_mapping,
                metadata={
                    "search_type": params.search_type.value,
                    "results_count": len(cleaned_results),
                    "time_limit": params.time_limit
                },
                summary=f"Found {len(cleaned_results)} search results for '{params.query}'",
                error=None
            )
        except Exception as e:
            logger.error(f"Search tool execution failed: {e}")
            return UnifiedToolResult(
                success=False,
                operation="search",
                query=params.query,
                sources=[],
                topics_extracted=[],
                topic_source_mapping={},
                metadata={"search_type": params.search_type.value},
                summary=None,
                error=str(e)
            )


class ResearcherScraperTool(BaseTool):
    """Enhanced scraper tool for researcher agents with detailed output format."""

    def __init__(self) -> None:
        name = "scrape"
        description = f"""
Scrape detailed content from a specific web page URL with interactive content support.

PARAMETER SCHEMA:
{ScrapeParams.model_json_schema()}

CORRECT USAGE EXAMPLES:
{{"url": "https://example.com/article"}}

INCORRECT USAGE (DO NOT USE):
{{"urls": ["https://example.com"]}} ❌ Wrong parameter name - use 'url' not 'urls'
{{"url": ["https://example.com"]}} ❌ Wrong type - use string not array

USAGE GUIDELINES:
- Parameter name must be 'url' (singular)
- Value must be a single URL string (not a list)
- Use for getting detailed information from credible sources found via search
- The tool extracts readable text content, handling JavaScript-heavy sites
- INTERACTIVE CONTENT SUPPORT: Automatically handles pages that require clicking buttons like:
  * "Continue reading" / "Read more" buttons
  * "Show more" / "Expand" buttons
  * Cookie consent popups and overlays
  * Content that loads dynamically after user interaction
- Only scrape ONE URL at a time - call multiple times for multiple URLs

RETURNS:
- success: Whether scraping was successful
- url: The URL that was scraped
- title: Page title if available
- content: Cleaned text content from the page
- word_count: Number of words in the extracted content
- error: Error message if scraping failed

Use this tool to get detailed facts and quotes from sources discovered through search.
Always verify the source credibility before using scraped content in stories.
"""
        super().__init__(name=name, description=description)
        self.params_model = ScrapeParams

    async def execute(self, params: BaseModel, model_speed: ModelSpeed = ModelSpeed.FAST) -> UnifiedToolResult:
        """Execute scraping with web scraper."""
        if not isinstance(params, ScrapeParams):
            raise ValueError(f"Expected ScrapeParams, got {type(params)}")

        try:
            # Import here to avoid circular imports
            from utils.playwright_scraper import AsyncPlaywrightScraper

            # Create the underlying scraper tool with interactive content handling
            scraper_tool = AsyncPlaywrightScraper(
                timeout=60,
                wait_time=3000,  # Wait longer for dynamic content
                headless=True,
                handle_interactive_content=True,  # Enable interactive content handling
                max_content_buttons=3  # Limit button clicks for efficiency
            )

            # Execute the scraping
            result = await scraper_tool.scrape_url(params.url)

            # Create story source from scraped content
            sources = []
            if result.success and result.content:
                source = StorySource(
                    url=result.url,
                    title=result.title or "Scraped Content",
                    summary=result.content[:200] + "..." if len(result.content) > 200 else result.content,
                    content=result.content,
                    source_type="scrape",
                    accessed_at=datetime.now()
                )
                sources.append(source)

                # Extract topic from title or URL
                topic_name = _normalize_topic_name(result.title or result.url.split('/')[-1])
                topic_source_mapping = {
                    topic_name: {
                        'source': source,
                        'url': result.url
                    }
                }
                topics_extracted = [topic_name]
            else:
                topic_source_mapping = {}
                topics_extracted = []

            return UnifiedToolResult(
                success=result.success,
                operation="scrape",
                query=result.url,
                sources=sources,
                topics_extracted=topics_extracted,
                topic_source_mapping=topic_source_mapping,
                metadata={
                    "word_count": result.word_count,
                    "title": result.title
                },
                summary=f"Scraped {result.word_count} words from {result.url}" if result.success else None,
                error=result.error_message
            )
        except Exception as e:
            logger.error(f"Scraper tool execution failed: {e}")
            return UnifiedToolResult(
                success=False,
                operation="scrape",
                query=params.url,
                sources=[],
                topics_extracted=[],
                topic_source_mapping={},
                metadata={},
                summary=None,
                error=str(e)
            )


# ============================================================================
# Tavily MCP-backed Tools (Search/Scrape)
# ============================================================================
class TavilyMCPSearchTool(BaseTool):
    """Search tool powered by Tavily MCP.

    Falls back to existing parsing if response schema varies.
    """

    _config_service: ConfigService | None = PrivateAttr(default=None)

    def __init__(self, config_service: ConfigService | None = None) -> None:
        name = "search"
        description = f"""
Use Tavily MCP for real-time web/news search.

PARAMETER SCHEMA:
{SearchParams.model_json_schema()}

Guidelines:
- Prefer NEWS for current events; TEXT for background.
- Results are normalized to CleanedSearchResult and StorySource.
"""
        super().__init__(name=name, description=description)
        self.params_model = SearchParams
        self._config_service = config_service or ConfigService()

    async def execute(self, params: BaseModel, model_speed: ModelSpeed = ModelSpeed.FAST) -> UnifiedToolResult:
        if not isinstance(params, SearchParams):
            raise ValueError(f"Expected SearchParams, got {type(params)}")

        # Lazy import to avoid hard dependency if not configured
        try:
            from mcp_clients.tavily_mcp_client import TavilyMCPClient  # type: ignore
        except Exception as e:
            logger.error(f"TavilyMCPClient import failed: {e}")
            return UnifiedToolResult(
                success=False,
                operation="search",
                query=params.query,
                sources=[],
                topics_extracted=[],
                topic_source_mapping={},
                metadata={},
                summary=None,
                error="Tavily MCP client unavailable. Please install fastmcp and configure tavily.api_key"
            )

        client = TavilyMCPClient(self._config_service)
        try:
            await client.connect()

            # Resolve a search tool name from Tavily server
            tool_name = await client.resolve_tool(["search", "tavily_search", "web_search"])
            if not tool_name:
                raise RuntimeError("No Tavily search tool found on MCP server")

            args: dict[str, Any] = {"query": params.query}
            raw = await client.call_tool(tool_name, arguments=args)

            # Normalize results into CleanedSearchResult
            cleaned_results: list[CleanedSearchResult] = []
            try:
                payload = raw
                # Handle FastMCP CallToolResult shape: has .content which may include text or json
                if hasattr(raw, "content") and isinstance(getattr(raw, "content"), list):
                    for part in getattr(raw, "content"):
                        # Try JSON content first
                        data = None
                        if hasattr(part, "json") and getattr(part, "json") is not None:
                            data = getattr(part, "json")
                        elif isinstance(part, dict) and part.get("type") == "json" and "json" in part:
                            data = part.get("json")
                        elif hasattr(part, "text") and isinstance(getattr(part, "text"), str):
                            # Sometimes text holds JSON string
                            try:
                                data = json.loads(getattr(part, "text"))
                            except Exception:
                                data = None
                        elif isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                            try:
                                data = json.loads(part.get("text", ""))
                            except Exception:
                                data = None

                        if isinstance(data, dict):
                            # Tavily tools often return {"results": [...]} in json
                            payload = data.get("results") or data
                        elif isinstance(data, list):
                            payload = data

                # Handle plain dict/list payloads as before
                if isinstance(payload, dict) and "results" in payload:
                    items = payload.get("results") or []
                elif isinstance(payload, list):
                    items = payload
                else:
                    items = []

                for item in items:
                    if isinstance(item, dict):
                        title = str(item.get("title") or item.get("url") or params.query)
                        url = str(item.get("url") or "")
                        snippet = str(item.get("snippet") or item.get("content") or item.get("text") or "")
                        cleaned_results.append(
                            CleanedSearchResult(
                                title=title,
                                url=url,
                                content=snippet,
                                image_url=None,
                                image_local_path=None,
                                image_size_kb=None,
                                source=item.get("source"),
                                date=item.get("published_date") or item.get("date"),
                            )
                        )
                # Fallbacks: content array with text, or raw string
                if not cleaned_results and hasattr(raw, "content"):
                    for part in getattr(raw, "content") or []:
                        text = getattr(part, "text", None) if hasattr(part, "text") else (part.get("text") if isinstance(part, dict) else None)
                        if isinstance(text, str) and text.strip():
                            cleaned_results.append(
                                CleanedSearchResult(
                                    title=params.query,
                                    url="",
                                    content=text,
                                    image_url=None,
                                    image_local_path=None,
                                    image_size_kb=None,
                                    source=None,
                                    date=None,
                                )
                            )
                            break
                if not cleaned_results and isinstance(raw, str):
                    cleaned_results.append(
                        CleanedSearchResult(
                            title=params.query,
                            url="",
                            content=raw,
                            image_url=None,
                            image_local_path=None,
                            image_size_kb=None,
                            source=None,
                            date=None,
                        )
                    )
            except Exception as e:
                logger.error(f"Failed to parse Tavily search response: {e}")

            topics_extracted, topic_source_mapping = _extract_topics_from_search_results(cleaned_results, params.query)
            story_sources = ResearcherToolRegistry.convert_search_results_to_sources(cleaned_results)

            return UnifiedToolResult(
                success=True,
                operation="search",
                query=params.query,
                sources=story_sources,
                topics_extracted=topics_extracted,
                topic_source_mapping=topic_source_mapping,
                metadata={"provider": "tavily_mcp", "results_count": len(cleaned_results)},
                summary=f"Found {len(cleaned_results)} Tavily results for '{params.query}'",
                error=None,
            )
        except Exception as e:
            logger.error(f"Tavily MCP search execution failed: {e}")
            return UnifiedToolResult(
                success=False,
                operation="search",
                query=params.query,
                sources=[],
                topics_extracted=[],
                topic_source_mapping={},
                metadata={"provider": "tavily_mcp"},
                summary=None,
                error=str(e),
            )
        finally:
            try:
                await client.close()
            except Exception:
                pass


class TavilyMCPScrapeTool(BaseTool):
    """Scrape/Read tool powered by Tavily MCP with graceful fallback to Playwright."""

    _config_service: ConfigService | None = PrivateAttr(default=None)

    def __init__(self, config_service: ConfigService | None = None) -> None:
        name = "scrape"
        description = f"""
Read full content from a specific URL using Tavily MCP. Falls back to Playwright if Tavily read tool isn't available.

PARAMETER SCHEMA:
{ScrapeParams.model_json_schema()}
"""
        super().__init__(name=name, description=description)
        self.params_model = ScrapeParams
        self._config_service = config_service or ConfigService()

    async def _try_tavily(self, url: str) -> tuple[bool, dict[str, Any]]:
        try:
            from mcp_clients.tavily_mcp_client import TavilyMCPClient  # type: ignore
            client = TavilyMCPClient(self._config_service)
            await client.connect()
            tool_name = await client.resolve_tool([
                "browse", "read", "get_content", "scrape", "web.get", "url_to_text"
            ])
            if not tool_name:
                return False, {"error": "No Tavily content tool found"}
            raw = await client.call_tool(tool_name, {"url": url})
            await client.close()
            # Normalize
            title = None
            content = None
            # FastMCP CallToolResult with content parts
            if hasattr(raw, "content") and isinstance(getattr(raw, "content"), list):
                for part in getattr(raw, "content"):
                    # Prefer json payload
                    data = None
                    if hasattr(part, "json") and getattr(part, "json") is not None:
                        data = getattr(part, "json")
                    elif isinstance(part, dict) and part.get("type") == "json" and "json" in part:
                        data = part.get("json")
                    elif hasattr(part, "text") and isinstance(getattr(part, "text"), str):
                        # Sometimes text holds raw content
                        data = {"content": getattr(part, "text")}
                    elif isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                        data = {"content": part.get("text")}

                    if isinstance(data, dict):
                        title = data.get("title") or data.get("page_title") or title
                        c = data.get("content") or data.get("text") or data.get("body")
                        if isinstance(c, str) and c.strip():
                            content = c
                            break
            # Plain dict or string
            if content is None and isinstance(raw, dict):
                title = raw.get("title") or raw.get("page_title")
                content = raw.get("content") or raw.get("text") or raw.get("body")
            elif content is None and isinstance(raw, str):
                content = raw
            if content:
                return True, {"title": title, "content": content}
            return False, {"error": "No content returned from Tavily"}
        except Exception as e:
            return False, {"error": str(e)}

    async def execute(self, params: BaseModel, model_speed: ModelSpeed = ModelSpeed.FAST) -> UnifiedToolResult:
        if not isinstance(params, ScrapeParams):
            raise ValueError(f"Expected ScrapeParams, got {type(params)}")

        # First try Tavily MCP
        ok, data = await self._try_tavily(params.url)
        if ok:
            content = data.get("content") or ""
            title = data.get("title") or "Scraped Content"
            source = StorySource(
                url=params.url,
                title=title,
                summary=content[:200] + "..." if len(content) > 200 else content,
                content=content,
                source_type="scrape",
                accessed_at=datetime.now(),
            )
            topic = _normalize_topic_name(title or params.url.split("/")[-1])
            return UnifiedToolResult(
                success=True,
                operation="scrape",
                query=params.url,
                sources=[source],
                topics_extracted=[topic],
                topic_source_mapping={topic: {"source": source, "url": params.url}},
                metadata={"provider": "tavily_mcp", "word_count": len(content.split())},
                summary=f"Fetched content from {params.url}",
                error=None,
            )

        # Fallback to Playwright scraper
        try:
            from utils.playwright_scraper import AsyncPlaywrightScraper
            scraper_tool = AsyncPlaywrightScraper(timeout=60, wait_time=3000, headless=True, handle_interactive_content=True, max_content_buttons=3)
            result = await scraper_tool.scrape_url(params.url)
            sources = []
            if result.success and result.content:
                source = StorySource(
                    url=result.url,
                    title=result.title or "Scraped Content",
                    summary=result.content[:200] + "..." if len(result.content) > 200 else result.content,
                    content=result.content,
                    source_type="scrape",
                    accessed_at=datetime.now(),
                )
                sources.append(source)
                topic_name = _normalize_topic_name(result.title or result.url.split("/")[-1])
                topic_source_mapping = {topic_name: {"source": source, "url": result.url}}
                topics_extracted = [topic_name]
            else:
                topic_source_mapping = {}
                topics_extracted = []

            return UnifiedToolResult(
                success=result.success,
                operation="scrape",
                query=params.url,
                sources=sources,
                topics_extracted=topics_extracted,
                topic_source_mapping=topic_source_mapping,
                metadata={"provider": "playwright", "word_count": result.word_count, "title": result.title},
                summary=f"Scraped {result.word_count} words from {result.url}" if result.success else None,
                error=result.error_message,
            )
        except Exception as e:
            logger.error(f"Scraper tool execution failed: {e}")
            return UnifiedToolResult(
                success=False,
                operation="scrape",
                query=params.url,
                sources=[],
                topics_extracted=[],
                topic_source_mapping={},
                metadata={"provider": "playwright"},
                summary=None,
                error=str(e),
            )


class ResearcherToolRegistry:
    """Registry for researcher tools with automatic schema generation."""

    def __init__(self, config_service: ConfigService | None = None) -> None:
        """Initialize the tool registry with researcher-specific tools."""
        self.config_service = config_service or ConfigService()

        # Import YouTube tool here to avoid circular imports
        from utils.youtube_tool import YouTubeResearcherTool

        # Prefer Tavily MCP tools when API key is configured; fallback otherwise
        if self.config_service.get("tavily.api_key") or self.config_service.get("tavili.api_key"):
            search_tool = TavilyMCPSearchTool(self.config_service)
            scrape_tool = TavilyMCPScrapeTool(self.config_service)
        else:
            search_tool = ResearcherSearchTool()
            scrape_tool = ResearcherScraperTool()

        self.tools = {
            "search": search_tool,
            "scrape": scrape_tool,
            "youtube_search": YouTubeResearcherTool(self.config_service),  # Add YouTube tool
            "fetch_from_memory": FetchFromMemoryTool(),  # Add memory fetch tool
        }

    def get_all_tools(self) -> list[BaseTool]:
        """Get all researcher tools as a list."""
        return list(self.tools.values())

    def get_tool_by_name(self, name: str) -> BaseTool | None:
        """Get a specific tool by name."""
        return self.tools.get(name)

    def get_tool_descriptions(self) -> dict[str, str]:
        """Get tool names and descriptions for prompt generation."""
        return {name: tool.description for name, tool in self.tools.items()}

    def get_tool_schemas(self) -> dict[str, dict[str, Any]]:
        """Get tool parameter schemas for prompt generation."""
        schemas: dict[str, dict[str, Any]] = {}
        for name, tool in self.tools.items():
            if tool.params_model:
                schemas[name] = tool.params_model.model_json_schema()
        return schemas

    def format_tools_for_prompt(self) -> str:
        """Format all tools for inclusion in system prompt."""
        tool_sections = []

        for name, tool in self.tools.items():
            section = f"## {name.upper()}\n**Tool name:** `{name}`\n\n{tool.description.strip()}"
            tool_sections.append(section)

        return "\n\n".join(tool_sections)

    @staticmethod
    def convert_search_results_to_sources(search_results: list[CleanedSearchResult]) -> list[StorySource]:
        """Convert search results to StorySource objects.

        Args:
            search_results: List of CleanedSearchResult objects

        Returns:
            List of StorySource objects
        """
        story_sources: list[StorySource] = []
        for result in search_results:
            if hasattr(result, "url") and hasattr(result, "title"):
                # Get full content if available, otherwise use snippet/body
                full_content = getattr(result, "content", None) or getattr(result, "full_text", None)
                summary_text = getattr(result, "snippet", None) or getattr(result, "body", None)

                source = StorySource(
                    url=result.url,
                    title=result.title,
                    summary=summary_text,
                    content=full_content,  # Include full scraped content if available
                    source_type="search",
                    accessed_at=datetime.now()
                )
                story_sources.append(source)

        return story_sources

    @staticmethod
    def convert_youtube_results_to_sources(youtube_result: Any) -> list[StorySource]:
        """Convert YouTube tool results to StorySource objects.

        Args:
            youtube_result: YouTubeToolResult object

        Returns:
            List of StorySource objects
        """
        # Use Pydantic model validation instead of hasattr
        try:
            return youtube_result.sources if youtube_result.sources else []
        except AttributeError:
            logger.warning("YouTubeToolResult missing sources field")
            return []
