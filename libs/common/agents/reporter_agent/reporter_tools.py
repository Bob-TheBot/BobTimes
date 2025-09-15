"""Reporter-specific tools for news research and content creation."""

from datetime import datetime
from enum import StrEnum
from typing import Any

from agents.models.search_models import CleanedSearchResult
from agents.models.story_models import StorySource
from agents.tools.base_tool import BaseTool
from core.llm_service import ModelSpeed
from core.logging_service import get_logger
from pydantic import BaseModel, Field

logger = get_logger(__name__)


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
class ReporterSearchTool(BaseTool):
    """Enhanced search tool for reporter agents with detailed output format."""

    def __init__(self) -> None:
        name = "search"
        description = f"""
Search for current information and news using DuckDuckGo.

PARAMETER SCHEMA:
{SearchParams.model_json_schema()}

CORRECT USAGE EXAMPLES:
{{"query": "AI tools technology trends", "search_type": "news", "time_limit": "w", "max_results": 5}}
{{"query": "OpenAI GPT release", "search_type": "news", "time_limit": "d"}}
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

    async def execute(self, params: BaseModel, model_speed: ModelSpeed = ModelSpeed.FAST) -> SearchToolResult:
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

            return SearchToolResult(
                search_type=params.search_type.value,
                query=params.query,
                success=True,
                cleaned_results=cleaned_results,
                error=None
            )
        except Exception as e:
            logger.error(f"Search tool execution failed: {e}")
            return SearchToolResult(
                search_type=params.search_type.value,
                query=params.query,
                success=False,
                error=str(e)
            )


class ReporterScraperTool(BaseTool):
    """Enhanced scraper tool for reporter agents with detailed output format."""

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

    async def execute(self, params: BaseModel, model_speed: ModelSpeed = ModelSpeed.FAST) -> ScraperToolResult:
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

            return ScraperToolResult(
                success=result.success,
                url=result.url,
                title=result.title,
                content=result.content,
                word_count=result.word_count,
                error=result.error_message
            )
        except Exception as e:
            logger.error(f"Scraper tool execution failed: {e}")
            return ScraperToolResult(
                success=False,
                url=params.url,
                error=str(e)
            )


class ReporterToolRegistry:
    """Registry for reporter tools with automatic schema generation."""

    def __init__(self) -> None:
        """Initialize the tool registry with reporter-specific tools."""
        self.tools = {
            "search": ReporterSearchTool(),
            "scrape": ReporterScraperTool()
        }

    def get_all_tools(self) -> list[BaseTool]:
        """Get all reporter tools as a list."""
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
                source = StorySource(
                    url=result.url,
                    title=result.title,
                    summary=getattr(result, "snippet", None) or getattr(result, "body", None),
                    accessed_at=datetime.now()
                )
                story_sources.append(source)

        return story_sources
