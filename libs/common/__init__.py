"""Common utilities and shared components for BobTimes."""

from .core import (
    ConfigService,
    LLMService,
    config_service,
)
from .utils import (
    DDGImageColor,
    DDGImageLayout,
    DDGImageLicense,
    DDGImageSize,
    DDGImageType,
    DDGRegion,
    DDGSafeSearch,
    DDGTimeLimit,
    DDGVideoDuration,
    DDGVideoLicense,
    DDGVideoResolution,
    DuckDuckGoSearchTool,
    ImageSearchResult,
    NewsSearchResult,
    AsyncPlaywrightScraper,
    ScrapedContent,
    SearchResponse,
    TextSearchResult,
    VideoSearchResult,
    create_duckduckgo_search_tool,
    create_async_playwright_scraper,
)

__version__ = "0.1.0"

__all__ = [
    # Core services
    "LLMService",
    "ConfigService",
    "config_service",
    # Utils
    "DuckDuckGoSearchTool",
    "create_duckduckgo_search_tool",
    "DDGRegion",
    "DDGSafeSearch",
    "DDGTimeLimit",
    "DDGImageSize",
    "DDGImageColor",
    "DDGImageType",
    "DDGImageLayout",
    "DDGImageLicense",
    "DDGVideoResolution",
    "DDGVideoDuration",
    "DDGVideoLicense",
    "TextSearchResult",
    "ImageSearchResult",
    "VideoSearchResult",
    "NewsSearchResult",
    "SearchResponse",
    "AsyncPlaywrightScraper",
    "create_async_playwright_scraper",
    "ScrapedContent",
]
