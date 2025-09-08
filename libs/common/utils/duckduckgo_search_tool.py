"""DuckDuckGo Search Tool for LLM Integration

This module provides a comprehensive search tool that LLMs can use to perform
web searches using DuckDuckGo. It supports text, image, video, and news searches
with fully typed responses using Pydantic models.
"""

from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from core.logging_service import get_logger
from ddgs import DDGS
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ddgs import DDGS

logger = get_logger(__name__)


class DDGRegion(StrEnum):
    """Supported DuckDuckGo search regions."""
    arabia = "xa-ar"
    arabia_en = "xa-en"
    argentina = "ar-es"
    australia = "au-en"
    austria = "at-de"
    belgium_fr = "be-fr"
    belgium_nl = "be-nl"
    brazil = "br-pt"
    bulgaria = "bg-bg"
    canada = "ca-en"
    canada_fr = "ca-fr"
    catalan = "ct-ca"
    chile = "cl-es"
    china = "cn-zh"
    colombia = "co-es"
    croatia = "hr-hr"
    czech_republic = "cz-cs"
    denmark = "dk-da"
    estonia = "ee-et"
    finland = "fi-fi"
    france = "fr-fr"
    germany = "de-de"
    greece = "gr-el"
    hong_kong = "hk-tzh"
    hungary = "hu-hu"
    india = "in-en"
    indonesia = "id-id"
    indonesia_en = "id-en"
    ireland = "ie-en"
    israel = "il-he"
    italy = "it-it"
    japan = "jp-jp"
    korea = "kr-kr"
    latvia = "lv-lv"
    lithuania = "lt-lt"
    latin_america = "xl-es"
    malaysia = "my-ms"
    malaysia_en = "my-en"
    mexico = "mx-es"
    netherlands = "nl-nl"
    new_zealand = "nz-en"
    norway = "no-no"
    peru = "pe-es"
    philippines = "ph-en"
    philippines_tl = "ph-tl"
    poland = "pl-pl"
    portugal = "pt-pt"
    romania = "ro-ro"
    russia = "ru-ru"
    singapore = "sg-en"
    slovak_republic = "sk-sk"
    slovenia = "sl-sl"
    south_africa = "za-en"
    spain = "es-es"
    sweden = "se-sv"
    switzerland_de = "ch-de"
    switzerland_fr = "ch-fr"
    switzerland_it = "ch-it"
    taiwan = "tw-tzh"
    thailand = "th-th"
    turkey = "tr-tr"
    ukraine = "ua-uk"
    united_kingdom = "uk-en"
    united_states = "us-en"
    united_states_es = "ue-es"
    venezuela = "ve-es"
    vietnam = "vn-vi"
    no_region = "wt-wt"


class DDGSafeSearch(StrEnum):
    """DuckDuckGo safe search options."""
    on = "on"
    moderate = "moderate"
    off = "off"


class DDGTimeLimit(StrEnum):
    """DuckDuckGo time limit options."""
    day = "d"
    week = "w"
    month = "m"
    year = "y"


class DDGImageSize(StrEnum):
    """DuckDuckGo image size options."""
    small = "Small"
    medium = "Medium"
    large = "Large"
    wallpaper = "Wallpaper"


class DDGImageColor(StrEnum):
    """DuckDuckGo image color options."""
    color = "color"
    monochrome = "Monochrome"
    red = "Red"
    orange = "Orange"
    yellow = "Yellow"
    green = "Green"
    blue = "Blue"
    purple = "Purple"
    pink = "Pink"
    brown = "Brown"
    black = "Black"
    gray = "Gray"
    teal = "Teal"
    white = "White"


class DDGImageType(StrEnum):
    """DuckDuckGo image type options."""
    photo = "photo"
    clipart = "clipart"
    gif = "gif"
    transparent = "transparent"
    line = "line"


class DDGImageLayout(StrEnum):
    """DuckDuckGo image layout options."""
    square = "Square"
    tall = "Tall"
    wide = "Wide"


class DDGImageLicense(StrEnum):
    """DuckDuckGo image license options."""
    any = "any"  # All Creative Commons
    public = "Public"  # Public Domain
    share = "Share"  # Free to Share and Use
    share_commercially = "ShareCommercially"  # Free to Share and Use Commercially
    modify = "Modify"  # Free to Modify, Share, and Use
    # Free to Modify, Share, and Use Commercially
    modify_commercially = "ModifyCommercially"


class DDGVideoResolution(StrEnum):
    """DuckDuckGo video resolution options."""
    high = "high"
    standard = "standard"


class DDGVideoDuration(StrEnum):
    """DuckDuckGo video duration options."""
    short = "short"
    medium = "medium"
    long = "long"


class DDGVideoLicense(StrEnum):
    """DuckDuckGo video license options."""
    creative_common = "creativeCommon"
    youtube = "youtube"


# Pydantic Models for Search Results

class TextSearchResult(BaseModel):
    """Represents a single text search result from DuckDuckGo."""
    title: str = Field(..., description="Title of the search result")
    href: str = Field(..., description="URL link to the full article")
    body: str = Field(..., description="Snippet/description of the content")


class ImageSearchResult(BaseModel):
    """Represents a single image search result from DuckDuckGo."""
    title: str = Field(..., description="Title of the image")
    image: str = Field(..., description="Direct URL to the full-size image")
    thumbnail: str = Field(..., description="URL to the thumbnail version")
    url: str = Field(..., description="URL to the page containing the image")
    height: int = Field(..., description="Height of the image in pixels")
    width: int = Field(..., description="Width of the image in pixels")
    source: str = Field(..., description="Source of the image (e.g., 'Bing')")


class VideoImages(BaseModel):
    """Represents video thumbnail images at different sizes."""
    large: str = Field(..., description="Large thumbnail URL")
    medium: str = Field(..., description="Medium thumbnail URL")
    motion: str = Field("", description="Motion thumbnail URL (may be empty)")
    small: str = Field(..., description="Small thumbnail URL")


class VideoStatistics(BaseModel):
    """Represents video statistics."""
    viewCount: int = Field(0, description="Number of views")


class VideoSearchResult(BaseModel):
    """Represents a single video search result from DuckDuckGo."""
    content: str = Field(..., description="Direct URL to the video")
    description: str = Field(...,
                             description="Description of the video content")
    duration: str = Field(...,
                          description="Duration of the video (e.g., '8:22')")
    embed_html: str = Field(..., description="HTML embed code for the video")
    embed_url: str = Field(..., description="Embeddable URL for the video")
    image_token: str = Field(...,
                             description="Unique token for the video image")
    images: VideoImages = Field(...,
                                description="Thumbnail images at different sizes")
    provider: str = Field(..., description="Video provider (e.g., 'Bing')")
    published: str = Field(..., description="Publication date in ISO format")
    publisher: str = Field(..., description="Publisher name (e.g., 'YouTube')")
    statistics: VideoStatistics = Field(..., description="Video statistics")
    title: str = Field(..., description="Title of the video")
    uploader: str = Field(..., description="Name of the uploader")


class NewsSearchResult(BaseModel):
    """Represents a single news search result from DuckDuckGo."""
    date: str = Field(..., description="Publication date in ISO format")
    title: str = Field(..., description="Title of the news article")
    body: str = Field(..., description="Summary/excerpt of the news content")
    url: str = Field(..., description="URL link to the full news article")
    image: str | None = Field(
        None, description="URL to associated image (if available)")
    source: str = Field(..., description="News source name")


# Typed default factories to satisfy type checkers for Pydantic default_factory lists

def _default_text_results() -> list[TextSearchResult]:
    return []


def _default_image_results() -> list[ImageSearchResult]:
    return []


def _default_video_results() -> list[VideoSearchResult]:
    return []


def _default_news_results() -> list[NewsSearchResult]:
    return []


class SearchResponse(BaseModel):
    """Comprehensive search response containing all types of results."""
    query: str = Field(..., description="The original search query")
    region: str = Field(..., description="Region used for the search")
    safe_search: str = Field(..., description="Safe search setting used")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When the search was performed")
    text_results: list[TextSearchResult] = Field(
        default_factory=_default_text_results, description="Text search results")
    image_results: list[ImageSearchResult] = Field(
        default_factory=_default_image_results, description="Image search results")
    video_results: list[VideoSearchResult] = Field(
        default_factory=_default_video_results, description="Video search results")
    news_results: list[NewsSearchResult] = Field(
        default_factory=_default_news_results, description="News search results")
    total_results: int = Field(
        0, description="Total number of results across all categories")


class DuckDuckGoSearchTool:
    """A comprehensive DuckDuckGo search tool that LLMs can use to perform web searches.

    This tool provides structured, typed responses for text, image, video, and news searches
    with support for various filtering options and regions.
    """

    def __init__(self, proxy: str | None = None, timeout: int = 10, verify: bool = True) -> None:
        """Initialize the DuckDuckGo search tool.

        Args:
            proxy: Proxy server URL (supports http/https/socks5)
            timeout: Request timeout in seconds
            verify: SSL verification for requests
        """
        self.proxy = proxy
        self.timeout = timeout
        self.verify = verify
        self._ddgs: "DDGS | None" = None

        logger.info(
            f"Initialized DuckDuckGoSearchTool with timeout={timeout}, verify={verify}")

    def _get_ddgs_instance(self) -> "DDGS":
        """Get or create a DDGS instance with current settings."""
        try:
            if self._ddgs is None:
                self._ddgs = DDGS(
                    proxy=self.proxy,
                    timeout=self.timeout,
                    verify=self.verify
                )
            return self._ddgs
        except ImportError as e:
            logger.exception(
                "ddgs package not installed. Install with: uv add ddgs")
            raise ImportError("ddgs package is required") from e

    def search_text(
        self,
        query: str,
        region: DDGRegion = DDGRegion.no_region,
        safe_search: DDGSafeSearch = DDGSafeSearch.moderate,
        time_limit: DDGTimeLimit | None = None,
        max_results: int = 10,
    ) -> list[TextSearchResult]:
        """Perform a text search using DuckDuckGo.

        Args:
            query: Search keywords
            region: Geographic region for search results
            safe_search: Safe search filtering level
            time_limit: Time limit for results (day, week, month, year)
            max_results: Maximum number of results to return

        Returns:
            List of text search results
        """
        try:
            ddgs = self._get_ddgs_instance()

            logger.info(
                f"Performing text search: '{query}' (region={region}, max_results={max_results})")

            results = ddgs.text(
                query=query,
                region=region.value,
                safesearch=safe_search.value,
                timelimit=time_limit.value if time_limit else None,
                max_results=max_results,
            )

            # Convert to Pydantic models
            text_results: list[TextSearchResult] = []
            for result in results:
                try:
                    text_result = TextSearchResult(
                        title=result.get("title", ""),
                        href=result.get("href", ""),
                        body=result.get("body", ""),
                    )
                    text_results.append(text_result)
                except Exception as e:
                    logger.warning(f"Failed to parse text result: {e}")
                    continue

            logger.info(f"Retrieved {len(text_results)} text results")
            return text_results

        except Exception as e:
            logger.exception(f"Text search failed: {e}")
            raise

    def search_images(
        self,
        query: str,
        region: DDGRegion = DDGRegion.united_states,
        safe_search: DDGSafeSearch = DDGSafeSearch.moderate,
        time_limit: DDGTimeLimit | None = None,
        size: DDGImageSize | None = None,
        color: DDGImageColor | None = None,
        type_image: DDGImageType | None = None,
        layout: DDGImageLayout | None = None,
        license_image: DDGImageLicense | None = None,
        max_results: int = 10,
    ) -> list[ImageSearchResult]:
        """Perform an image search using DuckDuckGo.

        Args:
            query: Search keywords
            region: Geographic region for search results
            safe_search: Safe search filtering level
            time_limit: Time limit for results
            size: Image size filter
            color: Image color filter
            type_image: Image type filter
            layout: Image layout filter
            license_image: Image license filter
            max_results: Maximum number of results to return

        Returns:
            List of image search results
        """
        try:
            ddgs = self._get_ddgs_instance()

            logger.info(
                f"Performing image search: '{query}' (region={region}, max_results={max_results})")

            results = ddgs.images(
                query=query,
                region=region.value,
                safesearch=safe_search.value,
                timelimit=time_limit.value if time_limit else None,
                size=size.value if size else None,
                color=color.value if color else None,
                type_image=type_image.value if type_image else None,
                layout=layout.value if layout else None,
                license_image=license_image.value if license_image else None,
                max_results=max_results,
            )

            # Convert to Pydantic models
            image_results: list[ImageSearchResult] = []
            for result in results:
                try:
                    image_result = ImageSearchResult(
                        title=result.get("title", ""),
                        image=result.get("image", ""),
                        thumbnail=result.get("thumbnail", ""),
                        url=result.get("url", ""),
                        height=result.get("height", 0),
                        width=result.get("width", 0),
                        source=result.get("source", ""),
                    )
                    image_results.append(image_result)
                except Exception as e:
                    logger.warning(f"Failed to parse image result: {e}")
                    continue

            logger.info(f"Retrieved {len(image_results)} image results")
            return image_results

        except Exception as e:
            logger.exception(f"Image search failed: {e}")
            raise

    def search_videos(
        self,
        query: str,
        region: DDGRegion = DDGRegion.united_states,
        safe_search: DDGSafeSearch = DDGSafeSearch.moderate,
        time_limit: DDGTimeLimit | None = None,
        resolution: DDGVideoResolution | None = None,
        duration: DDGVideoDuration | None = None,
        license_videos: DDGVideoLicense | None = None,
        max_results: int = 10,
    ) -> list[VideoSearchResult]:
        """Perform a video search using DuckDuckGo.

        Args:
            query: Search keywords
            region: Geographic region for search results
            safe_search: Safe search filtering level
            time_limit: Time limit for results
            resolution: Video resolution filter
            duration: Video duration filter
            license_videos: Video license filter
            max_results: Maximum number of results to return

        Returns:
            List of video search results
        """
        try:
            ddgs = self._get_ddgs_instance()

            logger.info(
                f"Performing video search: '{query}' (region={region}, max_results={max_results})")

            results = ddgs.videos(
                query=query,
                region=region.value,
                safesearch=safe_search.value,
                timelimit=time_limit.value if time_limit else None,
                resolution=resolution.value if resolution else None,
                duration=duration.value if duration else None,
                license_videos=license_videos.value if license_videos else None,
                max_results=max_results,
            )

            # Convert to Pydantic models
            video_results: list[VideoSearchResult] = []
            for result in results:
                try:
                    # Parse video images
                    images_data = result.get("images", {})
                    video_images = VideoImages(
                        large=images_data.get("large", ""),
                        medium=images_data.get("medium", ""),
                        motion=images_data.get("motion", ""),
                        small=images_data.get("small", ""),
                    )

                    # Parse video statistics
                    stats_data = result.get("statistics", {})
                    video_stats = VideoStatistics(
                        viewCount=stats_data.get("viewCount", 0)
                    )

                    video_result = VideoSearchResult(
                        content=result.get("content", ""),
                        description=result.get("description", ""),
                        duration=result.get("duration", ""),
                        embed_html=result.get("embed_html", ""),
                        embed_url=result.get("embed_url", ""),
                        image_token=result.get("image_token", ""),
                        images=video_images,
                        provider=result.get("provider", ""),
                        published=result.get("published", ""),
                        publisher=result.get("publisher", ""),
                        statistics=video_stats,
                        title=result.get("title", ""),
                        uploader=result.get("uploader", ""),
                    )
                    video_results.append(video_result)
                except Exception as e:
                    logger.warning(f"Failed to parse video result: {e}")
                    continue

            logger.info(f"Retrieved {len(video_results)} video results")
            return video_results

        except Exception:
            logger.exception("Video search failed")
            raise

    def search_news(
        self,
        query: str,
        region: DDGRegion = DDGRegion.united_states,
        safe_search: DDGSafeSearch = DDGSafeSearch.moderate,
        time_limit: DDGTimeLimit | None = None,
        max_results: int = 10,
    ) -> list[NewsSearchResult]:
        """Perform a news search using DuckDuckGo.

        Args:
            query: Search keywords
            region: Geographic region for search results
            safe_search: Safe search filtering level
            time_limit: Time limit for results
            max_results: Maximum number of results to return

        Returns:
            List of news search results
        """
        try:
            ddgs = self._get_ddgs_instance()

            logger.info(
                f"Performing news search: '{query}' (region={region}, max_results={max_results})")

            results = ddgs.news(
                query=query,
                region=region.value,
                safesearch=safe_search.value,
                timelimit=time_limit.value if time_limit else None,
                max_results=max_results,
            )

            # Convert to Pydantic models
            news_results: list[NewsSearchResult] = []
            for result in results:
                try:
                    news_result = NewsSearchResult(
                        date=result.get("date", ""),
                        title=result.get("title", ""),
                        body=result.get("body", ""),
                        url=result.get("url", ""),
                        image=result.get("image"),  # Can be None
                        source=result.get("source", ""),
                    )
                    news_results.append(news_result)
                except Exception as e:
                    logger.warning(f"Failed to parse news result: {e}")
                    continue

            logger.info(f"Retrieved {len(news_results)} news results")
            return news_results

        except Exception:
            logger.exception("News search failed")
            raise

    def comprehensive_search(
        self,
        query: str,
        region: DDGRegion = DDGRegion.no_region,
        safe_search: DDGSafeSearch = DDGSafeSearch.moderate,
        time_limit: DDGTimeLimit | None = None,
        max_results_per_category: int = 5,
        include_text: bool = True,
        include_images: bool = True,
        include_videos: bool = True,
        include_news: bool = True,
    ) -> SearchResponse:
        """Perform a comprehensive search across all categories.

        Args:
            query: Search keywords
            region: Geographic region for search results
            safe_search: Safe search filtering level
            time_limit: Time limit for results
            max_results_per_category: Maximum results per search category
            include_text: Whether to include text search results
            include_images: Whether to include image search results
            include_videos: Whether to include video search results
            include_news: Whether to include news search results

        Returns:
            Comprehensive search response with all result types
        """
        logger.info(f"Performing comprehensive search: '{query}'")

        text_results: list[TextSearchResult] = []
        image_results: list[ImageSearchResult] = []
        video_results: list[VideoSearchResult] = []
        news_results: list[NewsSearchResult] = []

        # Perform text search
        if include_text:
            try:
                text_results = self.search_text(
                    query=query,
                    region=region,
                    safe_search=safe_search,
                    time_limit=time_limit,
                    max_results=max_results_per_category,
                )
            except Exception as e:
                logger.warning(
                    f"Text search failed in comprehensive search: {e}")

        # Perform image search
        if include_images:
            try:
                image_results = self.search_images(
                    query=query,
                    region=region,
                    safe_search=safe_search,
                    time_limit=time_limit,
                    max_results=max_results_per_category,
                )
            except Exception as e:
                logger.warning(
                    f"Image search failed in comprehensive search: {e}")

        # Perform video search
        if include_videos:
            try:
                video_results = self.search_videos(
                    query=query,
                    region=region,
                    safe_search=safe_search,
                    time_limit=time_limit,
                    max_results=max_results_per_category,
                )
            except Exception as e:
                logger.warning(
                    f"Video search failed in comprehensive search: {e}")

        # Perform news search
        if include_news:
            try:
                news_results = self.search_news(
                    query=query,
                    region=region,
                    safe_search=safe_search,
                    time_limit=time_limit,
                    max_results=max_results_per_category,
                )
            except Exception as e:
                logger.warning(
                    f"News search failed in comprehensive search: {e}")

        total_results = len(text_results) + len(image_results) + \
            len(video_results) + len(news_results)

        response = SearchResponse(
            query=query,
            region=region.value,
            safe_search=safe_search.value,
            text_results=text_results,
            image_results=image_results,
            video_results=video_results,
            news_results=news_results,
            total_results=total_results,
        )

        logger.info(
            f"Comprehensive search completed: {total_results} total results")
        return response

    def search_with_filters(
        self,
        query: str,
        search_type: str = "text",
        region: DDGRegion = DDGRegion.no_region,
        safe_search: DDGSafeSearch = DDGSafeSearch.moderate,
        time_limit: DDGTimeLimit | None = None,
        max_results: int = 10,
        **filters: Any,
    ) -> list[TextSearchResult] | list[ImageSearchResult] | list[VideoSearchResult] | list[NewsSearchResult]:
        """Perform a search with specific filters based on search type.

        Args:
            query: Search keywords
            search_type: Type of search ('text', 'images', 'videos', 'news')
            region: Geographic region for search results
            safe_search: Safe search filtering level
            time_limit: Time limit for results
            max_results: Maximum number of results to return
            **filters: Additional filters specific to search type

        Returns:
            List of search results based on search type
        """
        if search_type == "text":
            return self.search_text(
                query=query,
                region=region,
                safe_search=safe_search,
                time_limit=time_limit,
                max_results=max_results,
            )
        elif search_type == "images":
            return self.search_images(
                query=query,
                region=region,
                safe_search=safe_search,
                time_limit=time_limit,
                max_results=max_results,
                **filters,
            )
        elif search_type == "videos":
            return self.search_videos(
                query=query,
                region=region,
                safe_search=safe_search,
                time_limit=time_limit,
                max_results=max_results,
                **filters,
            )
        elif search_type == "news":
            return self.search_news(
                query=query,
                region=region,
                safe_search=safe_search,
                time_limit=time_limit,
                max_results=max_results,
            )
        else:
            raise ValueError(f"Unsupported search type: {search_type}")


# Convenience function for creating a search tool instance
def create_duckduckgo_search_tool(
    proxy: str | None = None,
    timeout: int = 10,
    verify: bool = True,
) -> DuckDuckGoSearchTool:
    """Create a DuckDuckGo search tool instance.

    Args:
        proxy: Proxy server URL (supports http/https/socks5)
        timeout: Request timeout in seconds
        verify: SSL verification for requests

    Returns:
        Configured DuckDuckGoSearchTool instance
    """
    return DuckDuckGoSearchTool(proxy=proxy, timeout=timeout, verify=verify)
