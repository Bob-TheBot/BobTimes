"""YouTube tool for reporter agents."""

from datetime import datetime
from typing import Any

from agents.models.story_models import StorySource
from agents.tools.base_tool import BaseTool, UnifiedToolResult
from core.config_service import ConfigService
from core.llm_service import ModelSpeed
from core.logging_service import get_logger
from pydantic import BaseModel, Field, PrivateAttr

from .youtube_models import YouTubeSearchParams, TranscriptionMethod, YouTubeField
from .youtube_service import YouTubeService

# Import subsection types for type hints
from agents.types import TechnologySubSection, EconomicsSubSection, ScienceSubSection

logger = get_logger(__name__)


def _normalize_topic_name(topic: str) -> str:
    """Normalize topic name for consistent storage and retrieval."""
    return topic.strip().title()


class YouTubeToolParams(BaseModel):
    """Parameters for YouTube tool operations."""
    channel_ids: list[str] = Field(
        default_factory=list,
        description="User-provided list of YouTube channel URLs or IDs (optional if field is specified)"
    )
    field: YouTubeField | None = Field(
        default=None,
        description="Field name to get channels from env config"
    )
    subsection: TechnologySubSection | EconomicsSubSection | ScienceSubSection | None = Field(
        default=None,
        description="Optional subsection within the field for more targeted channel selection"
    )
    max_videos_per_channel: int = Field(default=3, description="Max videos per channel")
    days_back: int = Field(default=7, description="Days to look back for videos")
    operation: str = Field(default="topics", description="Operation: 'topics' or 'transcribe'")
    specific_video_ids: list[str] = Field(
        default_factory=list,
        description="Specific video IDs to transcribe (for 'transcribe' operation)"
    )


class YouTubeToolResult(BaseModel):
    """Result from YouTube tool execution."""
    success: bool = Field(description="Whether the operation was successful")
    operation: str = Field(description="Operation performed: 'topics' or 'transcribe'")
    videos_found: int = Field(default=0, description="Number of videos found")
    channels_searched: int = Field(default=0, description="Number of channels searched")
    transcripts_obtained: int = Field(default=0, description="Number of transcripts obtained")
    topics_extracted: list[str] = Field(default_factory=list, description="Video titles as potential topics")
    sources: list[StorySource] = Field(default_factory=lambda: [], description="YouTube video sources for story tracking")
    video_metadata: list[dict[str, Any]] = Field( # type: ignore
        default_factory=list,
        description="Video metadata including video_id, channel_id, title for later transcript retrieval"
    )
    topic_source_mapping: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Maps topic names to their source data (videos, metadata, etc.)"
    )
    summary: str = Field(description="Summary of findings")
    detailed_results: list[dict[str, Any]] = Field( # type: ignore
        default_factory=list,
        description="Detailed video information with optional transcripts"
    )
    error: str | None = Field(None, description="Error message if failed")


class YouTubeReporterTool(BaseTool):
    """YouTube data source tool for reporter agents."""

    # Private attributes for internal state
    _config_service: ConfigService = PrivateAttr()
    _youtube_service: Any = PrivateAttr(default=None)

    def __init__(self, config_service: ConfigService | None = None):
        """Initialize YouTube tool."""
        name = "youtube_search"
        description = f"""
Extract topics from YouTube channels or transcribe specific videos for content generation.

PARAMETER SCHEMA:
{YouTubeToolParams.model_json_schema()}

CORRECT USAGE EXAMPLES:
# Extract topics from specific channels (URLs or IDs)
{{"channel_ids": ["https://www.youtube.com/@GoogleDevelopers", "https://www.youtube.com/@LinusTechTips"], "operation": "topics", "days_back": 7}}

# Extract topics from field-configured channels
{{"field": "technology", "operation": "topics", "days_back": 7}}

# Extract topics from subsection-specific channels (preferred for targeted content)
{{"field": "technology", "subsection": "ai_tools", "operation": "topics", "days_back": 7}}

# Transcribe specific videos for content
{{"operation": "transcribe", "specific_video_ids": ["dQw4w9WgXcQ", "oHg5SJYRHA0"]}}

# Get recent videos from field and transcribe them
{{"field": "science", "operation": "transcribe", "max_videos_per_channel": 2}}

# Get videos from specific subsection and transcribe
{{"field": "economics", "subsection": "crypto", "operation": "transcribe", "max_videos_per_channel": 2}}

# Mix of direct channels and field channels
{{"channel_ids": ["https://www.youtube.com/@3Blue1Brown"], "field": "science", "operation": "topics"}}

USAGE GUIDELINES:
- ALWAYS provide channel_ids list (user-specified channels) OR field parameter
- Use subsection parameter for more targeted channel selection within a field
- Use operation="topics" to extract video titles for topic discovery
- Use operation="transcribe" to get full transcripts for content generation
- Set days_back to control how recent the videos should be (1-30 days)
- Use specific_video_ids for targeted transcription of known videos
- Max 5 videos per channel recommended for performance
- Subsection-specific channels take priority over field-level channels

RETURNS:
- operation: The operation performed ("topics" or "transcribe")
- topics_extracted: List of video titles (for topic discovery)
- detailed_results: Full video data with optional transcripts
- summary: Brief summary of findings

This tool supports two workflows:
1. Topic Discovery: Extract recent video titles from user-specified channels
2. Content Generation: Transcribe videos for detailed content analysis
"""
        super().__init__(name=name, description=description)
        self.params_model = YouTubeToolParams

        # Set private attributes
        self._config_service = config_service or ConfigService()

        # Initialize YouTube service
        try:
            self._youtube_service = YouTubeService(self._config_service)
        except Exception as e:
            logger.error(f"Failed to initialize YouTube service: {e}")
            self._youtube_service = None

    async def execute(self, params: BaseModel, model_speed: ModelSpeed = ModelSpeed.FAST) -> UnifiedToolResult:
        """Execute YouTube search with structured params."""
        if not isinstance(params, YouTubeToolParams):
            return UnifiedToolResult(
                success=False,
                operation="youtube",
                query=None,
                sources=[],
                topics_extracted=[],
                topic_source_mapping={},
                metadata={},
                summary="Invalid parameters",
                error=f"Expected YouTubeToolParams, got {type(params)}"
            )

        if not self._youtube_service:
            return UnifiedToolResult(
                success=False,
                operation="youtube",
                query=None,
                sources=[],
                topics_extracted=[],
                topic_source_mapping={},
                metadata={"operation": params.operation},
                summary="YouTube service not available",
                error="YouTube service not available - check API key configuration"
            )

        try:
            # Determine channel IDs to use
            channel_ids = params.channel_ids.copy() if params.channel_ids else []

            # If field is specified, get channels from environment configuration
            if params.field is not None:
                field_channels = self._youtube_service.get_channels_for_field(
                    params.field,
                    params.subsection
                )
                channel_ids.extend(field_channels)

            # Convert any URLs to channel IDs
            resolved_channel_ids = []
            for channel in channel_ids:
                channel_id = self._youtube_service._extract_channel_id(channel)
                if channel_id:
                    resolved_channel_ids.append(channel_id)
                else:
                    logger.warning(f"Could not resolve channel: {channel}")

            # Validate we have channels to work with
            if not resolved_channel_ids and params.operation == "topics":
                return UnifiedToolResult(
                    success=False,
                    operation="youtube",
                    query=None,
                    sources=[],
                    topics_extracted=[],
                    topic_source_mapping={},
                    metadata={"operation": params.operation},
                    summary="No valid channel IDs found",
                    error="No valid channel IDs found. Please specify either channel_ids parameter or field parameter."
                )

            # Update params with resolved channel IDs
            params.channel_ids = resolved_channel_ids

            # Handle different operations
            if params.operation == "topics":
                return await self._extract_topics(params)
            else:
                return UnifiedToolResult(
                    success=False,
                    operation="youtube",
                    query=None,
                    sources=[],
                    topics_extracted=[],
                    topic_source_mapping={},
                    metadata={"operation": params.operation},
                    summary="Unknown operation",
                    error=f"Unknown operation: {params.operation}. Only 'topics' operation is supported since transcripts are now embedded in video objects."
                )

        except Exception as e:
            logger.error(f"YouTube tool execution failed: {e}")
            return UnifiedToolResult(
                success=False,
                operation="youtube",
                query=None,
                sources=[],
                topics_extracted=[],
                topic_source_mapping={},
                metadata={"operation": params.operation},
                summary="Execution failed",
                error=str(e)
            )

    async def _extract_topics(self, params: YouTubeToolParams) -> UnifiedToolResult:
        """Extract video titles as potential topics from user-specified channels."""
        try:
            # Create search parameters for topic extraction WITH transcripts
            # We need transcripts to create useful sources for story writing
            search_params = YouTubeSearchParams(
                channel_ids=params.channel_ids,
                max_videos_per_channel=params.max_videos_per_channel,
                days_back=params.days_back,
                extract_topics_only=False,  # We need full video data including transcripts
                include_transcripts=True,   # REQUIRED: Only videos with transcripts are useful
                transcription_method=TranscriptionMethod.YOUTUBE_AUTO
            )

            # Execute search
            logger.info(f"Extracting topics from {len(params.channel_ids)} channels...")
            result = await self._youtube_service.search_channel_videos(search_params)

            if not result.success:
                return UnifiedToolResult(
                    success=False,
                    operation="youtube",
                    query=None,
                    sources=[],
                    topics_extracted=[],
                    topic_source_mapping={},
                    metadata={"operation": "topics"},
                    summary="Failed to extract topics",
                    error=result.error or "Failed to extract topics"
                )

            # Filter videos to only include those with transcripts
            # Transcripts are now embedded directly in video objects
            videos_with_transcripts = [video for video in result.videos if video.transcript is not None]

            # Extract topics only from videos with transcripts - normalize for consistency
            topics = [_normalize_topic_name(video.title) for video in videos_with_transcripts]

            # Create StorySource objects ONLY for videos with transcripts
            sources: list[StorySource] = []
            for video in videos_with_transcripts:
                # Get transcript from the video object
                transcript_text = video.transcript.transcript if video.transcript else ""

                source = StorySource(
                    url=video.url,
                    title=video.title,
                    summary=f"YouTube video from {video.channel_title} with transcript: {video.description[:200]}..." if video.description else f"YouTube video from {video.channel_title} with transcript",
                    content=transcript_text,  # Full transcript content for story writing
                    source_type="youtube",
                    accessed_at=datetime.now()
                )
                sources.append(source)

            logger.info(f"Filtered to {len(videos_with_transcripts)} videos with transcripts out of {len(result.videos)} total videos")

            # Create summary
            summary = f"Extracted {len(topics)} potential topics from {len(videos_with_transcripts)} videos with transcripts (out of {len(result.videos)} total videos)"

            # Create video metadata and topic-source mapping ONLY for videos with transcripts
            video_metadata = []
            topic_source_mapping = {}

            for video in videos_with_transcripts:
                metadata = {
                    "video_id": video.video_id,
                    "channel_id": video.channel_id,
                    "channel_title": video.channel_title,
                    "title": video.title,
                    "url": video.url,
                    "published_at": video.published_at.isoformat(),
                    "description": video.description[:500] if video.description else ""
                }
                video_metadata.append(metadata)

                # Map topic (video title) to its source data - normalize for consistency
                topic_name = _normalize_topic_name(video.title)
                topic_source_mapping[topic_name] = {
                    "video_metadata": metadata,
                    "source": next((s for s in sources if s.url == video.url), None),
                    "channel_info": {
                        "channel_id": video.channel_id,
                        "channel_title": video.channel_title
                    }
                }

            return UnifiedToolResult(
                success=True,
                operation="youtube",
                query=f"channels: {', '.join(params.channel_ids)}",
                sources=sources,
                topics_extracted=topics,
                topic_source_mapping=topic_source_mapping,
                metadata={
                    "operation": "topics",
                    "videos_found": len(videos_with_transcripts),
                    "channels_searched": len(params.channel_ids),
                    "video_metadata": video_metadata,
                    "detailed_results": [{
                        "title": video.title,
                        "channel": video.channel_title,
                        "published": video.published_at.isoformat(),
                        "url": video.url,
                        "views": video.view_count,
                        "description": video.description[:200] + "..." if len(video.description) > 200 else video.description,
                        "has_transcript": True,  # All included videos have transcripts
                        "transcript_word_count": video.transcript.word_count if video.transcript else 0,
                        "transcript_language": video.transcript.language if video.transcript else "unknown"
                    } for video in videos_with_transcripts]
                },
                summary=summary,
                error=None
            )

        except Exception as e:
            logger.error(f"Topic extraction failed: {e}")
            return UnifiedToolResult(
                success=False,
                operation="youtube",
                query=None,
                sources=[],
                topics_extracted=[],
                topic_source_mapping={},
                metadata={"operation": "topics"},
                summary="Topic extraction failed",
                error=str(e)
            )


