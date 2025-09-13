"""YouTube tool for reporter agents."""

from datetime import datetime
from typing import Any

from agents.models.story_models import StorySource
from agents.tools.base_tool import BaseTool
from core.config_service import ConfigService
from core.llm_service import ModelSpeed
from core.logging_service import get_logger
from pydantic import BaseModel, Field, PrivateAttr

from .youtube_models import YouTubeSearchParams, TranscriptionMethod, YouTubeField
from .youtube_service import YouTubeService

logger = get_logger(__name__)


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
    summary: str = Field(description="Summary of findings")
    detailed_results: list[dict[str, Any]] = Field(
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

# Transcribe specific videos for content
{{"operation": "transcribe", "specific_video_ids": ["dQw4w9WgXcQ", "oHg5SJYRHA0"]}}

# Get recent videos from field and transcribe them
{{"field": "science", "operation": "transcribe", "max_videos_per_channel": 2}}

# Mix of direct channels and field channels
{{"channel_ids": ["https://www.youtube.com/@3Blue1Brown"], "field": "science", "operation": "topics"}}

USAGE GUIDELINES:
- ALWAYS provide channel_ids list (user-specified channels)
- Use operation="topics" to extract video titles for topic discovery
- Use operation="transcribe" to get full transcripts for content generation
- Set days_back to control how recent the videos should be (1-30 days)
- Use specific_video_ids for targeted transcription of known videos
- Max 5 videos per channel recommended for performance

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

    async def execute(self, params: BaseModel, model_speed: ModelSpeed = ModelSpeed.FAST) -> YouTubeToolResult:
        """Execute YouTube search with structured params."""
        if not isinstance(params, YouTubeToolParams):
            return YouTubeToolResult(
                success=False,
                operation="unknown",
                error=f"Expected YouTubeToolParams, got {type(params)}",
                summary="Invalid parameters"
            )

        if not self._youtube_service:
            return YouTubeToolResult(
                success=False,
                operation=params.operation,
                error="YouTube service not available - check API key configuration",
                summary="YouTube service not available"
            )

        try:
            # Determine channel IDs to use
            channel_ids = params.channel_ids.copy() if params.channel_ids else []

            # If field is specified, get channels from environment configuration
            if params.field is not None:
                field_channels = self._youtube_service.get_channels_for_field(params.field)
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
                return YouTubeToolResult(
                    success=False,
                    operation=params.operation,
                    error="No valid channel IDs found. Please specify either channel_ids parameter or field parameter.",
                    summary="No valid channel IDs found"
                )

            # Update params with resolved channel IDs
            params.channel_ids = resolved_channel_ids

            # Handle different operations
            if params.operation == "topics":
                return await self._extract_topics(params)
            elif params.operation == "transcribe":
                return await self._transcribe_videos(params)
            else:
                return YouTubeToolResult(
                    success=False,
                    operation=params.operation,
                    error=f"Unknown operation: {params.operation}. Use 'topics' or 'transcribe'.",
                    summary="Unknown operation"
                )

        except Exception as e:
            logger.error(f"YouTube tool execution failed: {e}")
            return YouTubeToolResult(
                success=False,
                operation=params.operation,
                error=str(e),
                summary="Execution failed"
            )

    async def _extract_topics(self, params: YouTubeToolParams) -> YouTubeToolResult:
        """Extract video titles as potential topics from user-specified channels."""
        try:
            # Create search parameters for topic extraction
            search_params = YouTubeSearchParams(
                channel_ids=params.channel_ids,
                max_videos_per_channel=params.max_videos_per_channel,
                days_back=params.days_back,
                extract_topics_only=True,
                include_transcripts=False,  # Don't need transcripts for topic extraction
                transcription_method=TranscriptionMethod.YOUTUBE_AUTO
            )

            # Execute search
            logger.info(f"Extracting topics from {len(params.channel_ids)} channels...")
            result = await self._youtube_service.search_channel_videos(search_params)

            if not result.success:
                return YouTubeToolResult(
                    success=False,
                    operation="topics",
                    error=result.error or "Failed to extract topics",
                    summary="Failed to extract topics"
                )

            # Extract topics (video titles)
            topics = [video.title for video in result.videos]

            # Create StorySource objects for source tracking
            sources = []
            for video in result.videos:
                source = StorySource(
                    url=video.url,
                    title=video.title,
                    summary=f"YouTube video from {video.channel_title}: {video.description[:200]}..." if video.description else f"YouTube video from {video.channel_title}",
                    accessed_at=datetime.now()
                )
                sources.append(source)

            # Create summary
            summary = f"Extracted {len(topics)} potential topics from {len(params.channel_ids)} channels"

            return YouTubeToolResult(
                success=True,
                operation="topics",
                videos_found=len(result.videos),
                channels_searched=len(params.channel_ids),
                topics_extracted=topics,
                sources=sources,
                summary=summary,
                error=None,
                detailed_results=[{
                    "title": video.title,
                    "channel": video.channel_title,
                    "published": video.published_at.isoformat(),
                    "url": video.url,
                    "views": video.view_count,
                    "description": video.description[:200] + "..." if len(video.description) > 200 else video.description
                } for video in result.videos]
            )

        except Exception as e:
            logger.error(f"Topic extraction failed: {e}")
            return YouTubeToolResult(
                success=False,
                operation="topics",
                error=str(e),
                summary="Topic extraction failed"
            )

    async def _transcribe_videos(self, params: YouTubeToolParams) -> YouTubeToolResult:
        """Transcribe specific videos or recent videos from channels for content generation."""
        try:
            videos_to_transcribe = []

            # If specific video IDs provided, use those
            if params.specific_video_ids:
                # Create dummy video objects for specific IDs
                for video_id in params.specific_video_ids:
                    videos_to_transcribe.append(video_id)
            else:
                # Get recent videos from channels first
                search_params = YouTubeSearchParams(
                    channel_ids=params.channel_ids,
                    max_videos_per_channel=params.max_videos_per_channel,
                    days_back=params.days_back,
                    extract_topics_only=False,
                    include_transcripts=False,  # We'll get transcripts separately
                    transcription_method=TranscriptionMethod.YOUTUBE_AUTO
                )

                result = await self._youtube_service.search_channel_videos(search_params)
                if not result.success:
                    return YouTubeToolResult(
                        success=False,
                        operation="transcribe",
                        error=result.error or "Failed to find videos to transcribe",
                        summary="Failed to find videos to transcribe"
                    )

                videos_to_transcribe = [video.video_id for video in result.videos]

            # Now get transcripts for the videos
            transcripts = {}
            detailed_results = []
            sources = []

            for video_id in videos_to_transcribe:
                transcript = await self._youtube_service._get_video_transcript(
                    video_id,
                    TranscriptionMethod.YOUTUBE_AUTO
                )

                video_url = f"https://www.youtube.com/watch?v={video_id}"

                if transcript:
                    transcripts[video_id] = transcript
                    detailed_results.append({
                        "video_id": video_id,
                        "url": video_url,
                        "transcript": transcript.transcript,
                        "word_count": transcript.word_count,
                        "language": transcript.language
                    })

                    # Create StorySource for transcribed video
                    source = StorySource(
                        url=video_url,
                        title=f"YouTube Video {video_id}",  # We might not have title for specific video IDs
                        summary=f"YouTube video transcript ({transcript.word_count} words): {transcript.transcript[:200]}..." if len(transcript.transcript) > 200 else transcript.transcript,
                        accessed_at=datetime.now()
                    )
                    sources.append(source)

            summary = f"Transcribed {len(transcripts)} videos out of {len(videos_to_transcribe)} requested"

            return YouTubeToolResult(
                success=True,
                operation="transcribe",
                videos_found=len(videos_to_transcribe),
                transcripts_obtained=len(transcripts),
                sources=sources,
                summary=summary,
                error=None,
                detailed_results=detailed_results
            )

        except Exception as e:
            logger.error(f"Video transcription failed: {e}")
            return YouTubeToolResult(
                success=False,
                operation="transcribe",
                error=str(e),
                summary="Video transcription failed"
            )
