"""YouTube data models for the BobTimes system."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class YouTubeField(StrEnum):
    """YouTube content fields."""
    TECHNOLOGY = "technology"
    SCIENCE = "science"
    ECONOMICS = "economics"
    SPORTS = "sports"


class VideoQuality(StrEnum):
    """Video quality options."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TranscriptionMethod(StrEnum):
    """Transcription method options."""
    YOUTUBE_AUTO = "youtube_auto"  # Use YouTube's auto-generated captions (primary method)


class YouTubeVideo(BaseModel):
    """Represents a YouTube video with metadata."""
    video_id: str = Field(description="YouTube video ID")
    title: str = Field(description="Video title")
    description: str = Field(description="Video description")
    channel_id: str = Field(description="Channel ID")
    channel_title: str = Field(description="Channel name")
    published_at: datetime = Field(description="Publication date")
    duration: str = Field(description="Video duration (ISO 8601)")
    view_count: int = Field(default=0, description="Number of views")
    like_count: int = Field(default=0, description="Number of likes")
    url: str = Field(description="Full YouTube URL")
    thumbnail_url: str | None = Field(None, description="Video thumbnail URL")
    transcript: "VideoTranscript | None" = Field(None, description="Video transcript if available")


class VideoTranscript(BaseModel):
    """Represents a video transcript."""
    video_id: str = Field(description="YouTube video ID")
    transcript: str = Field(description="Full transcript text")
    language: str = Field(default="en", description="Transcript language")
    method: TranscriptionMethod = Field(description="How transcript was obtained")
    confidence: float | None = Field(None, description="Transcription confidence score")
    word_count: int = Field(description="Number of words in transcript")
    created_at: datetime = Field(default_factory=datetime.now)


class YouTubeChannelInfo(BaseModel):
    """Represents YouTube channel information."""
    channel_id: str = Field(description="YouTube channel ID")
    title: str = Field(description="Channel title")
    description: str = Field(description="Channel description")
    subscriber_count: int = Field(default=0, description="Number of subscribers")
    video_count: int = Field(default=0, description="Total videos")
    url: str = Field(description="Channel URL")


class YouTubeSearchParams(BaseModel):
    """Parameters for YouTube search operations."""
    channel_ids: list[str] = Field(description="User-provided list of channel IDs to search")
    max_videos_per_channel: int = Field(default=5, description="Max videos per channel")
    days_back: int = Field(default=7, description="How many days back to search")
    extract_topics_only: bool = Field(default=False, description="Only extract titles for topics")
    include_transcripts: bool = Field(default=False, description="Fetch transcripts for content generation")
    specific_video_ids: list[str] = Field(default_factory=list, description="Specific videos to transcribe")
    transcription_method: TranscriptionMethod = Field(
        default=TranscriptionMethod.YOUTUBE_AUTO,
        description="Preferred transcription method"
    )


class YouTubeSearchResult(BaseModel):
    """Result from YouTube search operation."""
    success: bool = Field(description="Whether the search was successful")
    videos: list[YouTubeVideo] = Field(default_factory=lambda: [], description="Found videos")
    channels: dict[str, YouTubeChannelInfo] = Field(
        default_factory=dict,
        description="Channel info keyed by channel_id"
    )
    total_videos: int = Field(default=0, description="Total videos found")
    error: str | None = Field(None, description="Error message if failed")
