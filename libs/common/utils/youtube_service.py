"""YouTube service for fetching videos and transcripts."""

from datetime import datetime, timedelta
from typing import Any

from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

from core.config_service import ConfigService
from core.logging_service import get_logger
from .youtube_models import (
    YouTubeVideo, VideoTranscript, YouTubeChannelInfo,
    YouTubeSearchParams, YouTubeSearchResult, TranscriptionMethod, YouTubeField
)

logger = get_logger(__name__)


class YouTubeService:
    """Service for interacting with YouTube Data API and transcripts."""

    def __init__(self, config_service: ConfigService):
        """Initialize YouTube service with configuration."""
        self.config_service = config_service
        self.api_key = config_service.get("youtube.api_key")

        if not self.api_key:
            raise ValueError("YouTube API key not found in configuration")

        # Initialize YouTube API client
        self.youtube: Any = build('youtube', 'v3', developerKey=self.api_key)

        logger.info("YouTube service initialized successfully")

    def get_channels_for_field(self, field: YouTubeField) -> list[str]:
        """Get channel IDs for a specific field from environment configuration.

        Supports both YouTube URLs and channel IDs. URLs are automatically converted to channel IDs.
        """
        env_key = f"youtube_channels_{field.value}"
        channels_str = self.config_service.get(env_key, "")

        if not channels_str:
            logger.warning(f"No channels configured for field: {field}")
            return []

        # Split comma-separated entries and clean whitespace
        raw_channels = [channel.strip() for channel in channels_str.split(",") if channel.strip()]

        # Convert URLs to channel IDs
        channel_ids = []
        for channel in raw_channels:
            channel_id = self._extract_channel_id(channel)
            if channel_id:
                channel_ids.append(channel_id)
            else:
                logger.warning(f"Could not extract channel ID from: {channel}")

        logger.info(f"Found {len(channel_ids)} valid channels for field {field}")
        return channel_ids

    def _extract_channel_id(self, channel_input: str) -> str | None:
        """Extract channel ID from URL or return as-is if already a channel ID."""
        import re

        # If it's already a channel ID (starts with UC and is 24 chars), return as-is
        # Format: UCxxxxxxxxxxxxxxxxxxxxx (UC + 22 random characters)
        if channel_input.startswith("UC") and len(channel_input) == 24:
            return channel_input

        # Extract from various YouTube URL formats
        url_patterns = [
            r"youtube\.com/channel/([a-zA-Z0-9_-]{24})",  # /channel/UCxxxxx
            r"youtube\.com/@([a-zA-Z0-9_.-]+)",           # /@username
            r"youtube\.com/c/([a-zA-Z0-9_.-]+)",          # /c/username
            r"youtube\.com/user/([a-zA-Z0-9_.-]+)",       # /user/username
        ]

        for pattern in url_patterns:
            match = re.search(pattern, channel_input)
            if match:
                extracted = match.group(1)
                # If it's a direct channel ID, return it
                if extracted.startswith("UC") and len(extracted) == 24:
                    return extracted
                # Otherwise, it's a username that needs to be resolved via API
                return self._resolve_username_to_channel_id(extracted)

        logger.warning(f"Could not parse channel input: {channel_input}")
        return None

    def _resolve_username_to_channel_id(self, username: str) -> str | None:
        """Resolve a username to channel ID using YouTube API."""
        try:
            # Try to search for the channel by username
            response: Any = self.youtube.search().list(
                part='snippet',
                q=username,
                type='channel',
                maxResults=1
            ).execute()

            if response['items']:
                return response['items'][0]['snippet']['channelId']

            logger.warning(f"Could not resolve username to channel ID: {username}")
            return None

        except Exception as e:
            logger.error(f"Error resolving username {username}: {e}")
            return None

    async def search_channel_videos(self, params: YouTubeSearchParams) -> YouTubeSearchResult:
        """Search for recent videos from specified channels."""
        try:
            all_videos = []
            all_transcripts = {}
            channel_info = {}

            # Calculate date threshold
            since_date = datetime.now() - timedelta(days=params.days_back)

            for channel_id in params.channel_ids:
                logger.info(f"Fetching videos from channel: {channel_id}")

                # Get channel info
                channel_data = await self._get_channel_info(channel_id)
                if channel_data:
                    channel_info[channel_id] = channel_data

                # Get recent videos
                videos = await self._get_channel_videos(
                    channel_id,
                    params.max_videos_per_channel,
                    since_date
                )

                all_videos.extend(videos)

                # Get transcripts if requested
                if params.include_transcripts:
                    for video in videos:
                        transcript = await self._get_video_transcript(
                            video.video_id,
                            params.transcription_method
                        )
                        if transcript:
                            all_transcripts[video.video_id] = transcript

            return YouTubeSearchResult(
                success=True,
                videos=all_videos,
                transcripts=all_transcripts,
                channels=channel_info,
                total_videos=len(all_videos),
                error=None
            )

        except Exception as e:
            logger.error(f"YouTube search failed: {e}")
            return YouTubeSearchResult(
                success=False,
                videos=[],
                transcripts={},
                channels={},
                total_videos=0,
                error=str(e)
            )

    async def _get_channel_info(self, channel_id: str) -> YouTubeChannelInfo | None:
        """Get information about a YouTube channel."""
        try:
            response: Any = self.youtube.channels().list(
                part='snippet,statistics',
                id=channel_id
            ).execute()

            if not response['items']:
                logger.warning(f"Channel not found: {channel_id}")
                return None

            item: Any = response['items'][0]
            snippet: Any = item['snippet']
            stats: Any = item.get('statistics', {})

            return YouTubeChannelInfo(
                channel_id=channel_id,
                title=snippet['title'],
                description=snippet.get('description', ''),
                subscriber_count=int(stats.get('subscriberCount', 0)),
                video_count=int(stats.get('videoCount', 0)),
                url=f"https://www.youtube.com/channel/{channel_id}"
            )

        except Exception as e:
            logger.error(f"Failed to get channel info for {channel_id}: {e}")
            return None

    async def _get_channel_videos(
        self,
        channel_id: str,
        max_results: int,
        since_date: datetime
    ) -> list[YouTubeVideo]:
        """Get recent videos from a channel."""
        try:
            # Search for videos from the channel
            search_response: Any = self.youtube.search().list(
                part='id,snippet',
                channelId=channel_id,
                type='video',
                order='date',
                maxResults=max_results,
                publishedAfter=since_date.isoformat() + 'Z'
            ).execute()

            video_ids: list[str] = [item['id']['videoId'] for item in search_response['items']]

            if not video_ids:
                return []

            # Get detailed video information
            videos_response: Any = self.youtube.videos().list(
                part='snippet,statistics,contentDetails',
                id=','.join(video_ids)
            ).execute()

            videos: list[YouTubeVideo] = []
            for item in videos_response['items']:
                video = self._parse_video_item(item)
                if video:
                    videos.append(video)

            return videos

        except Exception as e:
            logger.error(f"Failed to get videos for channel {channel_id}: {e}")
            return []

    def _parse_video_item(self, item: dict[str, Any]) -> YouTubeVideo | None:
        """Parse a video item from YouTube API response."""
        try:
            snippet: Any = item['snippet']
            stats: Any = item.get('statistics', {})
            content_details: Any = item.get('contentDetails', {})

            return YouTubeVideo(
                video_id=item['id'],
                title=snippet['title'],
                description=snippet.get('description', ''),
                channel_id=snippet['channelId'],
                channel_title=snippet['channelTitle'],
                published_at=datetime.fromisoformat(snippet['publishedAt'].replace('Z', '+00:00')),
                duration=content_details.get('duration', 'PT0S'),
                view_count=int(stats.get('viewCount', 0)),
                like_count=int(stats.get('likeCount', 0)),
                url=f"https://www.youtube.com/watch?v={item['id']}",
                thumbnail_url=snippet.get('thumbnails', {}).get('high', {}).get('url')
            )

        except Exception as e:
            logger.error(f"Failed to parse video item: {e}")
            return None

    async def _get_video_transcript(
        self,
        video_id: str,
        method: TranscriptionMethod
    ) -> VideoTranscript | None:
        """Get transcript for a video using YouTube's auto-generated captions."""
        try:
            # Only support YouTube's built-in transcription
            if method == TranscriptionMethod.YOUTUBE_AUTO:
                return await self._get_youtube_transcript(video_id)
            else:
                logger.warning(f"Unsupported transcription method: {method}. Only YouTube auto-captions supported.")
                return None

        except Exception as e:
            logger.error(f"Failed to get transcript for {video_id}: {e}")
            return None

    async def _get_youtube_transcript(self, video_id: str) -> VideoTranscript | None:
        """Get transcript using YouTube's auto-generated captions."""
        try:
            # Use the static method to get transcript
            transcript_list: Any = YouTubeTranscriptApi.get_transcript(video_id)  # type: ignore

            # Combine all transcript segments
            full_transcript = ' '.join([segment['text'] for segment in transcript_list])  # type: ignore

            return VideoTranscript(
                video_id=video_id,
                transcript=full_transcript,
                language='en',
                method=TranscriptionMethod.YOUTUBE_AUTO,
                word_count=len(full_transcript.split()),
                confidence=None  # Confidence not available from YouTube transcript API
            )

        except (TranscriptsDisabled, NoTranscriptFound) as e:
            logger.warning(f"No transcript available for {video_id}: {e}")
            return None
