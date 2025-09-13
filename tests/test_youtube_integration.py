"""Test script for YouTube integration."""

import asyncio
import sys
import os

# Add the libs/common directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'libs', 'common'))

from core.config_service import ConfigService
from utils.youtube_tool import YouTubeReporterTool, YouTubeToolParams
from utils.youtube_models import YouTubeField


async def test_youtube_tool():
    """Test the YouTube tool with different operations."""
    print("🔧 Initializing YouTube tool...")
    config_service = ConfigService()
    youtube_tool = YouTubeReporterTool(config_service)

    # Test 1: Extract topics from field-configured channels
    print("\n🔍 Testing Topic Extraction...")
    topic_params = YouTubeToolParams(
        field=YouTubeField.TECHNOLOGY,
        operation="topics",
        days_back=7,
        max_videos_per_channel=3
    )

    result = await youtube_tool.execute(topic_params)

    if result.success:
        print(f"✅ Operation: {result.operation}")
        print(f"✅ Found {result.videos_found} videos from {result.channels_searched} channels")
        print(f"📋 Topics extracted: {len(result.topics_extracted)}")
        print(f"🔗 Sources created: {len(result.sources)}")
        print(f"📄 Summary: {result.summary}")

        # Show first few topics
        if result.topics_extracted:
            print("\n🎯 Sample Topics:")
            for i, topic in enumerate(result.topics_extracted[:3]):
                print(f"   {i+1}. {topic}")

        # Show source tracking
        if result.sources:
            print("\n📚 Sources for Story Tracking:")
            for i, source in enumerate(result.sources[:2]):
                print(f"   {i+1}. {source.title}")
                print(f"      URL: {source.url}")
                print(f"      Summary: {source.summary[:100]}...")
    else:
        print(f"❌ Topic extraction failed: {result.error}")

    # Test 2: Test with specific channel IDs
    print("\n🎬 Testing with Specific Channel URLs...")
    specific_params = YouTubeToolParams(
        channel_ids=["https://www.youtube.com/@GoogleDevelopers"],
        operation="topics",
        days_back=14,
        max_videos_per_channel=2
    )

    specific_result = await youtube_tool.execute(specific_params)

    if specific_result.success:
        print(f"✅ Operation: {specific_result.operation}")
        print(f"✅ Found {specific_result.videos_found} videos")
        print(f"📄 Summary: {specific_result.summary}")

        # Show topics
        if specific_result.topics_extracted:
            print("\n🎯 Topics from Google Developers:")
            for i, topic in enumerate(specific_result.topics_extracted):
                print(f"   {i+1}. {topic}")
    else:
        print(f"❌ Specific channel test failed: {specific_result.error}")

    # Test 3: Test transcription (with a known video that should have transcripts)
    print("\n📝 Testing Video Transcription...")
    transcribe_params = YouTubeToolParams(
        operation="transcribe",
        specific_video_ids=["dQw4w9WgXcQ"]  # Rick Roll - should have transcripts
    )

    transcribe_result = await youtube_tool.execute(transcribe_params)

    if transcribe_result.success:
        print(f"✅ Operation: {transcribe_result.operation}")
        print(f"✅ Transcripts obtained: {transcribe_result.transcripts_obtained}")
        print(f"📄 Summary: {transcribe_result.summary}")

        # Show transcript preview
        if transcribe_result.detailed_results:
            first_transcript = transcribe_result.detailed_results[0]
            print(f"\n📝 Transcript preview: {first_transcript['transcript'][:200]}...")
            print(f"📊 Word count: {first_transcript['word_count']}")
    else:
        print(f"❌ Transcription failed: {transcribe_result.error}")


async def test_field_configurations():
    """Test all field configurations."""
    print("\n🔧 Testing Field Configurations...")
    config_service = ConfigService()
    youtube_tool = YouTubeReporterTool(config_service)

    fields = [YouTubeField.TECHNOLOGY, YouTubeField.SCIENCE, YouTubeField.ECONOMICS, YouTubeField.SPORTS]

    for field in fields:
        print(f"\n📂 Testing {field.upper()} field...")
        params = YouTubeToolParams(
            field=field,
            operation="topics",
            days_back=7,
            max_videos_per_channel=1
        )

        result = await youtube_tool.execute(params)
        if result.success:
            print(f"   ✅ {result.videos_found} videos found from {result.channels_searched} channels")
        else:
            print(f"   ❌ Failed: {result.error}")


if __name__ == "__main__":
    print("🚀 Starting YouTube Integration Tests...")
    try:
        asyncio.run(test_youtube_tool())
        asyncio.run(test_field_configurations())
        print("\n✅ All tests completed!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
