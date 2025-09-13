"""Test YouTube integration without API quota requirements."""

import asyncio
import sys
import os

# Add the libs/common directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'libs', 'common'))

from utils.youtube_tool import YouTubeReporterTool, YouTubeToolParams
from core.config_service import ConfigService


async def test_transcript_only_workflow():
    """Demonstrate how to use YouTube integration without API quota."""
    print("🎯 YouTube Integration - Transcript-Only Mode (No API Quota)")
    print("=" * 65)
    
    config_service = ConfigService()
    youtube_tool = YouTubeReporterTool(config_service)
    
    # Example: Recent tech videos (you would get these IDs from other sources)
    tech_video_ids = [
        "dQw4w9WgXcQ",  # Example video 1
        "jNQXAC9IVRw",  # Example video 2
        "9bZkp7q19f0",  # Example video 3
    ]
    
    print(f"📝 Extracting transcripts from {len(tech_video_ids)} videos...")
    print("💡 This approach works without YouTube API quota limits!")
    
    params = YouTubeToolParams(
        operation="transcribe",
        specific_video_ids=tech_video_ids
    )
    
    result = await youtube_tool.execute(params)
    
    if result.success:
        print(f"\n✅ Success! Transcribed {result.transcripts_obtained} videos")
        print(f"🔗 Created {len(result.sources)} sources for story tracking")
        
        # Show how this integrates with news generation
        print(f"\n📰 Integration with News Generation:")
        print(f"   • Sources available for story attribution")
        print(f"   • Transcript content available for analysis")
        print(f"   • No API quota consumed")
        
        if result.detailed_results:
            total_words = sum(detail['word_count'] for detail in result.detailed_results)
            print(f"   • Total content: {total_words} words across all videos")
    else:
        print(f"❌ Failed: {result.error}")


if __name__ == "__main__":
    print("🚀 Testing YouTube Integration - No API Quota Required")
    print()
    
    asyncio.run(test_transcript_only_workflow())
    
    print("\n" + "=" * 65)
    print("💡 Summary:")
    print("   • Transcript extraction: ✅ Works without API quota")
    print("   • Video discovery: ❌ Requires API quota")
    print("   • Channel browsing: ❌ Requires API quota")
    print()
    print("🎯 For full functionality, you need a YouTube API key")
    print("🎯 For transcript-only mode, no API key needed!")
