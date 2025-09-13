"""Test YouTube integration with reporter agent."""

import asyncio
import sys
import os

# Add the libs/common directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'libs', 'common'))

from agents.agent_factory import AgentFactory
from agents.models.task_models import ReporterTask
from agents.task_execution_service import TaskExecutionService
from agents.types import ReporterField, TaskType
from core.config_service import ConfigService


async def test_youtube_in_reporter():
    """Test YouTube tool integration with reporter agent."""
    print("🤖 Initializing reporter agent with YouTube integration...")
    
    config_service = ConfigService()
    
    # Create agent factory
    factory = AgentFactory(config_service)

    # Create a technology reporter
    reporter = factory.create_reporter(ReporterField.TECHNOLOGY)

    # Create a task that should use YouTube data
    task = ReporterTask(
        name=TaskType.WRITE_STORY,
        field=ReporterField.TECHNOLOGY,
        description="Write a story about recent developments in AI and machine learning",
        guidelines="Use YouTube videos and transcripts to find the latest discussions and announcements from tech channels"
    )

    print("🤖 Starting reporter task with YouTube integration...")
    print(f"📋 Task: {task.description}")
    print(f"🎯 Field: {task.field}")
    
    try:
        # Execute the task using the task execution service
        result = await factory.task_service.execute_reporter_task(task)

        if result.success:
            print("✅ Task completed successfully!")
            print(f"📄 Story Title: {result.story.title}")
            print(f"📝 Story Length: {len(result.story.content)} characters")
            print(f"🔗 Sources: {len(result.story.sources)} sources")

            # Check if YouTube was used
            youtube_sources = [s for s in result.story.sources if 'youtube.com' in s.url]
            if youtube_sources:
                print(f"🎥 YouTube sources used: {len(youtube_sources)}")
                for source in youtube_sources[:2]:  # Show first 2
                    print(f"   - {source.title}: {source.url}")
            else:
                print("⚠️  No YouTube sources were used")
                
            # Show story preview
            print(f"\n📖 Story Preview:")
            print(f"Title: {result.story.title}")
            print(f"Content: {result.story.content[:300]}...")
            
        else:
            print(f"❌ Task failed: {result.error}")
            
    except Exception as e:
        print(f"❌ Error during task execution: {e}")
        import traceback
        traceback.print_exc()


async def test_topic_discovery():
    """Test YouTube topic discovery workflow."""
    print("\n🔍 Testing YouTube Topic Discovery...")
    
    config_service = ConfigService()
    factory = AgentFactory(config_service)

    # Create a technology reporter
    reporter = factory.create_reporter(ReporterField.TECHNOLOGY)

    # Create a task for finding trending topics
    task = ReporterTask(
        name=TaskType.FIND_TRENDING_TOPICS,
        field=ReporterField.TECHNOLOGY,
        description="Find trending topics in technology using YouTube channels",
        guidelines="Use YouTube channels to discover what topics are currently being discussed in the tech community"
    )

    print("🔍 Starting topic discovery task...")
    
    try:
        result = await factory.task_service.execute_reporter_task(task)

        if result.success and result.topics:
            print("✅ Topic discovery completed!")
            print(f"📋 Found {len(result.topics.topics)} trending topics")
            print(f"🎯 Field: {result.topics.field}")
            print(f"💭 Reasoning: {result.topics.reasoning}")
            
            print("\n🎯 Trending Topics:")
            for i, topic in enumerate(result.topics.topics[:5]):  # Show first 5
                print(f"   {i+1}. {topic}")
                
        else:
            print(f"❌ Topic discovery failed: {result.error}")
            
    except Exception as e:
        print(f"❌ Error during topic discovery: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 Starting YouTube Reporter Integration Tests...")
    try:
        asyncio.run(test_youtube_in_reporter())
        asyncio.run(test_topic_discovery())
        print("\n✅ All reporter tests completed!")
    except Exception as e:
        print(f"\n❌ Reporter test failed with error: {e}")
        import traceback
        traceback.print_exc()
