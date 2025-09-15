#!/usr/bin/env python3
"""News Cycle Runner - Python script interface for triggering news cycle generation.

This script provides a simple interface to run news cycles without needing API calls.
Agents are internal classes that communicate directly with each other.

Usage:
    python run_news_cycle.py --topics technology sports economics
    python run_news_cycle.py --topics all
    python run_news_cycle.py --help
"""

import argparse
import asyncio
import sys
from pathlib import Path

from agents.types import ReporterField, FieldTopicRequest
from app.services.newspaper_service import NewspaperService

# Add the backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.config_service import ConfigService
from core.logging_service import get_logger

logger = get_logger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a news cycle with specified topics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --topics technology sports
  %(prog)s --topics all
  %(prog)s --topics economics politics science
  %(prog)s --list-fields
        """
    )

    parser.add_argument(
        "--topics",
        nargs="+",
        help="Topics to generate stories for. Use 'all' for all available fields."
    )

    parser.add_argument(
        "--list-fields",
        action="store_true",
        help="List all available reporter fields and exit"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def get_available_fields() -> list[ReporterField]:
    """Get all available reporter fields."""
    return [
        ReporterField.TECHNOLOGY,
        ReporterField.ECONOMICS,
        ReporterField.SCIENCE
    ]


def parse_topics(topic_args: list[str]) -> list[ReporterField]:
    """Parse topic arguments into ReporterField enums."""
    if not topic_args:
        return []

    if "all" in topic_args:
        return get_available_fields()

    topics: list[ReporterField] = []
    available_fields = {field.value: field for field in get_available_fields()}

    for topic in topic_args:
        topic_lower = topic.lower()
        if topic_lower in available_fields:
            topics.append(available_fields[topic_lower])
        else:
            logger.warning(f"Unknown topic: {topic}. Available: {list(available_fields.keys())}")

    return topics


async def run_news_cycle(topics: list[ReporterField]) -> bool:
    """Run a news cycle with the specified topics."""
    try:
        logger.info("🚀 Starting news cycle runner")
        logger.info(f"📰 Topics: {[topic.value for topic in topics]}")

        # Create services
        config_service = ConfigService()
        newspaper_service = NewspaperService(config_service)

        # Convert ReporterField list to FieldTopicRequest list
        field_requests = [FieldTopicRequest(field=field) for field in topics]

        # Run the news cycle
        logger.info("🔄 Running news cycle...")
        cycle = await newspaper_service.run_news_cycle(field_requests)

        # Display results
        logger.info("✅ News cycle completed successfully!")
        print("\n" + "=" * 60)
        print("📊 NEWS CYCLE RESULTS")
        print("=" * 60)
        print(f"Cycle ID: {cycle.cycle_id}")
        print(f"Status: {cycle.cycle_status.value}")
        # Guard against None end_time for safety
        if cycle.end_time is not None:
            duration_seconds = (cycle.end_time - cycle.start_time).total_seconds()
        else:
            # Should not happen in success path, but default to 0.0 for type safety
            duration_seconds = 0.0
        print(f"Duration: {duration_seconds:.1f} seconds")
        print(f"Stories Published: {len(cycle.published_stories)}")
        print(f"Total Submissions: {len(cycle.submissions)}")
        print(f"Reporter Tasks: {len(cycle.reporter_tasks)}")

        if cycle.published_stories:
            print("\n📚 PUBLISHED STORIES:")
            for i, story in enumerate(cycle.published_stories, 1):
                print(f"\n{i}. {story.title}")
                print(f"   Field: {story.field.value}")
                print(f"   Reporter: {story.reporter_id}")
                print(f"   Word Count: {story.word_count}")
                if hasattr(story, "summary") and story.summary:
                    print(f"   Summary: {story.summary[:100]}...")

        if cycle.editorial_decisions:
            print("\n📝 EDITORIAL DECISIONS:")
            approved = len([d for d in cycle.editorial_decisions if d.decision.value == "approve"])
            rejected = len([d for d in cycle.editorial_decisions if d.decision.value == "reject"])
            revised = len([d for d in cycle.editorial_decisions if d.decision.value == "revise"])
            print(f"   Approved: {approved}")
            print(f"   Rejected: {rejected}")
            print(f"   Revised: {revised}")

        print("\n" + "=" * 60)
        return True

    except Exception as e:
        logger.error(f"❌ News cycle failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    args = parse_arguments()

    # List available fields if requested
    if args.list_fields:
        print("Available reporter fields:")
        for field in get_available_fields():
            print(f"  - {field.value}")
        return 0

    # Validate topics argument
    if not args.topics:
        print("Error: --topics argument is required")
        print("Use --list-fields to see available topics")
        return 1

    # Parse topics
    topics = parse_topics(args.topics)
    if not topics:
        print("Error: No valid topics specified")
        print("Use --list-fields to see available topics")
        return 1

    # Run the news cycle
    success = asyncio.run(run_news_cycle(topics))
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
