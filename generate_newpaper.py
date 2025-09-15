#!/usr/bin/env python3
"""Minimal runner to generate a single-topic newspaper issue.

This script initializes one reporter (based on a single topic) and one editor,
then runs one complete news cycle. It is intentionally simple to make local
manual testing and debugging easy.

Run it directly or use the VS Code launch configuration we add in .vscode/launch.json.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Ensure Python can find shared libs and backend app modules when run from repo root
REPO_ROOT = Path(__file__).resolve().parent
COMMON_PATH = REPO_ROOT / "libs" / "common"
BACKEND_PATH = REPO_ROOT / "backend"
if str(COMMON_PATH) not in sys.path:
    sys.path.insert(0, str(COMMON_PATH))
if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))

# Now safe to import shared agents/types and backend services
from agents.types import (
    JournalistField, 
    TechnologySubSection, 
    EconomicsSubSection,
    ScienceSubSection,
    FieldTopicRequest
)
from core.config_service import ConfigService
from core.logging_service import get_logger
from app.services.newspaper_service import NewspaperService

logger = get_logger(__name__)


async def generate_all_issues() -> int:
    """Generate newspaper issues for all topics and subsections.
    
    This will loop through all fields and their subsections to generate 
    comprehensive news coverage across all areas.
    
    Returns 0 on success, 1 on failure.
    """
    # Define all field-subsection combinations
    field_subsection_map = {
        JournalistField.TECHNOLOGY: [
            TechnologySubSection.AI_TOOLS,
            TechnologySubSection.TECH_TRENDS,
            TechnologySubSection.QUANTUM_COMPUTING,
            TechnologySubSection.GENERAL_TECH,
            TechnologySubSection.MAJOR_DEALS
        ],
        JournalistField.ECONOMICS: [
            EconomicsSubSection.CRYPTO,
            EconomicsSubSection.US_STOCK_MARKET,
            EconomicsSubSection.GENERAL_NEWS,
            EconomicsSubSection.ISRAEL_ECONOMICS,
            EconomicsSubSection.EXITS,
            EconomicsSubSection.UPCOMING_IPOS,
            EconomicsSubSection.MAJOR_TRANSACTIONS
        ],
        JournalistField.SCIENCE: [
            ScienceSubSection.NEW_RESEARCH,
            ScienceSubSection.BIOLOGY,
            ScienceSubSection.CHEMISTRY,
            ScienceSubSection.SPACE,
            ScienceSubSection.PHYSICS
        ]
    }

    field_subsection_map = {
        JournalistField.TECHNOLOGY: [
            TechnologySubSection.AI_TOOLS
        ]
    }

    total_success = 0
    total_attempted = 0
    
    print("\n🗞️  Starting comprehensive newspaper generation for all fields and subsections...")
    print(f"Total combinations to generate: {sum(len(subsections) for subsections in field_subsection_map.values())}")
    
    for field, subsections in field_subsection_map.items():
        print(f"\n📰 Processing {field.value.upper()} field...")
        
        for subsection in subsections:
            total_attempted += 1
            print(f"\n🔄 Generating issue {total_attempted}: {field.value} - {subsection.value}")
            
            try:
                result = await generate_single_issue(field, subsection)
                if result == 0:
                    total_success += 1
                    print(f"✅ Successfully generated issue for {field.value} - {subsection.value}")
                else:
                    print(f"❌ Failed to generate issue for {field.value} - {subsection.value}")
            except Exception as e:
                logger.exception(f"Error generating issue for {field.value} - {subsection.value}: {e}")
                print(f"💥 Exception during {field.value} - {subsection.value}: {e}")
    
    # Summary
    print(f"\n🎯 GENERATION COMPLETE!")
    print(f"📊 Success rate: {total_success}/{total_attempted} ({(total_success/total_attempted*100):.1f}%)")
    
    return 0 if total_success == total_attempted else 1


async def generate_single_issue(field: JournalistField, sub_section: TechnologySubSection | EconomicsSubSection | ScienceSubSection | None = None) -> int:
    """Generate one newspaper issue for a single topic with one reporter and one editor.
    
    Args:
        field: The reporter field (TECHNOLOGY, ECONOMICS, SCIENCE, BUSINESS)
        sub_section: The specific sub-section within the field
    
    This will:
    1. Create one reporter for the specified field
    2. Have the reporter suggest topics
    3. Editor selects one topic
    4. Reporter writes one story
    5. Editor reviews and publishes

    Returns 0 on success, 1 on failure.
    """
    try:
        field_requests = [FieldTopicRequest(field=field, sub_section=sub_section)]

        # Initialize core services
        config_service = ConfigService()
        
        # Create LLM service for image generation
        from app.dependencies import get_default_llm_service
        llm_service = get_default_llm_service(config_service)
        
        newspaper_service = NewspaperService(config_service, llm_service)


        # Run the news cycle with exactly one topic field/sub-section and one story
        # This ensures we get exactly one reporter working on one story with specific focus
        cycle = await newspaper_service.run_news_cycle(
            field_requests=field_requests,
            stories_per_field=1,  # Explicitly set to 1 story
            sequential=True  # Generate sequentially for better control
        )

        # Present a detailed summary in the console
        print(f"\n================ SINGLE-TOPIC NEWS CYCLE ================")
        print(f"Field: {field.value}")
        print(f"Sub-section: {sub_section.value if sub_section else 'General'}")
        print(f"Cycle ID: {cycle.cycle_id}")
        print(f"Status: {cycle.cycle_status.value}")
        print(f"Total Submissions: {len(cycle.submissions)}")
        print(f"Editorial Decisions: {len(cycle.editorial_decisions)}")
        print(f"Stories Published: {len(cycle.published_stories)}")
        
        # Show submission details
        if cycle.submissions:
            print("\nSubmissions:")
            for sub in cycle.submissions:
                print(f"  - {sub.draft.title[:60]}... (Reporter: {sub.reporter_id})")
        
        # Show editorial decision details
        if cycle.editorial_decisions:
            print("\nEditorial Decisions:")
            for decision in cycle.editorial_decisions:
                print(f"  - Story {decision.story_id}: {decision.decision.value}")
                if decision.editor_notes:
                    print(f"    Notes: {decision.editor_notes[:100]}...")
        
        # Show published story details
        if cycle.published_stories:
            print("\nPublished Stories:")
            for story in cycle.published_stories:
                print(f"  Title: {story.title}")
                print(f"  Field: {story.field.value}")
                print(f"  Reporter: {story.reporter_id}")
                print(f"  Word Count: {story.word_count}")
                print(f"  Summary: {story.summary[:150]}...")
                if story.sources:
                    print(f"  Sources: {len(story.sources)} sources")
        else:
            print("\nNo stories were published.")
            
        print("=========================================================\n")

        return 0
    except Exception as e:  # pragma: no cover - quick manual runner
        logger.exception(f"Minimal news cycle failed: {e}")
        print(f"\nERROR: {e}")
        return 1


def main() -> None:
    """Main entry point. 
    
    By default generates all issues for all fields and subsections.
    Can be modified to generate single issues for testing.
    """
    # Default to development if not set
    os.environ.setdefault("APP_ENV", "development")
    
    # Generate all issues by default
    exit_code = asyncio.run(generate_all_issues())
    
    # Uncomment to test single issue generation:
    # exit_code = asyncio.run(generate_single_issue(ReporterField.TECHNOLOGY, TechnologySubSection.AI_TOOLS))
    
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

