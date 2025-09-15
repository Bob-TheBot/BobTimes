#!/usr/bin/env python3
"""Tiny end-to-end research task runner.

This script instantiates a Researcher agent and runs a minimal RESEARCH_TOPIC task
end-to-end using the configured tools. If Tavily MCP is configured (tavily.api_key
in libs/common/secrets.yaml), the researcher will use the Tavily-backed tools.

Safe by default: This script only triggers normal LLM/tool usage. Do not run in CI.
"""
from __future__ import annotations

import asyncio
import os
from typing import cast

from core.config_service import ConfigService
from core.logging_service import get_logger

from agents.researcher_agent.researcher_agent_main import ResearcherAgent
from agents.models.task_models import JournalistTask
from agents.types import JournalistField, TaskType, TechnologySubSection
from core.llm_service import ModelSpeed

logger = get_logger(__name__)


async def run_e2e() -> int:
    # Ensure environment defaults
    os.environ.setdefault("APP_ENV", "development")

    config = ConfigService()

    # Configure a tiny research task
    task = JournalistTask(
        agent_type="researcher",  # hint, but orchestration uses ResearcherAgent directly here
        name=TaskType.RESEARCH_TOPIC,
        field=JournalistField.TECHNOLOGY,
        sub_section=TechnologySubSection.TECH_TRENDS,
        description=(
            "Quick E2E research: find 2+ credible sources and key facts about 'GPT-4o' "
            "from the last week. Prefer news and official sources."
        ),
        topic="GPT-4o updates",
        min_sources=2,
        guidelines=(
            "Use search to find fresh sources, scrape 2-3 promising URLs, and compile key facts. "
            "Return ResearchResult once sufficient facts and sources are gathered."
        ),
    )

    # Instantiate a researcher and execute
    researcher = ResearcherAgent(
        field=task.field,
        sub_section=cast(TechnologySubSection, task.sub_section),
        config_service=config,
    )

    print("Starting tiny end-to-end research task...\n")
    result = await researcher.execute_task(task, model_speed=ModelSpeed.FAST)

    # Pretty-print outcome depending on type
    from agents.models.story_models import ResearchResult, TopicList, StoryDraft

    if isinstance(result, ResearchResult):
        print("Research completed:")
        print(f"- Sources: {len(result.sources)}")
        for i, src in enumerate(result.sources[:3], 1):
            print(f"  {i}. {src.title or src.url}")
        print(f"- Facts gathered: {len(result.facts)}")
        for fact in result.facts[:5]:
            print(f"  - {fact}")
        return 0

    if isinstance(result, TopicList):
        print("Received TopicList (unexpected for this task):")
        for i, t in enumerate(result.topics, 1):
            print(f"  {i}. {t}")
        return 0

    if isinstance(result, StoryDraft):  # unlikely for research-only task
        print("Received StoryDraft (unexpected for this task):")
        print(f"Title: {result.title}")
        print(f"Summary: {result.summary}")
        return 0

    print("Task finished without a recognized result type.")
    return 1


def main() -> None:
    try:
        code = asyncio.run(run_e2e())
        raise SystemExit(code)
    except KeyboardInterrupt:
        print("\nCancelled.")
        raise SystemExit(130)


if __name__ == "__main__":
    main()

