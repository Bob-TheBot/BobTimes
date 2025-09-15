"""News cycle models for agent workflows."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

# Import types from types module to avoid circular imports
from ..types import CycleStatus

# Import CyclePerformanceMetrics directly for default factory
from .performance_models import CyclePerformanceMetrics
from .submission_models import EditorialDecision, PublishedStory, StorySubmission
from .task_models import JournalistTask

# Typed default factories to aid static analysis without string annotations

def _empty_reporter_task_list() -> list[JournalistTask]:
    return []


def _empty_story_submission_list() -> list[StorySubmission]:
    return []


def _empty_editorial_decision_list() -> list[EditorialDecision]:
    return []


def _empty_published_story_list() -> list[PublishedStory]:
    return []


class NewsCycle(BaseModel):
    """Represents a complete news cycle."""
    cycle_id: str | None = None
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: datetime | None = None
    reporter_tasks: list[JournalistTask] = Field(default_factory=_empty_reporter_task_list)
    submissions: list[StorySubmission] = Field(default_factory=_empty_story_submission_list)
    editorial_decisions: list[EditorialDecision] = Field(default_factory=_empty_editorial_decision_list)
    published_stories: list[PublishedStory] = Field(default_factory=_empty_published_story_list)
    cycle_status: CycleStatus = CycleStatus.PLANNING
    performance_metrics: CyclePerformanceMetrics = Field(default_factory=CyclePerformanceMetrics)
