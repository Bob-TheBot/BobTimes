"""Task-related models for agent workflows."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

# Import types from types module to avoid circular imports
from ..types import EconomicsSubSection, NewspaperSection, QualityThreshold, ReporterField, ScienceSubSection, TaskType, TechnologySubSection
from .submission_models import StorySubmission


class ReporterTask(BaseModel):
    """Task assignment for a reporter agent."""
    name: TaskType  # Type of task (find_topics, write_story)
    field: ReporterField  # Reporter's field of expertise
    sub_section: TechnologySubSection | EconomicsSubSection | ScienceSubSection | None = None  # Optional sub-section within the field
    description: str  # Detailed task description
    guidelines: str | None = None  # Additional instructions
    editor_remarks: str | None = None  # Feedback from editor if revision needed
    # Optional fields for specific task types
    topic: str | None = None  # For write_story tasks
    min_sources: int = 1  # For story tasks (minimum sources to research)
    target_word_count: int = 500  # For write_story tasks
    require_images: bool = False  # For write_story tasks
    # Revision tracking
    is_revision: bool = False  # True if this is a revision based on editor feedback
    original_story_id: str | None = None  # ID of original story being revised
    assigned_at: datetime = Field(default_factory=datetime.now)


# Typed default factory to avoid partially unknown list element types

def _empty_newspaper_sections() -> list[NewspaperSection]:
    return []


class EditorTask(BaseModel):
    """Task assignment for an editor agent."""
    task_id: str | None = None
    submissions: list[StorySubmission]
    max_stories_to_publish: int = 5
    quality_threshold: QualityThreshold = QualityThreshold.MEDIUM
    sections_to_fill: list[NewspaperSection] = Field(default_factory=_empty_newspaper_sections)
    assigned_at: datetime = Field(default_factory=datetime.now)
