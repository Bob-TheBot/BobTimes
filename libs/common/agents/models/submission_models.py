"""Submission and editorial models for agent workflows."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

# Import types from types module to avoid circular imports
from ..types import (
    EditorialDecisionType,
    NewspaperSection,
    OverallQuality,
    ReporterField,
    StoryPriority,
    StoryStatus,
)
from .story_models import StoryDraft, StoryImage, StorySource


class EditorialFeedback(BaseModel):
    """Feedback provided by an editor."""
    overall_quality: OverallQuality
    clarity_score: int = Field(ge=1, le=10)
    relevance_score: int = Field(ge=1, le=10)
    accuracy_concerns: list[str] = Field(default_factory=list)
    suggested_edits: str | None = None
    additional_sources_needed: bool = False
    fact_check_required: bool = False


class EditorialDecision(BaseModel):
    """Decision made by an editor agent on a story submission."""
    story_id: str
    decision: EditorialDecisionType
    feedback: EditorialFeedback | None = None
    edits: str | None = None
    publish_section: NewspaperSection | None = None
    publish_priority: StoryPriority | None = None
    front_page: bool = False  # Whether this should be a cover story
    decision_time: datetime = Field(default_factory=datetime.now)
    editor_notes: str | None = None


class StorySubmission(BaseModel):
    """Represents a story submission from reporter to editor."""
    submission_id: str | None = None
    draft: StoryDraft
    reporter_id: str
    reporter_field: ReporterField
    submission_time: datetime = Field(default_factory=datetime.now)
    status: StoryStatus = StoryStatus.SUBMITTED
    urgency: StoryPriority = StoryPriority.MEDIUM
    reporter_notes: str | None = None


class PublishedStory(BaseModel):
    """Represents a published story in the newspaper."""
    story_id: str
    title: str
    content: str
    summary: str
    author: str
    reporter_id: str  # Add reporter_id field for compatibility
    field: ReporterField
    section: NewspaperSection
    priority: StoryPriority
    sources: list[StorySource]
    keywords: list[str]
    images: list[StoryImage]
    published_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime | None = None
    word_count: int
    views: int = 0
    editorial_decision: EditorialDecision | None = None

    @classmethod
    def from_submission(
        cls,
        submission: StorySubmission,
        decision: EditorialDecision,
        story_id: str
    ) -> "PublishedStory":
        """Create a published story from a submission and editorial decision."""
        return cls(
            story_id=story_id,
            title=submission.draft.title,
            content=decision.edits or submission.draft.content,
            summary=submission.draft.summary,
            author=submission.reporter_id,
            reporter_id=submission.reporter_id,  # Add reporter_id field
            field=submission.draft.field,
            section=decision.publish_section or NewspaperSection.TECHNOLOGY,
            priority=decision.publish_priority or StoryPriority.MEDIUM,
            sources=submission.draft.sources,
            keywords=submission.draft.keywords,
            images=submission.draft.suggested_images,
            word_count=submission.draft.word_count,
            editorial_decision=decision
        )
