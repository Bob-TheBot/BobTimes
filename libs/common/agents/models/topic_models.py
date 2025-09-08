"""Topic-related models for news cycle orchestration."""

from pydantic import BaseModel, Field

from ..types import ReporterField


class FieldTopicProposals(BaseModel):
    """All proposed topics for a specific field."""
    field: ReporterField
    proposed_topics: list[str]
    source: str = "reporter"  # Who proposed these topics


class SelectedTopic(BaseModel):
    """A topic selected by the editor for story creation."""
    field: ReporterField
    topic: str
    reason: str = ""  # Why this topic was selected
    priority: int = 0  # Priority order for story creation


def _empty_rejected_topics() -> dict[ReporterField, list[str]]:
    """Typed default factory for rejected topics."""
    return {}


class SelectedTopicList(BaseModel):
    """List of selected topics from editor review."""
    reasoning: str  # Overall reasoning for selections
    topics: list[SelectedTopic]
    rejected_topics: dict[ReporterField, list[str]] = Field(default_factory=_empty_rejected_topics)


class TopicSelectionRequest(BaseModel):
    """Request for topic selection by editor."""
    proposals_by_field: dict[ReporterField, list[str]]
    stories_per_field: int
    avoid_duplicates: bool = True
    prioritize_breaking_news: bool = True
