"""Story-related models for agent workflows."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# Import types from types module to avoid circular imports
from ..types import EconomicsSubSection, ReporterField, ScienceSubSection, TechnologySubSection


class ImageSourceType(StrEnum):
    """Source type for story images."""
    SEARCH = "search"
    GENERATED = "generated"


class StorySource(BaseModel):
    """Represents a source used in a story."""
    url: str
    title: str
    summary: str | None = None
    content: str | None = None  # Full content (transcript for YouTube, article text for web sources)
    source_type: str | None = None  # Type of source (youtube, web, search, etc.)
    accessed_at: datetime = Field(default_factory=datetime.now)


class StoryImage(BaseModel):
    """Represents an image for a story."""
    url: str
    local_path: str | None = None
    base64_data: str | None = None  # Base64 encoded image data for direct client display
    mime_type: str | None = None   # MIME type for the base64 data (e.g., 'image/jpeg', 'image/png')
    caption: str | None = None
    alt_text: str | None = None
    is_generated: bool = False
    source_type: ImageSourceType = ImageSourceType.SEARCH
    file_size_kb: int | None = None


# Typed default factories to help Pylance resolve concrete list element types

def _empty_story_source_list() -> list[StorySource]:
    return []


def _empty_story_image_list() -> list[StoryImage]:
    return []


class StoryDraft(BaseModel):
    """Represents a draft story created by a reporter agent."""
    title: str
    content: str
    summary: str
    field: ReporterField
    sub_section: TechnologySubSection | EconomicsSubSection | ScienceSubSection | None = None  # Sub-section within the field
    sources: list[StorySource] = Field(default_factory=_empty_story_source_list)
    keywords: list[str] = Field(default_factory=list)
    suggested_images: list[StoryImage] = Field(default_factory=_empty_story_image_list)
    word_count: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    # Revision tracking for editor feedback
    revision_count: int = 0
    editor_feedback: str | None = None
    previous_versions: list[str] = Field(default_factory=list)  # Store previous content versions

    def model_post_init(self, __context: Any) -> None:
        """Calculate word count after initialization if not already set."""
        if self.content and self.word_count == 0:
            self.word_count = len(self.content.split())

    def calculate_word_count(self) -> int:
        """Calculate and update word count."""
        self.word_count = len(self.content.split())
        return self.word_count

    def apply_editor_feedback(self, feedback: str, new_content: str) -> None:
        """Apply editor feedback and create a new revision."""
        # Store previous version
        if self.content:
            self.previous_versions.append(self.content)

        # Apply changes
        self.editor_feedback = feedback
        self.content = new_content
        self.revision_count += 1
        self.calculate_word_count()


class TopicList(BaseModel):
    """List of trending topics found by reporter."""
    reasoning: str  # Why these topics were selected
    topics: list[str]
    field: ReporterField
    sub_section: TechnologySubSection | EconomicsSubSection | ScienceSubSection | None = None  # Sub-section within the field


class ResearchResult(BaseModel):
    """Research findings about a specific topic."""
    field: ReporterField
    sub_section: TechnologySubSection | EconomicsSubSection | ScienceSubSection | None = None  # Sub-section within the field
    facts: list[str]
    sources: list[StorySource] = Field(default_factory=_empty_story_source_list)
    summary: str
    key_points: list[str] = Field(default_factory=list)


class ToolCall(BaseModel):
    """Request to execute a tool."""
    name: str
    parameters: dict[str, Any]


class AgentResponse(BaseModel):
    """Single response type from reporter agent with optional fields."""
    reasoning: str

    # Tool execution request (if agent needs to call a tool)
    tool_call: ToolCall | None = None

    # Final responses (only one will be set based on task type)
    story_draft: StoryDraft | None = None
    topic_list: TopicList | None = None
    research_result: ResearchResult | None = None

    # Metadata
    iteration: int  # Current iteration number
    max_iterations: int  # Maximum allowed iterations
