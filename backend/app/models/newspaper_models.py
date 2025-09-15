"""Newspaper models for API responses."""

from typing import Any

from agents.models.submission_models import PublishedStory
from pydantic import BaseModel


class NewspaperContent(BaseModel):
    """Content for displaying in the newspaper UI."""
    title: str
    tagline: str
    stories: list[PublishedStory]
    metadata: dict[str, Any]


class NewspaperContentResponse(BaseModel):
    """API response model for newspaper content."""
    title: str
    tagline: str
    stories: list[PublishedStory]
    metadata: dict[str, Any]
    
    @classmethod
    def from_newspaper_content(cls, content: NewspaperContent) -> "NewspaperContentResponse":
        """Create response from newspaper content."""
        return cls(
            title=content.title,
            tagline=content.tagline,
            stories=content.stories,
            metadata=content.metadata
        )