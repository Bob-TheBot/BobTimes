"""Pydantic models for content management system.
Defines the structure for stories, images, sections, and main page content.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class ContentStatus(StrEnum):
    """Status of content items."""
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    FEATURED = "featured"


class ContentSection(StrEnum):
    """Available content sections."""
    NEWS = "news"
    SPORTS = "sports"
    TECHNOLOGY = "technology"
    ECONOMICS = "economics"
    POLITICS = "politics"


class Priority(StrEnum):
    """Content priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class ImageModel(BaseModel):
    """Model for images served from backend."""
    id: UUID
    filename: str
    original_filename: str
    url: str
    alt_text: str
    caption: str | None = None
    width: int
    height: int
    file_size: int  # in bytes
    mime_type: str
    created_at: datetime
    updated_at: datetime

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate that the URL is properly formatted."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v

    class Config:
        from_attributes = True


class AuthorModel(BaseModel):
    """Model for story authors."""
    id: UUID
    name: str
    email: str
    bio: str | None = None
    avatar_image: ImageModel | None = None
    specialties: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class StorySource(BaseModel):
    """Model for story sources."""
    url: str
    title: str
    summary: str | None = None
    accessed_at: datetime

    class Config:
        from_attributes = True


class StoryModel(BaseModel):
    """Model for news stories."""
    id: UUID
    title: str
    slug: str
    summary: str
    content: str
    section: ContentSection
    status: ContentStatus
    priority: Priority
    author: AuthorModel
    featured_image: ImageModel | None = None
    gallery_images: list[ImageModel] = Field(default_factory=list)  # type: ignore[misc]
    sources: list[StorySource] = Field(default_factory=list)  # type: ignore[misc]
    tags: list[str] = Field(default_factory=list)
    read_time_minutes: int
    view_count: int = 0
    published_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class SectionSummary(BaseModel):
    """Summary information for a content section."""
    section: ContentSection
    name: str
    description: str
    story_count: int
    latest_update: datetime | None = None


class MainPageContent(BaseModel):
    """Content structure for the main page."""
    breaking_news: list[StoryModel] = Field(default_factory=list)  # type: ignore[misc]
    featured_stories: list[StoryModel] = Field(default_factory=list)  # type: ignore[misc]
    recent_stories: list[StoryModel] = Field(default_factory=list)  # type: ignore[misc]
    section_highlights: dict[ContentSection, list[StoryModel]] = Field(default_factory=dict)  # type: ignore[misc]
    sections_summary: list[SectionSummary] = Field(default_factory=list)  # type: ignore[misc]
    last_updated: datetime


class SectionPageContent(BaseModel):
    """Content structure for individual section pages."""
    section: ContentSection
    section_info: SectionSummary
    featured_stories: list[StoryModel] = Field(default_factory=list)  # type: ignore[misc]
    recent_stories: list[StoryModel] = Field(default_factory=list)  # type: ignore[misc]
    all_stories: list[StoryModel] = Field(default_factory=list)  # type: ignore[misc]
    total_count: int
    page: int = 1
    page_size: int = 20
    last_updated: datetime


# Request/Response models for API endpoints

class StoryCreateRequest(BaseModel):
    """Request model for creating a new story."""
    title: str = Field(..., min_length=1, max_length=200)
    summary: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    section: ContentSection
    priority: Priority = Priority.MEDIUM
    author_id: UUID
    featured_image_id: UUID | None = None
    gallery_image_ids: list[UUID] = Field(default_factory=list)  # type: ignore[misc]
    sources: list[StorySource] = Field(default_factory=list)  # type: ignore[misc]
    tags: list[str] = Field(default_factory=list)
    status: ContentStatus = ContentStatus.DRAFT


class StoryUpdateRequest(BaseModel):
    """Request model for updating a story."""
    title: str | None = Field(None, min_length=1, max_length=200)
    summary: str | None = Field(None, min_length=1, max_length=500)
    content: str | None = Field(None, min_length=1)
    section: ContentSection | None = None
    priority: Priority | None = None
    featured_image_id: UUID | None = None
    gallery_image_ids: list[UUID] | None = None
    sources: list[StorySource] | None = None
    tags: list[str] | None = None
    status: ContentStatus | None = None


class ImageUploadResponse(BaseModel):
    """Response model for image upload."""
    image: ImageModel
    message: str = "Image uploaded successfully"


class ContentResponse(BaseModel):
    """Generic response wrapper for content."""
    success: bool = True
    message: str = "Request completed successfully"
    data: dict[str, Any] | None = None


class PaginationParams(BaseModel):
    """Parameters for pagination."""
    page: int = Field(1, ge=1)
    page_size: int = Field(20, ge=1, le=100)


class ContentFilters(BaseModel):
    """Filters for content queries."""
    section: ContentSection | None = None
    status: ContentStatus | None = None
    priority: Priority | None = None
    author_id: UUID | None = None
    tags: list[str] | None = None
    search: str | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None


