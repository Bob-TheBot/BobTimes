"""SQLAlchemy models for content management system.
Database models for stories, images, authors, and their relationships.
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Table, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from shared_db.models import Base

# Association table for story-image relationships (gallery)
story_images = Table(
    "story_images",
    Base.metadata,
    Column("story_id", UUID(as_uuid=True), ForeignKey("stories.id"), primary_key=True),
    Column("image_id", UUID(as_uuid=True), ForeignKey("images.id"), primary_key=True),
    Column("order_index", Integer, default=0)
)


class Image(Base):
    """Database model for images."""
    __tablename__ = "images"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    alt_text: Mapped[str] = mapped_column(String(500), nullable=False)
    caption: Mapped[str] = mapped_column(Text, nullable=True)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)  # in bytes
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    featured_in_stories: Mapped[list["Story"]] = relationship("Story", back_populates="featured_image")
    gallery_stories: Mapped[list["Story"]] = relationship("Story", secondary=story_images, back_populates="gallery_images")
    author_avatars: Mapped[list["Author"]] = relationship("Author", back_populates="avatar_image")


class Author(Base):
    """Database model for story authors."""
    __tablename__ = "authors"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    bio: Mapped[str] = mapped_column(Text, nullable=True)
    specialties: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    avatar_image_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("images.id"), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    avatar_image: Mapped["Image"] = relationship("Image", back_populates="author_avatars")
    stories: Mapped[list["Story"]] = relationship("Story", back_populates="author")


class Story(Base):
    """Database model for news stories."""
    __tablename__ = "stories"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(250), nullable=False, unique=True)
    summary: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Enums stored as strings
    section: Mapped[str] = mapped_column(String(50), nullable=False)  # ContentSection
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="draft")  # ContentStatus
    priority: Mapped[str] = mapped_column(String(50), nullable=False, default="medium")  # Priority

    # Foreign keys
    author_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("authors.id"), nullable=False)
    featured_image_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("images.id"), nullable=True)

    # Additional fields
    tags: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    sources: Mapped[str] = mapped_column(Text, nullable=True)  # JSON string of source URLs and titles
    read_time_minutes: Mapped[int] = mapped_column(Integer, default=0)
    view_count: Mapped[int] = mapped_column(Integer, default=0)

    # Timestamps
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    author: Mapped["Author"] = relationship("Author", back_populates="stories")
    featured_image: Mapped["Image"] = relationship("Image", back_populates="featured_in_stories")
    gallery_images: Mapped[list["Image"]] = relationship("Image", secondary=story_images, back_populates="gallery_stories")

    def __repr__(self) -> str:
        return f"<Story(id={self.id}, title='{self.title}', section='{self.section}')>"


class ContentMetrics(Base):
    """Database model for tracking content metrics."""
    __tablename__ = "content_metrics"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    story_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("stories.id"), nullable=False)
    metric_type: Mapped[str] = mapped_column(String(50), nullable=False)  # 'view', 'share', 'like', etc.
    metric_value: Mapped[int] = mapped_column(Integer, default=0)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    story: Mapped["Story"] = relationship("Story")


class ContentCache(Base):
    """Database model for caching computed content structures."""
    __tablename__ = "content_cache"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    cache_key: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    cache_data: Mapped[str] = mapped_column(Text, nullable=False)  # JSON string
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
